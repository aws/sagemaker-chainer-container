# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import os
import json

import numpy as np
import chainer
from chainer import training
from chainer import serializers
from chainer.training import extensions

import nets
from nlp_utils import convert_seq


# ------------------------------------------------------------ #
# Training methods                                             #
# ------------------------------------------------------------ #

def train(hyperparameters, num_gpus, output_data_dir, channel_input_dirs):
    """
    This function is called by the Chainer container during training when running on SageMaker with
    values populated by the training environment.

    Args:
        hyperparameters (dict): map of hyperparameters given to the training job.
        num_gpus (int): number of gpus available to the container, determined by instance type.
        output_data_dir (str): path to the directory to write output artifacts to
        channel_input_dirs (dict): Dictionary mapping input channel names to local filesystem paths

    Returns:
        a trained Chainer model

    For more on `train`, please visit the sagemaker-python-sdk repository:
    https://github.com/aws/sagemaker-python-sdk

    For more on the Chainer container, please visit the sagemaker-chainer-containers repository:
    https://github.com/aws/sagemaker-chainer-containers
    """
    train_data = np.load(os.path.join(channel_input_dirs['train'], 'train_data.npy'))
    train_labels = np.load(os.path.join(channel_input_dirs['train'], 'train_labels.npy'))

    test_data = np.load(os.path.join(channel_input_dirs['test'], 'test_data.npy'))
    test_labels = np.load(os.path.join(channel_input_dirs['test'], 'test_labels.npy'))

    vocab = np.load(os.path.join(channel_input_dirs['vocab'], 'vocab.npy')).tolist()

    train = chainer.datasets.TupleDataset(train_data, train_labels)
    test = chainer.datasets.TupleDataset(test_data, test_labels)

    print('# train data: {}'.format(len(train)))
    print('# test  data: {}'.format(len(test)))
    print('# vocab: {}'.format(len(vocab)))
    num_classes = len(set([int(d[1]) for d in train]))
    print('# class: {}'.format(num_classes))

    batch_size = hyperparameters.get('batch_size', 64)
    epochs = hyperparameters.get('epochs', 30)
    dropout = hyperparameters.get('dropout', 0.4)
    num_layers = hyperparameters.get('num_layers', 1)
    num_units = hyperparameters.get('num_units', 300)
    model_type = hyperparameters.get('model', 'rnn')

    print('# Minibatch-size: {}'.format(batch_size))
    print('# epoch: {}'.format(epochs))
    print('# Dropout: {}'.format(dropout))
    print('# Layers: {}'.format(num_layers))
    print('# Units: {}'.format(num_units))

    # Setup a model
    if model_type == 'rnn':
        Encoder = nets.RNNEncoder
    elif model_type == 'cnn':
        Encoder = nets.CNNEncoder
    elif model_type == 'bow':
        Encoder = nets.BOWMLPEncoder
    else:
        raise ValueError('model_type must be "rnn", "cnn", or "bow"')

    encoder = Encoder(n_layers=num_layers, n_vocab=len(vocab), n_units=num_units, dropout=dropout)
    model = nets.TextClassifier(encoder, num_classes)

    if num_gpus >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(0).use()

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

    train_iter = chainer.iterators.SerialIterator(train, batch_size)
    test_iter = chainer.iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)

    # Set up a trainer
    device = 0 if num_gpus > 0 else -1  # -1 indicates CPU, 0 indicates first GPU device.
    if num_gpus > 0:
        updater = training.updater.ParallelUpdater(
            train_iter,
            optimizer,
            converter=convert_seq,
            # The device of the name 'main' is used as a "master", while others are
            # used as slaves. Names other than 'main' are arbitrary.
            devices={('main' if device == 0 else str(device)): device for device in range(num_gpus)},
        )
    else:
        updater = training.updater.StandardUpdater(train_iter, optimizer, converter=convert_seq, device=device)

    trainer = training.Trainer(updater, (epochs, 'epoch'), out=output_data_dir)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, converter=convert_seq, device=0))

    # Take a best snapshot.
    record_trigger = training.triggers.MaxValueTrigger('validation/main/accuracy', (1, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, 'best_model.npz'), trigger=record_trigger)

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # Save additional model settings, which will be used to reconstruct the model during hosting
    model_setup = {}
    model_setup['num_classes'] = num_classes
    model_setup['model_type'] = model_type
    model_setup['num_layers'] = num_layers
    model_setup['num_units'] = num_units
    model_setup['dropout'] = dropout

    # Run the training
    trainer.run()

    # SageMaker saves the return value of train() in the `save` function in the resulting
    # model artifact model.tar.gz, and the contents of `output_data_dir` in the output
    # artifact output.tar.gz.

    # return the best model
    best_model = np.load(os.path.join(output_data_dir, 'best_model.npz'))

    # remove the best model from output artifacts (since it will be saved as a model artifact)
    os.remove(os.path.join(output_data_dir, 'best_model.npz'))
    return best_model, vocab, model_setup


def save(model, model_dir):
    """
    Writes model artifacts to `model_dir`.

    During hosting, `model_fn` will load these artifacts to reconstruct the model for inference.

    Args:
        model: the return value of `train` -- in this case, a tuple of the trained model,
               vocab dict, and model_setup dict.
        model_dir: the directory to save model artifacts to, the contents of which are tarred,
               zipped, and uploaded to S3.

    For more on `save`, please visit the sagemaker-python-sdk repository:
    https://github.com/aws/sagemaker-python-sdk

    For more on the Chainer container, please visit the sagemaker-chainer-containers repository:
    https://github.com/aws/sagemaker-chainer-containers
    """
    trained_model, vocab, model_setup = model

    serializers.save_npz(os.path.join(model_dir, 'model.npz'), trained_model)
    with open(os.path.join(model_dir, 'vocab.json'), 'w') as f:
        json.dump(vocab, f)
    with open(os.path.join(model_dir, 'args.json'), 'w') as f:
        json.dump(model_setup, f)


# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #

def model_fn(model_dir):
    model_path = os.path.join(model_dir, 'model.npz')

    vocab_path = os.path.join(model_dir, 'vocab.json')
    model_setup_path = os.path.join(model_dir, 'args.json')
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    with open(model_setup_path, 'r') as f:
        model_setup = json.load(f)

    model_type =
    if model_type == 'rnn':
        Encoder = nets.RNNEncoder
    elif model_type == 'cnn':
        Encoder = nets.CNNEncoder
    elif model_type == 'bow':
        Encoder = nets.BOWMLPEncoder


def transform_fn(model, data, input_content_type, output_content_type):
    pass

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

from __future__ import print_function, absolute_import

import os

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import serializers
from chainer.training import extensions

import net


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

    train_data = np.load(os.path.join(channel_input_dirs['train'], 'train.npz'))['data']
    train_labels = np.load(os.path.join(channel_input_dirs['train'], 'train.npz'))['labels']

    test_data = np.load(os.path.join(channel_input_dirs['test'], 'test.npz'))['data']
    test_labels = np.load(os.path.join(channel_input_dirs['test'], 'test.npz'))['labels']

    train = chainer.datasets.TupleDataset(train_data, train_labels)
    test = chainer.datasets.TupleDataset(test_data, test_labels)

    # retrieve the hyperparameters we set in notebook (with some defaults)
    batch_size = hyperparameters.get('batch_size', 64)
    epochs = hyperparameters.get('epochs', 300)
    learning_rate = hyperparameters.get('learning_rate', 0.05)
    num_loaders = hyperparameters.get('num_loaders', None) # defaults to num_cpus

    print('# Minibatch-size: {}'.format(batch_size))
    print('# epoch: {}'.format(epochs))

    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(net.VGG(10))

    optimizer = chainer.optimizers.MomentumSGD(learning_rate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    # Set up a trainer
    device = 0 if num_gpus > 0 else -1  # -1 indicates CPU, 0 indicates first GPU device.
    if num_gpus > 1:
        devices = range(num_gpus)
        train_iters = [chainer.iterators.MultiprocessIterator(i, batch_size, n_processes=num_gpus) \
                    for i in chainer.datasets.split_dataset_n_random(train, len(devices))]
        test_iter = chainer.iterators.MultiprocessIterator(test, batch_size, repeat=False, n_processes=num_gpus)
        updater = training.updaters.MultiprocessParallelUpdater(train_iters, optimizer, devices=range(num_gpus))
    else:
        train_iter = chainer.iterators.MultiprocessIterator(train, batch_size)
        test_iter = chainer.iterators.MultiprocessIterator(test, batch_size, repeat=False)
        updater = training.updater.StandardUpdater(train_iter, optimizer, device=device)

    stop_trigger = (epochs, 'epoch')
    trainer = training.Trainer(updater, stop_trigger, out=output_data_dir)
    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=device))

    # Reduce the learning rate by half every 25 epochs.
    trainer.extend(extensions.ExponentialShift('lr', 0.5), trigger=(25, 'epoch'))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # Run the training

    trainer.run()

    return model


def model_fn(model_dir):
    """
    This function is called by the Chainer container during hosting when running on SageMaker with
    values populated by the hosting environment.
    
    By default, the Chainer container saves models as .npz files, with the name 'model.npz'. In
    your training script, you can override this behavior by implementing a function with
    signature `save(model, model_dir)`.

    Args:
        model_dir (str): path to the directory containing the saved model artifacts

    Returns:
        a loaded Chainer model
    
    For more on `model_fn` and `save`, please visit the sagemaker-python-sdk repository:
    https://github.com/aws/sagemaker-python-sdk
    
    For more on the Chainer container, please visit the sagemaker-chainer-containers repository:
    https://github.com/aws/sagemaker-chainer-containers
    """
    chainer.config.train = False
    model = L.Classifier(net.VGG(10))
    serializers.load_npz(os.path.join(model_dir, 'model.npz'), model)
    return model.predictor

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
import chainermn
from chainer import initializers
from chainer import serializers
from chainer import training
from chainer.training import extensions

import net


def train(hyperparameters, num_gpus, output_data_dir, channel_input_dirs, model_dir):
    """
    This function is called by the Chainer container during training when running on SageMaker with
    values populated by the training environment.
    
    When running in distributed mode, this function is called with `mpirun`, spawning (by default) one
    process per GPU (when running with GPU instances), or one process per host (when running with
    CPU instances). Ranks are initialized in the expected way -- comm.intra_rank refers to the rank of the
    process on an instance, and comm.inter_rank refers to the rank of the instance.

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
    batch_size = hyperparameters.get('batch_size', 256)
    epochs = hyperparameters.get('epochs', 50)
    learning_rate = hyperparameters.get('learning_rate', 0.05)
    communicator = hyperparameters.get('communicator', 'pure_nccl' if num_gpus > 0 else 'naive')

    comm = chainermn.create_communicator(communicator)

    # comm.inter_rank gives the rank of the node. This should only print on one node.
    if comm.inter_rank == 0:
        print('# Minibatch-size: {}'.format(batch_size))
        print('# epoch: {}'.format(epochs))
        print('# learning rate: {}'.format(learning_rate))
        print('# communicator: {}'.format(communicator))

    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.

    model = L.Classifier(net.VGG(10))
    # Make a specified GPU current

    # comm.intra_rank gives the rank of the process on a given node.
    device = comm.intra_rank if num_gpus > 0 else -1
    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()

    optimizer = chainermn.create_multi_node_optimizer(chainer.optimizers.MomentumSGD(learning_rate), comm)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))
    
    num_loaders = 2
    train_iter = chainer.iterators.MultiprocessIterator(train, batch_size, n_processes=num_loaders)
    test_iter = chainer.iterators.MultiprocessIterator(test, batch_size, repeat=False, n_processes=num_loaders)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (epochs, 'epoch'), out=output_data_dir)

    # Evaluate the model with the test dataset for each epoch

    evaluator = extensions.Evaluator(test_iter, model, device=device)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator)

    # Reduce the learning rate by half every 25 epochs.
    trainer.extend(extensions.ExponentialShift('lr', 0.5), trigger=(25, 'epoch'))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())
    if comm.rank == 0:
        if extensions.PlotReport.available():
            trainer.extend(
                extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                      'epoch', file_name='loss.png'))
            trainer.extend(
                extensions.PlotReport(
                    ['main/accuracy', 'validation/main/accuracy'],
                    'epoch', file_name='accuracy.png'))

        trainer.extend(extensions.dump_graph('main/loss'))

        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Run the training
    serializers.save_npz(os.path.join(model_dir, 'model.npz'), model)
    
    trainer.run()
    return model


def model_fn(model_dir):
    """
    This function is called by the Chainer container during hosting when running on SageMaker with
    values populated by the hosting environment.
    
    This function loads models written during training into `model_dir`.
    

    Args:
        model_dir (str): path to the directory containing the saved model artifacts

    Returns:
        a loaded Chainer model
    
    For more on `model_fn`, please visit the sagemaker-python-sdk repository:
    https://github.com/aws/sagemaker-python-sdk
    
    For more on the Chainer container, please visit the sagemaker-chainer-containers repository:
    https://github.com/aws/sagemaker-chainer-containers
    """
    chainer.config.train = False
    model = L.Classifier(net.VGG(10))
    serializers.load_npz(os.path.join(model_dir, 'model.npz'), model)
    return model.predictor
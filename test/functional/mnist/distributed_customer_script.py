from __future__ import print_function

import os

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import serializers
from chainer.datasets import tuple_dataset

from mpi4py import MPI
import chainermn


class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def _preprocess_mnist(raw, withlabel, ndim, scale, image_dtype, label_dtype,
                     rgb_format):
    images = raw['x']
    if ndim == 2:
        images = images.reshape(-1, 28, 28)
    elif ndim == 3:
        images = images.reshape(-1, 1, 28, 28)
        if rgb_format:
            images = np.broadcast_to(images,
                                        (len(images), 3) + images.shape[2:])
    elif ndim != 1:
        raise ValueError('invalid ndim for MNIST dataset')
    images = images.astype(image_dtype)
    images *= scale / 255.

    if withlabel:
        labels = raw['y'].astype(label_dtype)
        return tuple_dataset.TupleDataset(images, labels)
    else:
        return images


def train(channel_input_dirs, hyperparameters, num_gpus, output_data_dir):
    batch_size = hyperparameters.get('batch_size', 100)
    epochs = hyperparameters.get('epochs', 20)
    frequency = hyperparameters.get('frequency', epochs)
    units = hyperparameters.get('unit', 1000)
    communicator = hyperparameters.get('communicator', 'naive' if num_gpus == 0 else 'hierarchical')

    comm = chainermn.create_communicator(communicator)
    device = comm.intra_rank if num_gpus > 0 else -1

    if comm.mpi_comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(MPI.COMM_WORLD.Get_size()))
        print('Using {} communicator'.format(comm))
        print('Num unit: {}'.format(units))
        print('Num Minibatch-size: {}'.format(batch_size))
        print('Num epoch: {}'.format(epochs))
        print('==========================================')

    model = L.Classifier(MLP(units, 10))
    if device >= 0:
        chainer.cuda.get_device(device).use()
        model.to_gpu()

    # Create a multi node optimizer from a standard Chainer optimizer.
    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.Adam(), comm)
    optimizer.setup(model)

    train_file = np.load(os.path.join(channel_input_dirs['training'], 'train.npz'))
    test_file = np.load(os.path.join(channel_input_dirs['training'], 'test.npz'))

    preprocess_mnist_options = {'withlabel': True,
                           'ndim': 1,
                           'scale': 1.,
                           'image_dtype': np.float32,
                           'label_dtype': np.int32,
                           'rgb_format': False}

    # Assumption: dataset is sharded, so no need to scatter.
    train = _preprocess_mnist(train_file, **preprocess_mnist_options)
    test = _preprocess_mnist(test_file, **preprocess_mnist_options)

    # Split and distribute the dataset. Only worker 0 loads the whole dataset.
    # Datasets of worker 0 are evenly split and distributed to all workers.
    '''
    if comm.rank == 0:
        train, test = chainer.datasets.get_mnist()
    else:
        train, test = None, None
    # TODO: fully replicated dataset?
    train = chainermn.scatter_dataset(train, comm, shuffle=True)
    test = chainermn.scatter_dataset(test, comm, shuffle=True)
    '''

    train_iter = chainer.iterators.SerialIterator(train, batch_size)
    test_iter = chainer.iterators.SerialIterator(test, batch_size,
                                                 repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (epochs, 'epoch'), out=output_data_dir)

    # Create a multi node evaluator from a standard Chainer evaluator.
    evaluator = extensions.Evaluator(test_iter, model, device=device)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator)

    # Some display and output extensions are necessary only for one worker.
    # (Otherwise, there would just be repeated outputs.)
    if comm.rank == 0:
        if extensions.PlotReport.available():
            trainer.extend(
                extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                      'epoch', file_name='loss.png'))
            trainer.extend(
                extensions.PlotReport(
                    ['main/accuracy', 'validation/main/accuracy'],
                    'epoch', file_name='accuracy.png'))
        trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
        trainer.extend(extensions.dump_graph('main/loss'))
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
        trainer.extend(extensions.ProgressBar())

    trainer.run()
    return model


def model_fn(model_dir):
    model = MLP()
    serializers.load_npz(os.path.join(model_dir, 'model.npz'), model)
    return model

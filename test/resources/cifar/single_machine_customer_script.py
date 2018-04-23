from __future__ import print_function

import os

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import serializers
from chainer.training import extensions


class BottleNeck(chainer.Chain):

    def __init__(self, n_in, n_mid, n_out, stride=1, use_conv=False):
        w = chainer.initializers.HeNormal()
        super(BottleNeck, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(n_in, n_mid, 1, stride, 0, True, w)
            self.bn1 = L.BatchNormalization(n_mid)
            self.conv2 = L.Convolution2D(n_mid, n_mid, 3, 1, 1, True, w)
            self.bn2 = L.BatchNormalization(n_mid)
            self.conv3 = L.Convolution2D(n_mid, n_out, 1, 1, 0, True, w)
            self.bn3 = L.BatchNormalization(n_out)
            if use_conv:
                self.conv4 = L.Convolution2D(
                    n_in, n_out, 1, stride, 0, True, w)
                self.bn4 = L.BatchNormalization(n_out)
        self.use_conv = use_conv

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return h + self.bn4(self.conv4(x)) if self.use_conv else h + x


class Block(chainer.ChainList):

    def __init__(self, n_in, n_mid, n_out, n_bottlenecks, stride=2):
        super(Block, self).__init__()
        self.add_link(BottleNeck(n_in, n_mid, n_out, stride, True))
        for _ in range(n_bottlenecks - 1):
            self.add_link(BottleNeck(n_out, n_mid, n_out))

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x


class ResNet(chainer.Chain):

    def __init__(self, n_class=10, n_blocks=[3, 4, 6, 3]):
        super(ResNet, self).__init__()
        w = chainer.initializers.HeNormal()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 3, 1, 0, True, w)
            self.bn2 = L.BatchNormalization(64)
            self.res3 = Block(64, 64, 256, n_blocks[0], 1)
            self.res4 = Block(256, 128, 512, n_blocks[1], 2)
            self.res5 = Block(512, 256, 1024, n_blocks[2], 2)
            self.res6 = Block(1024, 512, 2048, n_blocks[3], 2)
            self.fc7 = L.Linear(None, n_class)

    def __call__(self, x):
        h = F.relu(self.bn2(self.conv1(x)))
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = self.res6(h)
        h = F.average_pooling_2d(h, h.shape[2:])
        h = self.fc7(h)
        return h


class ResNet50(ResNet):

    def __init__(self, n_class=10):
        super(ResNet50, self).__init__(n_class, [3, 4, 6, 3])


def train(hyperparameters, num_gpus, output_data_dir, channel_input_dirs):
    train_data = np.load(os.path.join(channel_input_dirs['train'], 'cifar10-train-data.npz'))['arr_0']
    train_labels = np.load(os.path.join(channel_input_dirs['train'], 'cifar10-train-labels.npz'))['arr_0']

    test_data = np.load(os.path.join(channel_input_dirs['test'], 'cifar10-test-data.npz'))['arr_0']
    test_labels = np.load(os.path.join(channel_input_dirs['test'], 'cifar10-test-labels.npz'))['arr_0']

    train = chainer.datasets.TupleDataset(train_data, train_labels)
    test = chainer.datasets.TupleDataset(test_data, test_labels)

    batch_size = hyperparameters.get('batch_size', 64)
    epochs = hyperparameters.get('epochs', 300)
    learning_rate = hyperparameters.get('learning_rate', 0.05)

    print('# Minibatch-size: {}'.format(batch_size))
    print('# epoch: {}'.format(epochs))

    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(ResNet50(10))

    # Make a specified GPU current
    if num_gpus > 0:
        chainer.cuda.get_device_from_id(0).use()

    optimizer = chainer.optimizers.MomentumSGD(learning_rate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    train_iter = chainer.iterators.SerialIterator(train, batch_size)
    test_iter = chainer.iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)

    # Set up a trainer
    device = 0 if num_gpus > 0 else -1  # -1 indicates CPU, 0 indicates first GPU device.
    if num_gpus > 0:
        updater = training.updater.ParallelUpdater(
            train_iter,
            optimizer,
            # The device of the name 'main' is used as a "master", while others are
            # used as slaves. Names other than 'main' are arbitrary.
            devices={('main' if device == 0 else str(device)): device for device in range(num_gpus)},
        )
    else:
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

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(), trigger=(epochs, 'epoch'))

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
    model = L.Classifier(ResNet50(10))
    serializers.load_npz(os.path.join(model_dir, 'model.npz'), model)
    return model.predictor

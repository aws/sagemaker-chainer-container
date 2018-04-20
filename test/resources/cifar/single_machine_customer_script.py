from __future__ import print_function

import os

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer import serializers
from chainer.training import extensions
from chainer.datasets import get_cifar10

class Block(chainer.Chain):

    """A convolution, batch norm, ReLU block.
    A block in a feedforward network that performs a
    convolution followed by batch normalization followed
    by a ReLU activation.
    For the convolution operation, a square filter size is used.
    Args:
        out_channels (int): The number of output channels.
        ksize (int): The size of the filter is ksize x ksize.
        pad (int): The padding to use for the convolution.
    """

    def __init__(self, out_channels, ksize, pad=1):
        super(Block, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, out_channels, ksize, pad=pad,
                                        nobias=True)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn(h)
        return F.relu(h)


class VGG(chainer.Chain):

    """A VGG-style network for very small images.
    This model is based on the VGG-style model from
    http://torch.ch/blog/2015/07/30/cifar.html
    which is based on the network architecture from the paper:
    https://arxiv.org/pdf/1409.1556v6.pdf
    This model is intended to be used with either RGB or greyscale input
    images that are of size 32x32 pixels, such as those in the CIFAR10
    and CIFAR100 datasets.
    On CIFAR10, it achieves approximately 89% accuracy on the test set with
    no data augmentation.
    On CIFAR100, it achieves approximately 63% accuracy on the test set with
    no data augmentation.
    Args:
        class_labels (int): The number of class labels.
    """

    def __init__(self, class_labels=10):
        super(VGG, self).__init__()
        with self.init_scope():
            self.block1_1 = Block(64, 3)
            self.block1_2 = Block(64, 3)
            self.block2_1 = Block(128, 3)
            self.block2_2 = Block(128, 3)
            self.block3_1 = Block(256, 3)
            self.block3_2 = Block(256, 3)
            self.block3_3 = Block(256, 3)
            self.block4_1 = Block(512, 3)
            self.block4_2 = Block(512, 3)
            self.block4_3 = Block(512, 3)
            self.block5_1 = Block(512, 3)
            self.block5_2 = Block(512, 3)
            self.block5_3 = Block(512, 3)
            self.fc1 = L.Linear(None, 512, nobias=True)
            self.bn_fc1 = L.BatchNormalization(512)
            self.fc2 = L.Linear(None, class_labels, nobias=True)

    def __call__(self, x):
        # 64 channel blocks:
        h = self.block1_1(x)
        h = F.dropout(h, ratio=0.3)
        h = self.block1_2(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 128 channel blocks:
        h = self.block2_1(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block2_2(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 256 channel blocks:
        h = self.block3_1(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block3_2(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block3_3(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 512 channel blocks:
        h = self.block4_1(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block4_2(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block4_3(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 512 channel blocks:
        h = self.block5_1(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block5_2(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block5_3(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.dropout(h, ratio=0.5)
        h = self.fc1(h)
        h = self.bn_fc1(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.5)
        return self.fc2(h)


def train(hyperparameters, num_gpus, output_data_dir):
    batch_size = hyperparameters.get('batch_size', 64)
    epochs = hyperparameters.get('epochs', 300)
    learning_rate = hyperparameters.get('learning_rate', 0.05)

    print('# Minibatch-size: {}'.format(batch_size))
    print('# epoch: {}'.format(epochs))

    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.

    class_labels = 10
    train, test = get_cifar10()
    train = train[:100]
    test = test[:100]

    model = L.Classifier(VGG(class_labels))

    # Make a specified GPU current
    if num_gpus > 0:
        chainer.cuda.get_device_from_id(0).use()

    optimizer = chainer.optimizers.MomentumSGD(learning_rate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    train_iter = chainer.iterators.SerialIterator(train, batch_size)
    test_iter = chainer.iterators.SerialIterator(test, batch_size,
                                                 repeat=False, shuffle=False)

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
    model = L.Classifier(VGG(10))
    serializers.load_npz(os.path.join(model_dir, 'model.npz'), model)
    return model.predictor

if __name__=="__main__":
    model = train({'epochs': 1}, 8, 'out')
    print(model)

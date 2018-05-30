# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import argparse
import os

import chainer
from chainer import serializers
from chainer.dataset import convert
from chainer.datasets import tuple_dataset
import chainer.functions as F
import chainer.links as L
import numpy as np
import sagemaker_containers


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


def _preprocess_mnist(raw, withlabel, ndim, scale, image_dtype, label_dtype, rgb_format):
    images = raw['x']
    if ndim == 2:
        images = images.reshape(-1, 28, 28)
    elif ndim == 3:
        images = images.reshape(-1, 1, 28, 28)
        if rgb_format:
            images = np.broadcast_to(images, (len(images), 3) + images.shape[2:])
    elif ndim != 1:
        raise ValueError('invalid ndim for MNIST dataset')
    images = images.astype(image_dtype)
    images *= scale / 255.

    if withlabel:
        labels = raw['y'].astype(label_dtype)
        return tuple_dataset.TupleDataset(images, labels)
    else:
        return images


if __name__ == '__main__':
    env = sagemaker_containers.training_env()

    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--units', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--model-dir', type=str, default=env.model_dir)

    parser.add_argument('--train', type=str, default=env.channel_input_dirs['train'])
    parser.add_argument('--test', type=str, default=env.channel_input_dirs['test'])

    parser.add_argument('--num-gpus', type=int, default=env.num_gpus)

    args = parser.parse_args()

    train_file = np.load(os.path.join(args.train, 'train.npz'))
    test_file = np.load(os.path.join(args.test, 'test.npz'))

    preprocess_mnist_options = {
        'withlabel': True,
        'ndim': 1,
        'scale': 1.,
        'image_dtype': np.float32,
        'label_dtype': np.int32,
        'rgb_format': False
    }

    train = _preprocess_mnist(train_file, **preprocess_mnist_options)
    test = _preprocess_mnist(test_file, **preprocess_mnist_options)

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(MLP(args.units, 10))

    if args.num_gpus > 0:
        chainer.cuda.get_device_from_id(0).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    train_iter = chainer.iterators.SerialIterator(train, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(
        test, args.batch_size, repeat=False, shuffle=False)

    sum_accuracy = 0
    sum_loss = 0

    train_count = len(train)
    test_count = len(train)

    device = 0 if args.num_gpus > 0 else -1  # -1 indicates CPU, 0 indicates first GPU device.

    while train_iter.epoch < args.epochs:
        batch = train_iter.next()
        x_array, t_array = convert.concat_examples(batch, device)
        x = chainer.Variable(x_array)
        t = chainer.Variable(t_array)
        optimizer.update(model, x, t)
        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

        if train_iter.is_new_epoch:
            print('epoch: ', train_iter.epoch)
            print('train mean loss: {}, accuracy: {}'.format(
                sum_loss / train_count, sum_accuracy / train_count))
            # evaluation
            sum_accuracy = 0
            sum_loss = 0
            for batch in test_iter:
                x_array, t_array = convert.concat_examples(batch, device)
                x = chainer.Variable(x_array)
                t = chainer.Variable(t_array)
                loss = model(x, t)
                sum_loss += float(loss.data) * len(t.data)
                sum_accuracy += float(model.accuracy.data) * len(t.data)

            test_iter.reset()
            print('test mean  loss: {}, accuracy: {}'.format(
                sum_loss / test_count, sum_accuracy / test_count))
            sum_accuracy = 0
            sum_loss = 0

    serializers.save_npz(os.path.join(args.model_dir, 'model.npz'), model)


def model_fn(model_dir):
    model = L.Classifier(MLP(1000, 10))
    serializers.load_npz(os.path.join(model_dir, 'model.npz'), model)
    return model.predictor

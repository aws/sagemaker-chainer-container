import os
import time
from os.path import join

import chainer.datasets
import numpy as np

from test.utils import local_mode
from test.utils.local_mode import request

dir_path = os.path.dirname(os.path.realpath(__file__))

num_records = 4

#TODO: simplify. generate data.
_, test_data = chainer.datasets.get_mnist()
test_data = np.array(test_data[:num_records])
request_data = test_data[:, 0]
labels = test_data[:, -1]

request_data = np.hstack(request_data).reshape((num_records, 784)).astype(np.float64)

labels = np.hstack(labels).astype(np.int64)


def test_chainer_mnist_single_machine(docker_image, opt_ml, use_gpu):
    data_dir = join(dir_path, 'data')
    customer_script = 'single_machine_customer_script.py'
    hyperparameters = {'batch_size': 10000, 'epochs': 1, 'unit':10}

    local_mode.train(customer_script, data_dir, docker_image, opt_ml,
                     hyperparameters=hyperparameters, source_dir=dir_path, use_gpu=use_gpu)

    files = ['model/model.npz', 'output/success', 'output/data/accuracy.png',
                               'output/data/cg.dot', 'output/data/log', 'output/data/loss.png']
    _assert_files_exist(opt_ml, files)
    assert not local_mode.file_exists(opt_ml, 'output/failure'), 'Failure happened'


def test_chainer_mnist_custom_loop(docker_image, opt_ml, use_gpu):
    data_dir = join(dir_path, 'data')
    customer_script = 'single_machine_custom_loop.py'
    hyperparameters = {'batch_size': 10000, 'epochs': 1, 'unit': 10}

    local_mode.train(customer_script, data_dir, docker_image, opt_ml, hyperparameters=hyperparameters,
                     source_dir=dir_path, use_gpu=use_gpu)

    files = ['model/model.npz', 'output/success']
    _assert_files_exist(opt_ml, files)
    assert not local_mode.file_exists(opt_ml, 'output/failure'), 'Failure happened'


def test_chainer_mnist_distributed(docker_image, opt_ml, use_gpu):
    data_dir = join(dir_path, 'data')
    customer_script = 'distributed_customer_script.py'
    hyperparameters = {'batch_size': 10000, 'epochs': 1, 'unit': 10}

    local_mode.train(customer_script, data_dir, docker_image, opt_ml, hyperparameters=hyperparameters,
                     cluster_size=2, source_dir=dir_path, use_gpu=use_gpu)

    files = ['model/model.npz', 'output/success', 'output/data/algo-1/accuracy.png',
             'output/data/algo-1/cg.dot', 'output/data/algo-1/log', 'output/data/algo-1/loss.png']

    _assert_files_exist(opt_ml, files)
    assert not local_mode.file_exists(opt_ml, 'output/failure'), 'Failure happened'


def test_serving(docker_image, opt_ml, py_version):
    model_dir = join(dir_path, 'model')
    customer_script = join(dir_path, 'single_machine_customer_script.py')

    with local_mode.serve(customer_script, model_dir=model_dir, image_name=docker_image, opt_ml=opt_ml):
        # TODO: poll or wait, don't sleep.
        time.sleep(10)

        data_as_list = request_data.tolist()
        _predict_and_compare_labels(data_as_list, 'application/json')
        _predict_and_compare_labels(data_as_list, 'text/csv')
        if py_version == 2: # model was pickled in python 2. TODO: change serialization format or add another model.
            _predict_and_compare_labels(data_as_list, 'application/pickle')


def _predict_and_compare_labels(data, content_type):
    predict_response = request(data, request_type=content_type)
    predict_data = np.array(predict_response)
    predict_labels = predict_data.argmax(axis=1)
    assert np.allclose(predict_labels, labels)

def _assert_files_exist(opt_ml, files):
    for file in files:
        assert local_mode.file_exists(opt_ml, file), 'file {} was not created'.format(file)
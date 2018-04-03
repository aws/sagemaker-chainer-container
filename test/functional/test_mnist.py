import os
from os.path import join

import numpy as np

from test.utils import local_mode
from test.utils.local_mode import request

mnist_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'resources', 'mnist')
data_dir = join(mnist_path, 'data')

def test_chainer_mnist_single_machine(docker_image, opt_ml, use_gpu):
    customer_script = 'single_machine_customer_script.py'
    hyperparameters = {'batch_size': 10000, 'epochs': 1}

    local_mode.train(customer_script, data_dir, docker_image, opt_ml,
                     hyperparameters=hyperparameters, source_dir=mnist_path, use_gpu=use_gpu)

    files = ['model/model.npz', 'output/success', 'output/data/accuracy.png',
                               'output/data/cg.dot', 'output/data/log', 'output/data/loss.png']
    _assert_files_exist(opt_ml, files)
    assert not local_mode.file_exists(opt_ml, 'output/failure'), 'Failure happened'
    with local_mode.serve(join(mnist_path, customer_script), model_dir=None, image_name=docker_image, opt_ml=opt_ml,
                          use_gpu=use_gpu):
        request_data = np.zeros((100, 784))
        data_as_list = request_data.tolist()
        _predict_and_assert_response_length(data_as_list, 'application/json')
        _predict_and_assert_response_length(data_as_list, 'text/csv')


def test_chainer_mnist_custom_loop(docker_image, opt_ml, use_gpu):
    customer_script = 'single_machine_custom_loop.py'
    hyperparameters = {'batch_size': 10000, 'epochs': 1}

    local_mode.train(customer_script, data_dir, docker_image, opt_ml, hyperparameters=hyperparameters,
                     source_dir=mnist_path, use_gpu=use_gpu)

    files = ['model/model.npz', 'output/success']
    _assert_files_exist(opt_ml, files)
    assert not local_mode.file_exists(opt_ml, 'output/failure'), 'Failure happened'

    with local_mode.serve(join(mnist_path, customer_script), model_dir=None, image_name=docker_image, opt_ml=opt_ml):
        request_data = np.zeros((100, 784))
        data_as_list = request_data.tolist()
        _predict_and_assert_response_length(data_as_list, 'application/json')
        _predict_and_assert_response_length(data_as_list, 'text/csv')


def test_chainer_mnist_distributed(docker_image, opt_ml, use_gpu):
    customer_script = 'distributed_customer_script.py'
    cluster_size = 2
    hyperparameters = {'process_slots_per_host': 1,
                       'num_processes': cluster_size,
                       'batch_size': 10000,
                       'epochs': 1,
                       'rank': 'inter_rank'}

    local_mode.train(customer_script, data_dir, docker_image, opt_ml, hyperparameters=hyperparameters,
                     cluster_size=cluster_size, source_dir=mnist_path, use_gpu=use_gpu)

    files = ['model/model.npz', 'output/success', 'output/data/algo-1/accuracy.png',
             'output/data/algo-1/cg.dot', 'output/data/algo-1/log', 'output/data/algo-1/loss.png']

    _assert_files_exist(opt_ml, files)
    assert not local_mode.file_exists(opt_ml, 'output/failure'), 'Failure happened'

    with local_mode.serve(join(mnist_path, customer_script), model_dir=None, image_name=docker_image, opt_ml=opt_ml):
        request_data = np.zeros((100, 784))
        data_as_list = request_data.tolist()
        _predict_and_assert_response_length(data_as_list, 'application/json')
        _predict_and_assert_response_length(data_as_list, 'text/csv')


def _predict_and_assert_response_length(data, content_type):
    # TODO: npz
    predict_response = request(data, request_type=content_type)
    assert len(predict_response) == len(data)


def _assert_files_exist(opt_ml, files):
    for file in files:
        assert local_mode.file_exists(opt_ml, file), 'file {} was not created'.format(file)
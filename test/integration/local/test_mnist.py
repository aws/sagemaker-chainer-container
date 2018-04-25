import os

import numpy as np

from test.utils import local_mode, test_utils

mnist_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'resources', 'mnist')
data_dir = os.path.join(mnist_path, 'data')

def test_chainer_mnist_single_machine(docker_image, opt_ml, use_gpu):
    customer_script = 'single_machine_customer_script.py'
    hyperparameters = {'batch_size': 10000, 'epochs': 1}

    local_mode.train(customer_script, data_dir, docker_image, opt_ml,
                     hyperparameters=hyperparameters, source_dir=mnist_path, use_gpu=use_gpu)

    files = ['model/model.npz', 'output/success', 'output/data/accuracy.png',
                               'output/data/cg.dot', 'output/data/log', 'output/data/loss.png']
    test_utils.files_exist(opt_ml, files)
    assert not local_mode.file_exists(opt_ml, 'output/failure'), 'Failure happened'
    with local_mode.serve(os.path.join(mnist_path, customer_script), model_dir=None, image_name=docker_image,
                          opt_ml=opt_ml, use_gpu=use_gpu):
        test_arrays = [np.zeros((100, 784), dtype='float32'), np.zeros((100, 1, 28, 28), dtype='float32'),
                       np.zeros((100, 28, 28), dtype='float32')]
        request_data = np.zeros((100, 784), dtype='float32')
        data_as_list = request_data.tolist()
        test_utils.predict_and_assert_response_length(data_as_list, 'text/csv')
        for array in test_arrays:
            # JSON and NPY can take multidimensional (n > 2) arrays
            data_as_list = array.tolist()
            test_utils.predict_and_assert_response_length(data_as_list, 'application/json')
            test_utils.predict_and_assert_response_length(request_data, 'application/x-npy')


def test_chainer_mnist_custom_loop(docker_image, opt_ml, use_gpu):
    customer_script = 'single_machine_custom_loop.py'
    hyperparameters = {'batch_size': 10000, 'epochs': 1}

    local_mode.train(customer_script, data_dir, docker_image, opt_ml, hyperparameters=hyperparameters,
                     source_dir=mnist_path, use_gpu=use_gpu)

    files = ['model/model.npz', 'output/success']
    test_utils.files_exist(opt_ml, files)
    assert not local_mode.file_exists(opt_ml, 'output/failure'), 'Failure happened'

    with local_mode.serve(os.path.join(mnist_path, customer_script), model_dir=None, image_name=docker_image,
                          opt_ml=opt_ml):
        request_data = np.zeros((100, 784), dtype='float32')
        data_as_list = request_data.tolist()
        test_utils.predict_and_assert_response_length(data_as_list, 'application/json')
        test_utils.predict_and_assert_response_length(data_as_list, 'text/csv')
        test_utils.predict_and_assert_response_length(request_data, 'application/x-npy')


def test_chainer_mnist_distributed(docker_image, opt_ml, use_gpu):
    customer_script = 'distributed_customer_script.py'
    cluster_size = 2
    # pure_nccl communicator hangs when only one gpu is available.
    hyperparameters = {'process_slots_per_host': 1,
                       'num_processes': cluster_size,
                       'batch_size': 10000,
                       'epochs': 1,
                       'communicator':'hierarchical'}

    local_mode.train(customer_script, data_dir, docker_image, opt_ml, hyperparameters=hyperparameters,
                     cluster_size=cluster_size, source_dir=mnist_path, use_gpu=use_gpu)

    files = ['model/model.npz', 'output/success', 'output/data/algo-1/accuracy.png',
             'output/data/algo-1/cg.dot', 'output/data/algo-1/log', 'output/data/algo-1/loss.png']

    test_utils.files_exist(opt_ml, files)
    assert not local_mode.file_exists(opt_ml, 'output/failure'), 'Failure happened'

    with local_mode.serve(os.path.join(mnist_path, customer_script), model_dir=None, image_name=docker_image,
                          opt_ml=opt_ml):
        request_data = np.zeros((100, 784), dtype='float32')
        data_as_list = request_data.tolist()
        test_utils.predict_and_assert_response_length(data_as_list, 'application/json')
        test_utils.predict_and_assert_response_length(data_as_list, 'text/csv')
        test_utils.predict_and_assert_response_length(request_data, 'application/x-npy')

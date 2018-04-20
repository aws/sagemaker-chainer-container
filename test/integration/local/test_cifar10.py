import os

from test.utils import local_mode

cifar_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'resources', 'cifar')
data_dir = os.path.join(cifar_path, 'data')

def test_chainer_cifar_single_machine(docker_image, opt_ml, use_gpu):
    customer_script = 'single_machine_customer_script.py'
    hyperparameters = {'epochs': 1}
    local_mode.train(customer_script, data_dir, docker_image, opt_ml,
                     hyperparameters=hyperparameters, source_dir=cifar_path, use_gpu=use_gpu)

    files = ['model/model.npz', 'output/success', 'output/data/accuracy.png',
                               'output/data/cg.dot', 'output/data/log', 'output/data/loss.png']

    _assert_files_exist(opt_ml, files)
    assert not local_mode.file_exists(opt_ml, 'output/failure'), 'Failure happened'


def test_chainer_cifar_distributed(docker_image, opt_ml, use_gpu):
    customer_script = 'distributed_customer_script.py'
    cluster_size = 2

    hyperparameters = {'process_slots_per_host': 1,
                       'num_processes': cluster_size,
                       'epochs': 1,
                       'use_mpi': True,
                       'communicator': 'hierarchical'}

    local_mode.train(customer_script, data_dir, docker_image, opt_ml, hyperparameters=hyperparameters,
                     cluster_size=cluster_size, source_dir=cifar_path, use_gpu=use_gpu)

    files = ['model/model.npz', 'output/success', 'output/data/algo-1/accuracy.png',
             'output/data/algo-1/cg.dot', 'output/data/algo-1/log', 'output/data/algo-1/loss.png']

    _assert_files_exist(opt_ml, files)
    assert not local_mode.file_exists(opt_ml, 'output/failure'), 'Failure happened'


def _assert_files_exist(opt_ml, files):
    for f in files:
        assert local_mode.file_exists(opt_ml, f), 'file {} was not created'.format(f)

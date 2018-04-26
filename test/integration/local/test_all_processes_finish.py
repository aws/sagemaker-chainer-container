import os

import pytest

from test.utils import local_mode

resource_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'resources')



def test_all_processes_finish_with_mpi(docker_image, opt_ml, use_gpu):
    """
    This test validates that all training processes finish before containers are shut down.
    """
    customer_script = 'all_processes_finish_customer_script.py'

    cluster_size = 2
    hyperparameters = {'use_mpi': True,
                       'process_slots_per_host': 2,
                       'num_processes': 4}

    local_mode.train(customer_script, resource_path, docker_image, opt_ml,
                     hyperparameters=hyperparameters, source_dir=resource_path, use_gpu=use_gpu,
                     cluster_size=cluster_size)


    assert local_mode.file_exists(opt_ml, 'output/data/algo-2/process_could_complete', host='algo-2'), \
        'Model was not saved'


def test_training_jobs_do_not_stall(docker_image, opt_ml, use_gpu):
    """
    This test validates that training does not stall.
    https://github.com/chainer/chainermn/issues/236
    """
    customer_script = 'training_jobs_do_not_stall_customer_script.py'
    cluster_size = 2
    hyperparameters = {'use_mpi': True,
                       'process_slots_per_host': 1,
                       'num_processes': 2}

    local_mode.train(customer_script, resource_path, docker_image, opt_ml,
                     hyperparameters=hyperparameters, source_dir=resource_path, use_gpu=use_gpu,
                     cluster_size=cluster_size)

    assert local_mode.file_exists(opt_ml, 'output/failure'), 'Failure did not happen'
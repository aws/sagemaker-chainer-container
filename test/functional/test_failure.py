import os

import pytest

from test.utils import local_mode

resource_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'resources', 'failure')

customer_script = 'failure_script.py'

def test_single_machine_failure(docker_image, opt_ml, use_gpu):

    local_mode.train(customer_script, resource_path, docker_image, opt_ml, source_dir=resource_path,
                                 use_gpu=use_gpu)

    assert local_mode.file_exists(opt_ml, 'output/failure'), 'Failure did not happen'


@pytest.mark.parametrize('node_to_fail', [0, 1])
def test_distributed_failure(docker_image, opt_ml, use_gpu, node_to_fail):
    cluster_size = 2
    hyperparameters = {'rank': 'inter_rank',
                       'process_slots_per_host': 1,
                       'num_processes': cluster_size,
                       'node_to_fail': node_to_fail}

    local_mode.train(customer_script, resource_path, docker_image, opt_ml,
                     hyperparameters=hyperparameters, source_dir=resource_path, use_gpu=use_gpu,
                     cluster_size=cluster_size)

    assert local_mode.file_exists(opt_ml, 'output/failure'), 'Failure did not happen'

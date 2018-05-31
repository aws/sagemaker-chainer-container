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
from __future__ import absolute_import

import os

from test.utils import local_mode

current_dir = os.path.dirname(os.path.realpath(__file__))
resource_path = os.path.join(current_dir, '..', '..', 'resources', 'failure_scenarios')


def test_all_processes_finish_with_mpi(docker_image, opt_ml, use_gpu):
    """
    This test validates that all training processes finish before containers are shut down.
    """
    customer_script = 'all_processes_finish_customer_script.py'

    cluster_size = 2
    hyperparameters = {'sagemaker_use_mpi': True, 'sagemaker_process_slots_per_host': 2,
                       'sagemaker_num_processes': 4}

    local_mode.train(customer_script, resource_path, docker_image, opt_ml,
                     hyperparameters=hyperparameters, source_dir=resource_path, use_gpu=use_gpu,
                     cluster_size=cluster_size)

    file_name = 'output/data/algo-2/process_could_complete'

    assert local_mode.file_exists(opt_ml, file_name, host='algo-2'), 'Model was not saved'


def test_training_jobs_do_not_stall(docker_image, opt_ml, use_gpu):
    """
    This test validates that training does not stall.
    https://github.com/chainer/chainermn/issues/236
    """
    customer_script = 'training_jobs_do_not_stall_customer_script.py'
    cluster_size = 2
    hyperparameters = {'sagemaker_use_mpi': True, 'sagemaker_process_slots_per_host': 1,
                       'sagemaker_num_processes': 2}

    local_mode.train(customer_script, resource_path, docker_image, opt_ml,
                     hyperparameters=hyperparameters, source_dir=resource_path,
                     use_gpu=use_gpu, cluster_size=cluster_size)

    assert local_mode.file_exists(opt_ml, 'output/failure'), 'Failure did not happen'


def test_single_machine_failure(docker_image, opt_ml, use_gpu):
    customer_script = 'failure_script.py'

    local_mode.train(customer_script, resource_path, docker_image, opt_ml,
                     source_dir=resource_path, use_gpu=use_gpu)

    assert local_mode.file_exists(opt_ml, 'output/failure'), 'Failure did not happen'


def test_distributed_failure(docker_image, opt_ml, use_gpu):
    customer_script = 'failure_script.py'
    cluster_size = 2
    hyperparameters = {'sagemaker_process_slots_per_host': 1,
                       'sagemaker_num_processes': cluster_size, 'node_to_fail': 1}

    local_mode.train(customer_script, resource_path, docker_image, opt_ml,
                     hyperparameters=hyperparameters, source_dir=resource_path,
                     use_gpu=use_gpu, cluster_size=cluster_size)

    assert local_mode.file_exists(opt_ml, 'output/failure'), 'Failure did not happen'

# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import pytest
from sagemaker.chainer import Chainer

from test.utils import test_utils

current_dir = os.path.dirname(os.path.realpath(__file__))
resource_path = os.path.join(current_dir, '..', '..', 'resources', 'failure_scenarios')
role = 'unused/dummy-role'


def test_all_processes_finish_with_mpi(docker_image, sagemaker_local_session, tmpdir):
    """
    This test validates that all training processes finish before containers are shut down.
    """
    customer_script = 'all_processes_finish_customer_script.py'
    hyperparameters = {'sagemaker_use_mpi': True, 'sagemaker_process_slots_per_host': 2,
                       'sagemaker_num_processes': 4}

    estimator = Chainer(entry_point=customer_script,
                        source_dir=resource_path,
                        role=role,
                        image_name=docker_image,
                        train_instance_count=2,
                        train_instance_type='local',
                        sagemaker_session=sagemaker_local_session,
                        hyperparameters=hyperparameters,
                        output_path='file://{}'.format(tmpdir))

    estimator.fit()

    completion_file = {'output': [os.path.join('data', 'algo-2', 'process_could_complete')]}
    test_utils.files_exist(str(tmpdir), completion_file)


def test_training_jobs_do_not_stall(docker_image, sagemaker_local_session, tmpdir):
    """
    This test validates that training does not stall.
    https://github.com/chainer/chainermn/issues/236
    """
    customer_script = 'training_jobs_do_not_stall_customer_script.py'
    hyperparameters = {'sagemaker_use_mpi': True, 'sagemaker_process_slots_per_host': 1,
                       'sagemaker_num_processes': 2}

    estimator = Chainer(entry_point=customer_script,
                        source_dir=resource_path,
                        role=role,
                        image_name=docker_image,
                        train_instance_count=2,
                        train_instance_type='local',
                        sagemaker_session=sagemaker_local_session,
                        hyperparameters=hyperparameters,
                        output_path='file://{}'.format(tmpdir))

    with pytest.raises(RuntimeError):
        estimator.fit()

    failure_files = {'output': ['failure', os.path.join('data', 'this_file_is_expected')]}
    test_utils.files_exist(str(tmpdir), failure_files)


def test_single_machine_failure(docker_image, instance_type, sagemaker_local_session, tmpdir):
    customer_script = 'failure_script.py'
    estimator = Chainer(entry_point=customer_script,
                        source_dir=resource_path,
                        role=role,
                        image_name=docker_image,
                        train_instance_count=1,
                        train_instance_type=instance_type,
                        sagemaker_session=sagemaker_local_session,
                        output_path='file://{}'.format(tmpdir))

    with pytest.raises(RuntimeError):
        estimator.fit()

    failure_files = {'output': ['failure', os.path.join('data', 'this_file_is_expected')]}
    test_utils.files_exist(str(tmpdir), failure_files)


def test_distributed_failure(docker_image, sagemaker_local_session, tmpdir):
    customer_script = 'failure_script.py'
    cluster_size = 2
    failure_node = 1
    hyperparameters = {'sagemaker_process_slots_per_host': 1,
                       'sagemaker_num_processes': cluster_size, 'node_to_fail': failure_node}

    estimator = Chainer(entry_point=customer_script,
                        source_dir=resource_path,
                        role=role,
                        image_name=docker_image,
                        train_instance_count=cluster_size,
                        train_instance_type='local',
                        sagemaker_session=sagemaker_local_session,
                        hyperparameters=hyperparameters,
                        output_path='file://{}'.format(tmpdir))

    with pytest.raises(RuntimeError):
        estimator.fit()

    node_failure_file = os.path.join('data', 'file_from_node_{}'.format(failure_node))
    failure_files = {'output': ['failure', node_failure_file]}
    test_utils.files_exist(str(tmpdir), failure_files)

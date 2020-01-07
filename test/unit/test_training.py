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

import pytest
import sagemaker_containers.beta.framework as framework
from mock import MagicMock, patch

from sagemaker_chainer_container import training


@pytest.fixture(name='user_module')
def fixture_user_module():
    return MagicMock(spec=['train'])


def mock_training_env(current_host='algo-1', hosts=None, hyperparameters=None,
                      module_dir='s3://my/script', module_name='imagenet', **kwargs):
    hosts = hosts or ['algo-1']

    hyperparameters = hyperparameters or {}

    return MagicMock(current_host=current_host, hosts=hosts, hyperparameters=hyperparameters,
                     module_dir=module_dir, module_name=module_name, user_entry_point=module_name,
                     **kwargs)


@patch('sagemaker_containers.beta.framework.modules.download_and_install')
@patch('sagemaker_containers.beta.framework.entry_point.run')
def test_single_machine(run_entry, download_and_install):
    env = mock_training_env()
    training.train(env, {})

    download_and_install.assert_called_with('s3://my/script')
    run_entry.assert_called_with('s3://my/script',
                                 'imagenet',
                                 env.to_cmd_args(),
                                 env.to_env_vars(),
                                 runner=framework.runner.ProcessRunnerType,
                                 extra_opts={})


@patch('sagemaker_containers.beta.framework.modules.download_and_install')
@patch('sagemaker_containers.beta.framework.entry_point.run')
def test_distributed_training(run_entry, download_and_install):
    hosts = ['algo-1', 'algo-2']
    env = mock_training_env(hosts=hosts)
    training.train(env, {})

    download_and_install.assert_called_with('s3://my/script')
    run_entry.assert_called_with('s3://my/script',
                                 'imagenet',
                                 env.to_cmd_args(),
                                 env.to_env_vars(),
                                 runner=framework.runner.MPIRunnerType,
                                 extra_opts={'sagemaker_mpi_num_of_processes_per_host': None,
                                             'sagemaker_mpi_num_processes': None})

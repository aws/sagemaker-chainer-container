# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

current_dir = os.path.dirname(os.path.realpath(__file__))
resource_path = os.path.join(current_dir, '..', '..', 'resources', 'failure_scenarios')
role = 'SageMakerRole'


def test_distributed_failure(sagemaker_session, ecr_image, instance_type):
    customer_script = 'failure_script.py'
    cluster_size = 2
    failure_node = 1
    hyperparameters = {'sagemaker_process_slots_per_host': 1,
                       'sagemaker_num_processes': cluster_size, 'node_to_fail': failure_node}

    estimator = Chainer(entry_point=customer_script,
                        source_dir=resource_path,
                        role=role,
                        image_name=ecr_image,
                        train_instance_count=cluster_size,
                        train_instance_type=instance_type,
                        sagemaker_session=sagemaker_session,
                        hyperparameters=hyperparameters)

    with pytest.raises(ValueError) as e:
        estimator.fit()

    assert 'Failed Reason: AlgorithmError: framework error:' in str(e)


def test_training_jobs_do_not_stall(sagemaker_session, ecr_image, instance_type):
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
                        image_name=ecr_image,
                        train_instance_count=2,
                        train_instance_type=instance_type,
                        sagemaker_session=sagemaker_session,
                        hyperparameters=hyperparameters)

    with pytest.raises(ValueError) as e:
        estimator.fit()

    assert 'Failed Reason: AlgorithmError: framework error:' in str(e)

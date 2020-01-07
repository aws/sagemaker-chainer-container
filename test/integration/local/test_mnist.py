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

import numpy as np
import pytest
from sagemaker.chainer import Chainer
from sagemaker.predictor import csv_deserializer, csv_serializer, json_deserializer, json_serializer

from test.utils import test_utils

path = os.path.dirname(os.path.realpath(__file__))
mnist_path = os.path.join(path, '..', '..', 'resources', 'mnist')
data_dir = os.path.join(mnist_path, 'data')
role = 'unused/dummy-role'


def test_chainer_mnist_single_machine(docker_image, sagemaker_local_session, instance_type, tmpdir):
    customer_script = 'single_machine_customer_script.py'
    hyperparameters = {'batch-size': 10000, 'epochs': 1}

    estimator = Chainer(entry_point=customer_script,
                        source_dir=mnist_path,
                        role=role,
                        image_name=docker_image,
                        train_instance_count=1,
                        train_instance_type=instance_type,
                        sagemaker_session=sagemaker_local_session,
                        hyperparameters=hyperparameters,
                        output_path='file://{}'.format(tmpdir))

    estimator.fit({'train': 'file://{}'.format(os.path.join(data_dir, 'train')),
                   'test': 'file://{}'.format(os.path.join(data_dir, 'test'))})

    success_files = {
        'model': ['model.npz'],
        'output': ['success', 'data/accuracy.png', 'data/cg.dot', 'data/log', 'data/loss.png'],
    }
    test_utils.files_exist(str(tmpdir), success_files)

    request_data = np.zeros((100, 784), dtype='float32')

    test_utils.predict_and_assert_response_length(estimator, request_data, instance_type)
    test_utils.predict_and_assert_response_length(estimator, request_data, instance_type,
                                                  csv_serializer, csv_deserializer, 'text/csv')

    test_arrays = [np.zeros((100, 784), dtype='float32'),
                   np.zeros((100, 1, 28, 28), dtype='float32'),
                   np.zeros((100, 28, 28), dtype='float32')]

    with test_utils.local_mode_lock():
        try:
            predictor = _json_predictor(estimator, instance_type)
            for array in test_arrays:
                response = predictor.predict(array)
                assert len(response) == len(array)
        finally:
            predictor.delete_endpoint()


def test_chainer_mnist_custom_loop(docker_image, sagemaker_local_session, instance_type, tmpdir):
    customer_script = 'single_machine_custom_loop.py'
    hyperparameters = {'batch-size': 10000, 'epochs': 1}

    estimator = Chainer(entry_point=customer_script,
                        source_dir=mnist_path,
                        role=role,
                        image_name=docker_image,
                        train_instance_count=1,
                        train_instance_type=instance_type,
                        sagemaker_session=sagemaker_local_session,
                        hyperparameters=hyperparameters,
                        output_path='file://{}'.format(tmpdir))

    estimator.fit({'train': 'file://{}'.format(os.path.join(data_dir, 'train')),
                   'test': 'file://{}'.format(os.path.join(data_dir, 'test'))})

    success_files = {
        'model': ['model.npz'],
        'output': ['success'],
    }

    test_utils.files_exist(str(tmpdir), success_files)

    request_data = np.zeros((100, 784), dtype='float32')

    test_utils.predict_and_assert_response_length(estimator, request_data, instance_type)
    test_utils.predict_and_assert_response_length(estimator, request_data, instance_type,
                                                  json_serializer, json_deserializer,
                                                  'application/json')
    test_utils.predict_and_assert_response_length(estimator, request_data, instance_type,
                                                  csv_serializer, csv_deserializer, 'text/csv')


@pytest.mark.parametrize('customer_script',
                         ['distributed_customer_script.py',
                          'distributed_customer_script_with_env_vars.py'])
def test_chainer_mnist_distributed(docker_image, sagemaker_local_session, instance_type,
                                   customer_script, tmpdir):
    if instance_type == 'local_gpu':
        pytest.skip('Local Mode does not support distributed GPU training.')

    # pure_nccl communicator hangs when only one gpu is available.
    cluster_size = 2
    hyperparameters = {'sagemaker_process_slots_per_host': 1,
                       'sagemaker_num_processes': cluster_size,
                       'batch-size': 10000,
                       'epochs': 1,
                       'communicator': 'hierarchical'}

    estimator = Chainer(entry_point=customer_script,
                        source_dir=mnist_path,
                        role=role,
                        image_name=docker_image,
                        train_instance_count=cluster_size,
                        train_instance_type=instance_type,
                        sagemaker_session=sagemaker_local_session,
                        hyperparameters=hyperparameters,
                        output_path='file://{}'.format(tmpdir))

    estimator.fit({'train': 'file://{}'.format(os.path.join(data_dir, 'train')),
                   'test': 'file://{}'.format(os.path.join(data_dir, 'test'))})

    success_files = {
        'model': ['model.npz'],
        'output': ['success', 'data/accuracy.png', 'data/cg.dot', 'data/log', 'data/loss.png'],
    }

    test_utils.files_exist(str(tmpdir), success_files)

    request_data = np.zeros((100, 784), dtype='float32')

    test_utils.predict_and_assert_response_length(estimator, request_data, instance_type)
    test_utils.predict_and_assert_response_length(estimator, request_data, instance_type,
                                                  json_serializer, json_deserializer,
                                                  'application/json')
    test_utils.predict_and_assert_response_length(estimator, request_data, instance_type,
                                                  csv_serializer, csv_deserializer, 'text/csv')


def _json_predictor(estimator, instance_type):
    predictor = estimator.deploy(1, instance_type)
    predictor.content_type = 'application/json'
    predictor.serializer = json_serializer
    predictor.accept = 'application/json'
    predictor.deserializer = json_deserializer
    return predictor

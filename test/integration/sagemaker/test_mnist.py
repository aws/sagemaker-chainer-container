# Copyright 2018-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from __future__ import absolute_import

import os

import numpy as np
import pytest
import sagemaker.utils
from sagemaker.chainer import Chainer
from sagemaker.chainer.model import ChainerModel
from sagemaker.utils import sagemaker_timestamp

from timeout import timeout, timeout_and_delete_endpoint_by_name


@pytest.mark.deploy_test
def test_chainer_mnist_single_machine_train(sagemaker_session, ecr_image, instance_type):
    _test_mnist_train(sagemaker_session, ecr_image, instance_type, 1, 'single_machine_customer_script.py')


@pytest.mark.deploy_test
def test_chainer_deploy(sagemaker_session, instance_type):
    _test_mnist_deploy(sagemaker_session, instance_type)


def test_chainer_mnist_distributed_train(sagemaker_session, ecr_image, instance_type):
    _test_mnist_train(sagemaker_session, ecr_image, instance_type, 2, 'distributed_customer_script.py')


def _test_mnist_train(sagemaker_session, ecr_image, instance_type, instance_count, script):
    source_dir = 'test/resources/mnist'

    with timeout(minutes=15):
        data_path = 'test/resources/mnist/data'

        chainer = Chainer(entry_point=script,
                          source_dir=source_dir,
                          role='SageMakerRole',
                          train_instance_count=instance_count,
                          train_instance_type=instance_type,
                          sagemaker_session=sagemaker_session,
                          image_name=ecr_image,
                          hyperparameters={'batch-size': 10000, 'epochs': 1})

        prefix = 'chainer_mnist/{}'.format(sagemaker_timestamp())

        train_data_path = os.path.join(data_path, 'train')

        key_prefix = prefix + '/train'
        train_input = sagemaker_session.upload_data(path=train_data_path, key_prefix=key_prefix)

        test_path = os.path.join(data_path, 'test')
        test_input = sagemaker_session.upload_data(path=test_path, key_prefix=prefix + '/test')

        chainer.fit({'train': train_input, 'test': test_input})


def _test_mnist_deploy(sagemaker_session, instance_type):
    model_path = 'test/resources/mnist/model.tar.gz'
    script_path = 'test/resources/mnist/mnist.py'

    endpoint_name = sagemaker.utils.unique_name_from_base('sagemaker-chainer-test')
    model_data = sagemaker_session.upload_data(
        path=model_path,
        key_prefix='sagemaker-chainer/models',
    )

    with timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session, minutes=30):
        chainer = ChainerModel(
            model_data=model_data,
            role='SageMakerRole',
            entry_point=script_path,
            sagemaker_session=sagemaker_session,
        )
        predictor = chainer.deploy(initial_instance_count=1, instance_type=instance_type)

        batch_size = 100
        data = np.zeros(shape=(batch_size, 1, 28, 28), dtype='float32')
        output = predictor.predict(data)
        assert len(output) == batch_size

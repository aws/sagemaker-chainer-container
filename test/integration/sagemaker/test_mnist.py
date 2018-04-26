#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#  
#      http://www.apache.org/licenses/LICENSE-2.0
#  
#  or in the "license" file accompanying this file. This file is distributed 
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
#  express or implied. See the License for the specific language governing 
#  permissions and limitations under the License.

from estimator import ChainerTestEstimator
from sagemaker.utils import sagemaker_timestamp
from timeout import timeout, timeout_and_delete_endpoint
import numpy as np
import os


def test_chainer_mnist_single_machine(sagemaker_session, ecr_image, instance_type):
    script_path = 'test/resources/mnist/single_machine_customer_script.py'
    _test_mnist(sagemaker_session, ecr_image, instance_type, script_path, 1)


def test_chainer_mnist_distributed(sagemaker_session, ecr_image, instance_type):
    script_path = 'test/resources/mnist/distributed_customer_script.py'
    _test_mnist(sagemaker_session, ecr_image, instance_type, script_path, 2)


def _test_mnist(sagemaker_session, ecr_image, instance_type, script_path, instance_count):
    with timeout(minutes=15):
        data_path = 'test/resources/mnist/data'

        chainer = ChainerTestEstimator(entry_point=script_path, role='SageMakerRole',
                                train_instance_count=instance_count, train_instance_type=instance_type,
                                sagemaker_session=sagemaker_session,
                                docker_image_uri=ecr_image,
                                hyperparameters={'epochs': 1})

        prefix = 'chainer_mnist/{}'.format(sagemaker_timestamp())
        train_input = chainer.sagemaker_session.upload_data(path=os.path.join(data_path, 'train'),
                                                       key_prefix=prefix + '/train')
        test_input = chainer.sagemaker_session.upload_data(path=os.path.join(data_path, 'test'),
                                                      key_prefix=prefix + '/test')
        chainer.fit({'train': train_input, 'test': test_input})

    with timeout_and_delete_endpoint(estimator=chainer, minutes=30):
        predictor = chainer.deploy(initial_instance_count=1, instance_type=instance_type)

        batch_size = 100
        data = np.zeros(shape=(batch_size, 1, 28, 28), dtype='float32')
        output = predictor.predict(data)
        assert len(output) == batch_size

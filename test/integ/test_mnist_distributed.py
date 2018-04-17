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

from sagemaker.chainer.estimator import Chainer
from sagemaker.utils import sagemaker_timestamp
from timeout import timeout, timeout_and_delete_endpoint
import numpy as np
import os


class ChainerTestEstimator(Chainer):
    def __init__(self, docker_image_uri, **kwargs):
        super(ChainerTestEstimator, self).__init__(**kwargs)
        self.docker_image_uri = docker_image_uri

    def train_image(self):
        return self.docker_image_uri

    def create_model(self, model_server_workers=None):
        model = super(ChainerTestEstimator, self).create_model()
        model.image = self.docker_image_uri
        return model


def test_chainer_distributed(sagemaker_session, ecr_image, instance_type):
    with timeout(minutes=15):
        script_path = 'test/resources/mnist/distributed_customer_script.py'
        data_path = 'test/resources/mnist/data'

        chainer = ChainerTestEstimator(entry_point=script_path, role='SageMakerRole',
                                train_instance_count=2, train_instance_type=instance_type,
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
        data = np.zeros(shape=(batch_size, 1, 28, 28))
        output = predictor.predict(data)
        assert len(output) == batch_size

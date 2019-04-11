# Copyright 2017-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from sagemaker.chainer import ChainerModel
from sagemaker.predictor import BytesDeserializer

from test.utils import test_utils

path = os.path.dirname(os.path.realpath(__file__))
resources_path = os.path.abspath(os.path.join(path, '..', '..', 'resources', 'serving'))


def test_serving_calls_model_fn_once(docker_image, sagemaker_local_session):
    script_path = os.path.join(resources_path, 'call_model_fn_once.py')
    model_path = 'file://{}'.format(os.path.join(resources_path, 'model.tar.gz'))

    model = ChainerModel(model_path,
                         'SageMakerRole',
                         script_path,
                         image=docker_image,
                         model_server_workers=2,
                         sagemaker_session=sagemaker_local_session)

    with test_utils.local_mode_lock():
        try:
            predictor = model.deploy(1, 'local')
            predictor.accept = None
            predictor.deserializer = BytesDeserializer()

            # call enough times to ensure multiple requests to a worker
            for i in range(3):
                # will return 500 error if model_fn called during request handling
                response = predictor.predict(b'input')
                assert response == b'output'
        finally:
            predictor.delete_endpoint()

from __future__ import absolute_import

import os

import requests

from test.utils import local_mode

path = os.path.dirname(os.path.realpath(__file__))
resources_path = os.path.abspath(os.path.join(path, '..', '..', 'resources'))


def test_serving_calls_model_fn_once(docker_image, opt_ml):
    script_path = os.path.join(resources_path, 'call_model_fn_once.py')
    with local_mode.serve(script_path, model_dir=None, image_name=docker_image, opt_ml=opt_ml,
                          additional_env_vars=['SAGEMAKER_MODEL_SERVER_WORKERS=2']):

        # call enough times to ensure multiple requests to a worker
        for i in range(3):
            # will return 500 error if model_fn called during request handling
            assert b'output' == requests.post(local_mode.REQUEST_URL, data=b'input').content

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
import json
import os
import shutil
import tempfile

from mock import MagicMock
import pytest
from sagemaker_containers.beta.framework import params

INPUT_DATA_CONFIG = {
    "train": {
        "ContentType": "trainingContentType"
    },
    "evaluation": {
        "ContentType": "evalContentType"
    },
    "Validation": {}
}

HYPERPARAMETERS = {
    params.REGION_NAME_PARAM: 'us-west-2',
    params.USER_PROGRAM_PARAM: 'myscript.py',
    params.SUBMIT_DIR_PARAM: 's3://mybucket/code.tar.gz'
}


@pytest.fixture(name='training_env')
def fixture_training_env():
    return MagicMock()


@pytest.fixture(name='training_state')
def fixture_training_state():
    training_state = MagicMock()
    training_state.trained = False
    training_state.saved = False
    return training_state


@pytest.fixture(name='user_module')
def fixture_user_module():
    return MagicMock(spec=['train'])


@pytest.fixture(name='user_module_with_save')
def fixture_user_module_with_save():
    return MagicMock(spec=['train', 'save'])


@pytest.fixture(name='training_structure')
def fixture_training_structure():
    d = _setup_training_structure()
    yield d
    shutil.rmtree(d)


def _write_config_file(path, filename, data):
    path = os.path.join(path, "input/config/%s" % filename)
    with open(path, 'w') as f:
        json.dump(data, f)


def _write_resource_config(path, current_host, hosts):
    _write_config_file(path, 'resourceconfig.json', {
        'current_host': current_host,
        'hosts': hosts
    })


def _serialize_hyperparameters(hp):
    return {str(k): json.dumps(v) for (k, v) in hp.items()}


def _setup_training_structure():
    tmp = tempfile.mkdtemp()
    for d in ['input/data/training', 'input/config', 'model', 'output/data']:
        os.makedirs(os.path.join(tmp, d))

    with open(os.path.join(tmp, 'input/data/training/data.csv'), 'w') as f:
        f.write('dummy data file')

    _write_resource_config(tmp, 'a', ['a', 'b'])
    _write_config_file(tmp, 'inputdataconfig.json', INPUT_DATA_CONFIG)
    _write_config_file(tmp, 'hyperparameters.json',
                       _serialize_hyperparameters(HYPERPARAMETERS))

    return tmp

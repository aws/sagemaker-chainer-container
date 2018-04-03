import json
import os
import shutil
import tempfile

import pytest
from container_support import ContainerEnvironment
from mock import MagicMock, patch

from chainer_framework.training import train

INPUT_DATA_CONFIG = {
    "train": {"ContentType": "trainingContentType"},
    "evaluation": {"ContentType": "evalContentType"},
    "Validation": {}
}

HYPERPARAMETERS = {
    ContainerEnvironment.SAGEMAKER_REGION_PARAM_NAME: 'us-west-2',
    ContainerEnvironment.USER_SCRIPT_NAME_PARAM: 'myscript.py',
    ContainerEnvironment.USER_SCRIPT_ARCHIVE_PARAM: 's3://mybucket/code.tar.gz'
}


@pytest.fixture()
def training_env():
    return MagicMock()


@pytest.fixture()
def training_state():
    training_state = MagicMock()
    training_state.trained = False
    training_state.saved = False
    return training_state


@pytest.fixture()
def user_module():
    return MagicMock(spec=['train'])


@pytest.fixture()
def user_module_with_save():
    return MagicMock(spec=['train', 'save'])


@pytest.fixture()
def training_structure():
    d = _setup_training_structure()
    yield d
    shutil.rmtree(d)


def test_train(training_env, user_module_with_save, training_state):
    pass


def _write_config_file(path, filename, data):
    path = os.path.join(path, "input/config/%s" % filename)
    with open(path, 'w') as f:
        json.dump(data, f)


def _write_resource_config(path, current_host, hosts):
    _write_config_file(path, 'resourceconfig.json', {'current_host': current_host, 'hosts': hosts})


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
    _write_config_file(tmp, 'hyperparameters.json', _serialize_hyperparameters(HYPERPARAMETERS))

    return tmp

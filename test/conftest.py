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

import logging
import os
import platform
import shutil
import tempfile

import boto3
import pytest
from sagemaker import LocalSession, Session
from sagemaker.chainer import Chainer

from test.utils import image_utils

logger = logging.getLogger(__name__)
logging.getLogger('boto').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.INFO)
logging.getLogger('factory.py').setLevel(logging.INFO)
logging.getLogger('auth.py').setLevel(logging.INFO)
logging.getLogger('connectionpool.py').setLevel(logging.INFO)

dir_path = os.path.dirname(os.path.realpath(__file__))

NO_P2_REGIONS = ['ap-east-1', 'ap-northeast-3', 'ap-southeast-2', 'ca-central-1', 'eu-central-1',
                 'eu-north-1', 'eu-west-2', 'eu-west-3', 'sa-east-1', 'us-west-1', 'me-south-1']
NO_P3_REGIONS = ['ap-east-1', 'ap-northeast-3', 'ap-southeast-1', 'ap-southeast-2', 'ap-south-1', 'ca-central-1',
                 'eu-central-1', 'eu-north-1', 'eu-west-2', 'eu-west-3', 'sa-east-1', 'us-west-1', 'me-south-1']


def pytest_addoption(parser):
    parser.addoption('--build-image', '-D', action="store_true")
    parser.addoption('--build-base-image', '-B', action="store_true")
    parser.addoption('--aws-id')
    parser.addoption('--instance-type')
    parser.addoption('--docker-base-name', default='chainer')
    parser.addoption('--region', default='us-west-2')
    parser.addoption('--framework-version', default=Chainer.LATEST_VERSION)
    parser.addoption('--py-version', choices=['2', '3', '2,3'], default='3')
    parser.addoption('--processor', choices=['gpu', 'cpu'], default='cpu')
    # If not specified, will default to {framework-version}-{processor}-py{py-version}
    parser.addoption('--tag', default=None)


def pytest_generate_tests(metafunc):
    if 'py_version' in metafunc.fixturenames:
        py_version_params = ['py' + v for v in metafunc.config.getoption('--py-version').split(',')]
        metafunc.parametrize('py_version', py_version_params, scope='session')


# pylint: disable=unused-argument
@pytest.fixture(scope='session', name='docker_base_name')
def fixture_docker_base_name(request):
    return request.config.getoption('--docker-base-name')


@pytest.fixture(scope='session', name='region')
def fixture_region(request):
    return request.config.getoption('--region')


@pytest.fixture(scope='session', name='framework_version')
def fixture_framework_version(request):
    return request.config.getoption('--framework-version')


@pytest.fixture(scope='session', name='processor')
def fixture_processor(request):
    return request.config.getoption('--processor')


@pytest.fixture(scope='session', name='tag')
def _fixture_tag(request, framework_version, processor, py_version):
    provided_tag = request.config.getoption('--tag')
    default_tag = '{}-{}-{}'.format(framework_version, processor, py_version)
    return provided_tag if provided_tag else default_tag


@pytest.fixture(scope='session', name='docker_image')
def fixture_docker_image(docker_base_name, tag):
    return '{}:{}'.format(docker_base_name, tag)


@pytest.fixture
def opt_ml():
    tmp = tempfile.mkdtemp()
    os.mkdir(os.path.join(tmp, 'output'))

    # Docker cannot mount Mac OS /var folder properly see
    # https://forums.docker.com/t/var-folders-isnt-mounted-properly/9600
    opt_ml_dir = '/private{}'.format(tmp) if platform.system() == 'Darwin' else tmp

    yield opt_ml_dir

    shutil.rmtree(tmp, True)


@pytest.fixture(scope='session', name='use_gpu')
def fixture_use_gpu(processor):
    return processor == 'gpu'


@pytest.fixture(scope='session', name='aws_id')
def fixture_aws_id(request):
    return request.config.getoption('--aws-id')


@pytest.fixture(scope='session', name='instance_type')
def fixture_instance_type(request, processor):
    provided_instance_type = request.config.getoption('--instance-type')
    default_instance_type = 'local' if processor == 'cpu' else 'local_gpu'
    return provided_instance_type or default_instance_type


@pytest.fixture(scope='session', name='docker_registry')
def fixture_docker_registry(aws_id, region):
    return '{}.dkr.ecr.{}.amazonaws.com'.format(aws_id, region)


@pytest.fixture(scope='session', name='ecr_image')
def fixture_ecr_image(docker_registry, docker_base_name, tag):
    return '{}/{}:{}'.format(docker_registry, docker_base_name, tag)


@pytest.fixture(scope='session', name='sagemaker_session')
def fixture_sagemaker_session(region):
    return Session(boto_session=boto3.Session(region_name=region))


@pytest.fixture(scope='session', name='sagemaker_local_session')
def fixture_sagemaker_local_session(region):
    return LocalSession(boto_session=boto3.Session(region_name=region))


@pytest.fixture(scope='session', autouse=True, name='build_base_image')
def fixture_build_base_image(request, framework_version, processor, tag, docker_base_name):
    build_base_image = request.config.getoption('--build-base-image')
    if build_base_image:
        return image_utils.build_base_image(framework_name=docker_base_name,
                                            framework_version=framework_version,
                                            base_image_tag=tag,
                                            processor=processor,
                                            cwd=os.path.join(dir_path, '..'))

    return tag


@pytest.fixture(scope='session', autouse=True, name='build_image')
def fixture_build_image(request, py_version, framework_version, processor, tag, docker_base_name):
    build_image = request.config.getoption('--build-image')
    if build_image:
        return image_utils.build_image(framework_name=docker_base_name,
                                       py_version=py_version,
                                       framework_version=framework_version,
                                       processor=processor,
                                       tag=tag,
                                       cwd=os.path.join(dir_path, '..'))

    return tag


@pytest.fixture(autouse=True)
def skip_gpu_instance_restricted_regions(region, instance_type):
    if((region in NO_P2_REGIONS and instance_type.startswith('ml.p2')) or
       (region in NO_P3_REGIONS and instance_type.startswith('ml.p3'))):
        pytest.skip('Skipping GPU test in region {}'.format(region))

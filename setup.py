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
from glob import glob

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='sagemaker-chainer-container',
    version='1.0',
    description='Open source library template for creating containers to run on Amazon SageMaker.',
    packages=find_packages(where='src', exclude='test'),
    package_dir={'': 'src'},
    py_modules=[os.splitext(os.basename(path))[0] for path in glob('src/*.py')],
    long_description=read('README.rst'),
    author='Amazon Web Services',
    license='Apache License 2.0',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ],
    # Temporarily freeze sagemaker-containers version to 2.2.5 until we have a proper fix
    # freeze numpy version because of the python2 bug
    # in 16.0: https://github.com/numpy/numpy/pull/12754
    install_requires=['sagemaker-containers==2.5.0', 'chainer==5.0.0', 'retrying==1.3.3',
                      'numpy==1.16.2'],

    extras_require={
        'test': [
            'tox', 'flake8', 'coverage', 'flake8-import-order', 'pytest==4.5.0', 'pytest-cov',
            'pytest-xdist', 'mock', 'Flask', 'boto3>=1.4.8', 'docker-compose',
            'nvidia-docker-compose', 'sagemaker==1.29.0', 'PyYAML==3.10', 'pluggy==0.11'
        ]
    })

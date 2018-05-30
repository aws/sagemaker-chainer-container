from __future__ import absolute_import

from glob import glob
import os

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='sagemaker-chainer-container',
    version='1.0',
    description='Open source library template for creating containers to run on Amazon SageMaker.',
    packages=find_packages(where='src', exclude='test'),
    package_dir={'sagemaker_chainer_container': 'src/sagemaker_chainer_container'},
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
        'Programming Language :: Python :: 3.5',
    ],
    install_requires=['sagemaker-containers==2.0.2', 'chainer >= 4.0.0', 'retrying==1.3.3',
                      'numpy >= 1.14'],

    dependency_links=['pip install git+https://github.com/aws/sagemaker-python-sdk-staging'],
    extras_require={
        'test': [
            'tox', 'flake8', 'coverage', 'flake8-import-order', 'pytest', 'pytest-cov', 'pytest-xdist', 'mock',
            'Flask', 'boto3>=1.4.8', 'docker-compose', 'nvidia-docker-compose', 'sagemaker>=1.3.0',
            'PyYAML', 'requests'
        ]
    })

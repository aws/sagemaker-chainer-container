from glob import glob
import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='sagemaker-chainer-container',
    version='1.0',
    description='Open source library template for creating containers to run on Amazon SageMaker.',
    packages=find_packages(where='src', exclude=('test',)),
    package_dir={'chainer_framework': 'src/chainer_framework'},
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

    install_requires=['sagemaker-containers==2.0.0', 'chainer', 'retrying==1.3.3'],
    dependency_links=['pip install git+https://github.com/mvsusp/sagemaker-containers'
                      '@mvs-backward-compatibility-support',
                      'pip install git+https://github.com/aws/sagemaker-python-sdk-staging'],
    extras_require={
        'test': ['tox', 'flake8', 'flake8-import-order', 'pytest', 'pytest-cov', 'pytest-xdist', 'mock', 'Flask',
                 'boto3>=1.4.8', 'docker-compose', 'nvidia-docker-compose', 'sagemaker', 'PyYAML']
    }
)

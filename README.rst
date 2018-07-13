==========================
SageMaker Chainer Containers
==========================

SageMaker Chainer Containers is an open source library for making the
Chainer framework run on Amazon SageMaker.

This repository also contains Dockerfiles which install this library, Chainer, and dependencies
for building SageMaker Chainer images.

For information on running Chainer jobs on SageMaker: `Python
SDK <https://github.com/aws/sagemaker-python-sdk#chainer-sagemaker-estimators>`__.

For notebook examples: `SageMaker Notebook
Examples <https://github.com/awslabs/amazon-sagemaker-examples>`__.

Table of Contents
-----------------

#. `Getting Started <#getting-started>`__
#. `Building your Image <#building-your-image>`__
#. `Running the tests <#running-the-tests>`__

Getting Started
---------------

Prerequisites
~~~~~~~~~~~~~

Make sure you have installed all of the following prerequisites on your
development machine:

- `Docker <https://www.docker.com/>`__

For Testing on GPU
^^^^^^^^^^^^^^^^^^

-  `Nvidia-Docker <https://github.com/NVIDIA/nvidia-docker>`__

Recommended
^^^^^^^^^^^

-  A python environment management tool. (e.g.
   `PyEnv <https://github.com/pyenv/pyenv>`__,
   `VirtualEnv <https://virtualenv.pypa.io/en/stable/>`__)

Building your image
-------------------

`Amazon SageMaker <https://aws.amazon.com/documentation/sagemaker/>`__
utilizes Docker containers to run all training jobs & inference endpoints.

The Docker images are built from the Dockerfiles specified in
`Docker/ <https://github.com/aws/sagemaker-chainer-container/tree/master/docker>`__.

The Docker files are grouped based on Chainer version and separated
based on Python version and processor type.

The Docker images, used to run training & inference jobs, are built from
both corresponding "base" and "final" Dockerfiles.

Base Images
~~~~~~~~~~~

The "base" Dockerfile encompass the installation of the framework and all of the dependencies
needed.

Tagging scheme is based on <Chainer_version>-<processor>-<python_version>. (e.g. 4.0.0-cpu-py2)

All "final" Dockerfiles build images using base images that use the tagging scheme
above.

If you want to build your base docker image, then use:

::

    # All build instructions assume you're building from the same directory as the dockerfile.

    # CPU
    docker build -t chainer-base:<Chainer_version>-cpu-<python_version> -f Dockerfile.cpu .

    # GPU
    docker build -t chainer-base:<Chainer_version>-gpu-<python_version> -f Dockerfile.gpu .

::

    # Example

    # CPU
    docker build -t chainer-base:4.0.0-cpu-py2 -f Dockerfile.cpu .

    # GPU
    docker build -t chainer-base:4.0.0-gpu-py2 -f Dockerfile.gpu .

Final Images
~~~~~~~~~~~~

The "final" Dockerfiles encompass the installation of the SageMaker specific support code.

All "final" Dockerfiles use `base images for building <https://github
.com/aws/sagemaker-chainer-container/blob/master/docker/0.12.1/final/py2/Dockerfile.cpu#L2>`__.

These "base" images are specified with the naming convention of
chainer-base:<Chainer_version>-<processor>-<python_version>.

Before building "final" images:

Build your "base" image. Make sure it is named and tagged in accordance with your "final"
Dockerfile.


::

    # Create the SageMaker Chainer Container Python package.
    cd sagemaker-chainer-container
    python setup.py bdist_wheel

    #. Copy your Python package to "final" Dockerfile directory that you are building.
    cp -R dist/sagemaker_chainer_container-<package_version>.tar.gz docker/<Chainer_version>/final/<py_version>

If you want to build "final" Docker images, then use:

::

    # All build instructions assumes you're building from the same directory as the dockerfile.

    # CPU
    docker build -t <image_name>:<tag> -f Dockerfile.cpu .

    # GPU
    docker build -t <image_name>:<tag> -f Dockerfile.gpu .

::

    # Example

    # CPU
    docker build -t preprod-chainer:4.0.0-cpu-py2 -f Dockerfile.cpu .

    # GPU
    docker build -t preprod-chainer:4.0.0-gpu-py2 -f Dockerfile.gpu .


Running the tests
-----------------

Running the tests requires installation of the SageMaker Chainer Container code and its test
dependencies.

::

    git clone https://github.com/aws/sagemaker-chainer-container.git
    cd sagemaker-chainer-container
    pip install -e .[test]

Tests are defined in
`test/ <https://github.com/aws/sagemaker-chainer-container/tree/master/test>`__
and include unit, local integration, and SageMaker integration tests.

Unit Tests
~~~~~~~~~~

If you want to run unit tests, then use:

::

    # All test instructions should be run from the top level directory

    pytest test/unit

Local Integration Tests
~~~~~~~~~~~~~~~~~~~~~~~

Running local integration tests require `Docker <https://www.docker.com/>`__ and `AWS
credentials <https://docs.aws.amazon.com/sdk-for-java/v1/developer-guide/setup-credentials.html>`__,
as the local integration tests make calls to a couple AWS services. The local integration tests and
SageMaker integration tests require configurations specified within their respective
`conftest.py <https://github.com/aws/sagemaker-chainer-container/blob/master/test/conftest.py>`__.

Local integration tests on GPU require `Nvidia-Docker <https://github.com/NVIDIA/nvidia-docker>`__.

Before running local integration tests:

#. Build your Docker image.
#. Pass in the correct pytest arguments to run tests against your Docker image.

If you want to run local integration tests, then use:

::

    # Required arguments for integration tests are found in test/conftest.py

    pytest test/integration/local --docker-base-name <your_docker_image> \
                      --tag <your_docker_image_tag> \
                      --py-version <2_or_3> \
                      --framework-version <Chainer_version> \
                      --processor <cpu_or_gpu>

::

    # Example
    pytest test/integration/local --docker-base-name preprod-chainer \
                      --tag 1.0 \
                      --py-version 2 \
                      --framework-version 4.0.0 \
                      --processor cpu

SageMaker Integration Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~

SageMaker integration tests require your Docker image to be within an `Amazon ECR repository <https://docs
.aws.amazon.com/AmazonECS/latest/developerguide/ECS_Console_Repositories.html>`__.

The Docker-base-name is your `ECR repository namespace <https://docs.aws.amazon
.com/AmazonECR/latest/userguide/Repositories.html>`__.

The instance-type is your specified `Amazon SageMaker Instance Type
<https://aws.amazon.com/sagemaker/pricing/instance-types/>`__ that the SageMaker integration test will run on.

Before running SageMaker integration tests:

#. Build your Docker image.
#. Push the image to your ECR repository.
#. Pass in the correct pytest arguments to run tests on SageMaker against the image within your ECR repository.

If you want to run a SageMaker integration end to end test on `Amazon
SageMaker <https://aws.amazon.com/sagemaker/>`__, then use:

::

    # Required arguments for integration tests are found in test/conftest.py

    pytest test/integration/sagemaker --aws-id <your_aws_id> \
                           --docker-base-name <your_docker_image> \
                           --instance-type <amazon_sagemaker_instance_type> \
                           --tag <your_docker_image_tag> \

::

    # Example
    pytest test/integration/sagemaker --aws-id 12345678910 \
                           --docker-base-name preprod-chainer \
                           --instance-type ml.m4.xlarge \
                           --tag 1.0

Contributing
------------

Please read
`CONTRIBUTING.md <https://github.com/aws/sagemaker-chainer-container/blob/master/CONTRIBUTING.md>`__
for details on our code of conduct, and the process for submitting pull
requests to us.

License
-------

SageMaker Chainer Containers is licensed under the Apache 2.0 License. It is copyright 2018 Amazon
.com, Inc. or its affiliates. All Rights Reserved. The license is available at:
http://aws.amazon.com/apache2.0/
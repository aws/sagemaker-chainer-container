# Changelog

## v1.0.0 (2019-06-25)

### Bug fixes and other changes

 * pin sagemaker-container version
 * fix typo
 * enable release build in new build system
 * Fix logger in test scripts
 * Consolidate environment variable parsing in integ test training script
 * Use SageMaker Containers MPI entry point
 * pin pytest to 4.5.0 to work around configparser error
 * pin pluggy version to workaround configparser error
 * Fix broken line of buildspec
 * prevent hidden errors in buildspec
 * Add codebuild buildspec file for pull request
 * Use the SageMaker Python SDK for remote integ test
 * Use the SageMaker Python SDK for failure scenario local integ tests
 * Use the SageMaker Python SDK for local serving integ test
 * Use the SageMaker Python SDK for local MNIST integration tests
 * Upgrade numpy to 1.16.2 for compatibility to ChainerCV
 * Upgrade ChainerCV to 0.12.0 in Chainer 5.0.0
 * Upgrade to python3.6
 * Freeze PyYAML version to avoid conflict with Docker Compose
 * Read framework version from Python SDK for integ test default
 * Configure encoding to be utf-8
 * Freeze pip version to <= 18.1 and numpy version <=1.15.4
 * remove requests from test dependencies
 * Remove additional framework parameters from MPI command
 * fix response content_type handling
 * Upgrade SageMaker Containers version
 * update cupy to 5.0 for chainer 5.0
 * Add Dockerfiles for Chainer 5.0
 * Unfreeze requests version
 * add port label
 * Temporarily freeze sagemaker-containers version to 2.2.5 until we have a proper fix.
 * pin requests version
 * bump minimum sagemaker-containers version to 2.2.0
 * Remove installation of sagemaker containers in dockerfile
 * Migrate tests to nvidia-docker2
 * Fix broken image build instructions
 * initialize model once
 * Allow local code directories for training and serving
 * Fix broken image build instruction
 * Add version 4.1.0
 * Fix support for subdirectories in user code
 * revert startup time change, add timestamps to s3 prefix for code
 * increase startup time to stop spurious hosting failures
 * remove preprod from test fixture
 * change name to sagemaker_chainer_container
 * Using last sagemaker containers version
 * add Readme
 * Configure logging
 * Unit test fix
 * support for env vars
 * Script mode
 * fix unit test -- too many files opened
 * add workdir statement to docker file, fix hyperparameters in tests
 * apply patch during build to chainer to fix multiprocess parallel updater
 * add sagemaker prefix to hyperparameters
 * Chainer migration
 * wait for algo-1 to be available when running with mpi
 * disable backprop in default predict function. do not get list for npy…
 * add opencv, cv2 to images
 * update py3 gpu to 4.0.0
 * combine failure / process finishing tests
 * update version to 4.0.0
 * add -m mpi4py to mpi script to stop stalled jobs
 * refactor test code
 * Update to Chainer 3.5.0
 * reorganize tests
 * add subproject commits
 * Add numpy (npy) handlers
 * add integ tests, build changes for integ tests:
 * Merge pull request #7 from aws/mvsusp-flake8
 * Update setup.py
 * Fixing application import name
 * Create .flake8
 * add chainer test code
 * add chainer framework code
 * add dockerfiles for 3.4.0
 * Updating initial README.md from template
 * Creating initial file from template
 * Creating initial file from template
 * Creating initial file from template
 * Creating initial file from template
 * Creating initial file from template
 * Initial commit


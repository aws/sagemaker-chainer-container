# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import sagemaker_containers.beta.framework as framework

logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    level=logging.INFO)

logging.getLogger('boto3').setLevel(logging.INFO)
logging.getLogger('s3transfer').setLevel(logging.INFO)
logging.getLogger('botocore').setLevel(logging.WARN)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train(env, hyperparameters):
    """Runs Chainer training on a user supplied module in either a local or distributed
    SageMaker environment.

    The user supplied module and its dependencies are downloaded from S3.
    Training is invoked by calling a "train" function in the user supplied module.

    If the environment contains multiple hosts, then a distributed learning
    task is started with mpirun.

    The following is a list of other hyperparameters that can be used to change training behavior.
    None of these hyperparameters are required:

    * `sagemaker_use_mpi`: force use of MPI so that ChainerMN scripts can be run on a single host.
    * `sagemaker_process_slots_per_host`: the number of process slots per host.
    * `sagemaker_num_processes`: the total number of processes to run.
    * `sagemaker_additional_mpi_options`: a string of options to pass to mpirun.
    """
    framework.modules.download_and_install(env.module_dir)

    use_mpi = bool(hyperparameters.get('sagemaker_use_mpi', len(env.hosts) > 1))
    opts = {}

    if use_mpi:
        runner_type = framework.runner.MPIRunnerType
        opts = {
            framework.params.MPI_PROCESSES_PER_HOST: hyperparameters.get('sagemaker_process_slots_per_host'),
            framework.params.MPI_NUM_PROCESSES: hyperparameters.get('sagemaker_num_processes'),
        }
    else:
        runner_type = framework.runner.ProcessRunnerType

    framework.entry_point.run(env.module_dir,
                              env.user_entry_point,
                              env.to_cmd_args(),
                              env.to_env_vars(),
                              runner=runner_type,
                              extra_opts=opts)


def main():
    hyperparameters = framework.env.read_hyperparameters()
    env = framework.training_env(hyperparameters=hyperparameters)

    logger.setLevel(env.log_level)
    train(env, hyperparameters)


# This branch hit by mpi_script.sh (see docker base directory)
if __name__ == '__main__':
    main()

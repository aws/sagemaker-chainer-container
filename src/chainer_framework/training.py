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
import logging
import os
import shlex
import socket
import stat
import subprocess
import sys
import textwrap
import time

from retrying import retry
from sagemaker_containers import env, mapping, modules, trainer

from chainer_framework.timeout import timeout

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_PORT = 7777
_MPI_SCRIPT = "/mpi_script.sh"
_MPI_IS_RUNNING = "/mpi_is_running"
_MPI_IS_FINISHED = "/mpi_is_finished"
_CHANGE_HOSTNAME_LIBRARY = "/libchangehostname.so"

MODEL_FILE_NAME = "model.npz"


def train(training_env):
    """Runs Chainer training on a user supplied module in either a local or distributed
    SageMaker environment.

    The user supplied module and its dependencies are downloaded from S3.
    Training is invoked by calling a "train" function in the user supplied module.

    If the environment contains multiple hosts, then a distributed learning
    task is started with mpirun.

    The following is a list of other hyperparameters that can be used to change training behavior.
    None of these hyperparameters are required:

    * `use_mpi`: force use of MPI so that ChainerMN scripts can be run on a single host.
    * `process_slots_per_host`: the number of process slots per host.
    * `num_processes`: the total number of processes to run.
    * `additional_mpi_options`: a string of options to pass to mpirun.

    For more on how distributed training uses these parameters, please see :func:`_get_mpi_command`.

    Args:
        training_env : training environment object containing environment variables,
                               training arguments and hyperparameters
    """

    use_mpi = bool(training_env.hyperparameters.get('use_mpi', len(training_env.hosts) > 1))

    if use_mpi:

        current_host = training_env.current_host
        hosts = list(training_env.hosts)
        _change_hostname(current_host)
        _start_ssh_daemon()

        _create_mpi_script(training_env)

        if current_host == _get_master_host_name(hosts):
            _wait_for_worker_nodes_to_start_sshd(hosts)

            _run_mpi_on_all_nodes(training_env)
        else:
            _wait_for_training_to_finish(training_env)
    else:
        _run_training(training_env)


def _run_training(training_env):
    logger.info('Invoking user training script.')

    hyperparameters = mapping.to_cmd_args(training_env.hyperparameters)
    channels = mapping.to_cmd_args(training_env.channel_input_dirs)

    args = hyperparameters + channels

    _run_module_from_s3(args, training_env)


@trainer.report_training_status
def _run_module_from_s3(args, training_env):
    modules.run_module_from_s3(training_env.module_dir, args, training_env.module_name)


def _change_hostname(current_host):
    """Compiles a shared library to correct the behavior of the gethostname system call,
        which OpenMPI depends on.

    Args:
        current_host (str): name of the current host, such as algo-1, algo-2, etc.
    """
    os.system("change-hostname.sh {}".format(current_host))


def _get_master_host_name(hosts):
    return sorted(hosts)[0]


def _run_mpi_on_all_nodes(training_env):
    mpi_command = _get_mpi_command(training_env)
    logger.info("mpi_command: %s", mpi_command)

    _run_mpi(mpi_command)


@trainer.report_training_status
def _run_mpi(mpi_command):
    import pdb;
    pdb.set_trace()
    subprocess.check_call(shlex.split(mpi_command))


def _get_mpi_command(training_env):
    """Constructs a command to run distributed training with MPI using mpirun.

    Runs /mpi_script.sh on all hosts listed in the training environment. How many
    processes in total is determined by the 'num_processes' hyperparameter, or one
    per GPU, or one per CPU, as applicable. The 'process_slots_per_host'
    hyperparameter can be used to override how many processes can be placed on each host.

    Additional MPI options can be passed (and override other MPI options) using the
    'additional_mpi_options' hyperparameter.

    This command passes many options to the mpirun command:

    * --host [host:slots]: A comma-delimited list of hosts and the number of process
        slots on each host.
    * -mca btl_tcp_if_include [training_env.network_interface_name]: Tell OpenMPI to use
        the given network interface name for byte transfer layer communication.
    * -mca oob_tcp_if_include [training_env.network_interface_name]: Tell OpenMPI to use
        the given network interface name for out-of-band communication.
    * -mca btl ^openib: Don't look for openib components (this just avoids a warning)
    * -x PATH: pass $PATH from the current environment to the execution environments on remote hosts
    * -x LD_LIBRARY_PATH: pass $LD_LIBRARY_PATH from the current environment to the execution
        environments on remote hosts
    * -x LD_PRELOAD=[changehostname library]: Load the changehostname library to return
        correct values from gethostname system calls.
    * -mca orte_abort_on_non_zero_status 1: Return a non-zero exit code if any process exits
        with a non-zero exit code.
    * -x NCCL_DEBUG=INFO: Enable info level logging for NCCL.
    * -x NCCL_SOCKET_IFNAME=[training_env.network_interface_name]: Tell NCCL to use the given
        network interface name for socket communication.
    * -np [num_processes]: total number of processes to run across all nodes.

    Args:
        training_env: training environment object containing environment variables,
                              training arguments and hyperparameters.

    Returns:
        str: The mpirun command to run.
    """
    num_gpus = training_env.num_gpus
    hyperparameters = training_env.hyperparameters
    is_gpu = num_gpus if num_gpus > 0 else 1

    process_slots_per_host = int(hyperparameters.get('process_slots_per_host', is_gpu))

    num_hosts = len(training_env.hosts)
    num_processes = int(hyperparameters.get('num_processes', process_slots_per_host * num_hosts))

    # By default, use one process per GPU, or one process per node (if training with CPU).
    host_list = training_env.hosts if process_slots_per_host == 1 else \
        [host + ':{}'.format(process_slots_per_host) for host in training_env.hosts]

    additional_mpi_options = str(hyperparameters.get('additional_mpi_options', ''))

    credential_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN']

    logger.info('network interface name: %s', training_env.network_interface_name)

    mpi_command = 'mpirun --allow-run-as-root --host {}'.format(",".join(host_list)) \
                  + " -mca btl_tcp_if_include {}".format(training_env.network_interface_name) \
                  + " -mca oob_tcp_if_include {}".format(training_env.network_interface_name) \
                  + " -mca btl ^openib" \
                  + " -x PATH" \
                  + " -x LD_LIBRARY_PATH" \
                  + " -x LD_PRELOAD={}".format(_CHANGE_HOSTNAME_LIBRARY) \
                  + " -mca orte_abort_on_non_zero_status 1" \
                  + " -x NCCL_DEBUG=INFO" \
                  + " -x NCCL_SOCKET_IFNAME={}".format(training_env.network_interface_name) \
                  + " -np {} ".format(num_processes)

    for v in credential_vars:
        if v in os.environ:
            mpi_command += " -x {}".format(v)

    mpi_command += " {} ".format(additional_mpi_options) + " {}".format(_MPI_SCRIPT)

    return mpi_command


def _create_mpi_script(training_env):
    hyperparameters = mapping.to_cmd_args(training_env.hyperparameters)
    channels = mapping.to_cmd_args(training_env.channel_input_dirs)
    modules.download_and_install(training_env.module_dir)

    python_cmd = [sys.executable, '-m', 'mpi4py', '-m', training_env.module_name]
    python_cmd.extend(hyperparameters)
    python_cmd.extend(channels)

    content = textwrap.dedent("""#!/usr/bin/env bash
# For distributed training: the 'master node' runs mpirun with this script, '/mpi_script.sh'
# This script creates a file '/mpi_is_running' that worker nodes use to determine whether training (started by MPI from
# the master node) is still running. Processes on worker nodes use /mpi_is_finished file to determine when to exit.
touch /mpi_is_running
%s
EXIT_CODE=$?
touch /mpi_is_finished

exit ${EXIT_CODE}
""" % ' '.join(python_cmd))

    with open(_MPI_SCRIPT, 'w') as w:
        w.write(content)

    st = os.stat(_MPI_SCRIPT)
    os.chmod(_MPI_SCRIPT, st.st_mode | stat.S_IEXEC)


def _start_ssh_daemon():
    subprocess.Popen(["/usr/sbin/sshd", "-D"])


def _wait_for_training_to_finish(training_env):
    current_host = training_env.current_host

    logger.info("worker node %s is waiting for MPI to start training process", current_host)
    _wait_for_mpi_to_start_running()

    logger.info("MPI started training process on worker node %s", current_host)

    _wait_until_mpi_stops_running()
    logger.info("Training process started by MPI on worker node %s stopped", current_host)


def _wait_for_worker_nodes_to_start_sshd(hosts, interval=1, timeout_in_seconds=180):
    with timeout(seconds=timeout_in_seconds):
        while hosts:
            logger.info("hosts that aren't SSHable yet: %s", str(hosts))
            for host in hosts:
                ssh_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                if _can_connect(host, 22, ssh_socket):
                    hosts.remove(host)
            time.sleep(interval)


def _can_connect(host, port, s):
    try:
        logger.debug("testing connection to host %s", host)
        s.connect((host, port))
        s.close()
        logger.debug("can connect to host %s", host)
        return True
    except socket.error:
        logger.debug("can't connect to host %s", host)
        return False


def _retry_if_false(result):
    return result is False


@retry(stop_max_delay=30 * 1000, wait_fixed=1000, retry_on_result=_retry_if_false)
def _wait_for_mpi_to_start_running():
    return os.path.isfile(_MPI_IS_RUNNING)


@retry(wait_fixed=5000, retry_on_result=_retry_if_false)
def _wait_until_mpi_stops_running():
    return os.path.isfile(_MPI_IS_FINISHED)


def main():
    training_env = env.TrainingEnv()

    train(training_env)


# This branch hit by mpi_script.sh (see docker base directory)
if __name__ == '__main__':
    main()

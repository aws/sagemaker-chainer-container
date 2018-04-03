import logging
import os
import shlex
import socket
import subprocess
import time

from chainer_framework import timeout
from chainer import serializers

from container_support.app import TrainingEngine
import container_support as cs
from container_support.environment import TrainingEnvironment

logger = logging.getLogger(__name__)

engine = TrainingEngine()

_PORT = 7777
_MPI_SCRIPT = "/mpi_script.sh"
_MPI_IS_RUNNING = "/mpi_is_running"
_MPI_IS_FINISHED = "/mpi_is_finished"
_CHANGE_HOSTNAME_LIBRARY = "/libchangehostname.so"


@engine.train()
def train(user_module, training_environment):
    """Runs Chainer training on a user supplied module in either a local or distributed
    SageMaker environment.

    The user supplied module and its dependencies are downloaded from S3.
    Training is invoked by calling a "train" function in the user supplied module.

    If the environment contains multiple hosts, then a distributed learning
    task is started with mpirun.

    The following is a list of other hyperparameters that can be used to change training behavior. None of these
    hyperparameters are required:

    1. `use_mpi`: force use of MPI so that ChainerMN scripts can be run on a single host.
    2. `process_slots_per_host`: the number of process slots per host.
    3. `num_processes`: the total number of processes to run.
    4. `additional_mpi_options`: a string of options to pass to mpirun.

    For more on how distributed

    Args:
        user_module : a user supplied module.
        training_environment : training environment object containing environment variables,
                               training arguments and hyperparameters
    """

    use_mpi = training_environment.hyperparameters.get('use_mpi', len(training_environment.hosts) > 1)

    if not use_mpi:
        _run_training()
    else:
        current_host = training_environment.current_host
        hosts = training_environment.hosts
        _change_hostname(current_host)
        if current_host == _get_master_host_name(hosts):
            _wait_for_worker_nodes_to_start_sshd([host for host in hosts if host != current_host])
            _run_mpi_on_all_nodes(training_environment)
        else:
            _start_ssh_daemon()
            _wait_for_training_to_finish(training_environment)


def _run_training():
    env = TrainingEnvironment()
    user_module = env.import_user_module()
    training_parameters = env.matching_parameters(user_module.train)
    logger.info('Invoking user training script.')
    model = user_module.train(**training_parameters)

    if model and env.current_host == 'algo-1':
        if hasattr(user_module, 'save'):
            user_module.save(model, env.model_dir)
        else:
            serializers.save_npz(os.path.join(env.model_dir, 'model.npz'), model)
    if not model and env.current_host == 'algo-1':
        logger.warn("Model object is empty. No model was saved!")


def _change_hostname(current_host):
    """Compiles a shared library to correct the behavior of the gethostname system call, which OpenMPI depends on.

    Args:
        current_host (str): name of the current host, such as algo-1, algo-2, etc.
    """
    os.system("change-hostname.sh %s" % current_host)


def _get_master_host_name(hosts):
    return sorted(hosts)[0]


def _run_mpi_on_all_nodes(training_environment):
    mpi_command = _get_mpi_command(training_environment)
    logger.info("mpi_command: " + mpi_command)
    subprocess.check_call(shlex.split(mpi_command))


def _get_mpi_command(training_environment):
    """Constructs a command to run distributed training with MPI using mpirun.

    Runs /mpi_script.sh on all hosts listed in the training environment. How many processes in total is determined
    by the 'num_processes' hyperparameter, or one per GPU, or one per CPU, as applicable. The 'process_slots_per_host'
    hyperparameter can be used to override how many processes can be placed on each host.

    Additional MPI options can be passed (and override other MPI options) using the 'additional_mpi_options'
    hyperparameter.

    This command passes many options to the mpirun command:

    1. --host [host:slots]: A comma-delimited list of hosts and the number of process slots on each host.
    2. -mca btl_tcp_if_include [network_interface_name]: Tell OpenMPI to use the given network interface name for
         byte transfer layer communication.
    3. -mca oob_tcp_if_include [network_interface_name]: Tell OpenMPI to use the given network interface name for
         out-of-band communication.
    4. -mca btl ^openib: Don't look for openib components (this just avoids a warning)
    5. -x PATH: pass $PATH from the current environment to the execution environments on remote hosts
    6. -x LD_LIBRARY_PATH: pass $LD_LIBRARY_PATH from the current environment to the execution environments on remote
         hosts
    7. -x LD_PRELOAD=[changehostname library]: Load the changehostname library to return correct values from gethostname
         system calls.
    8. -mca orte_abort_on_non_zero_status 1: Return a non-zero exit code if any process exits with a non-zero exit code.
    9. -x NCCL_DEBUG=INFO: Enable info level logging for NCCL.
    10. -x NCCL_SOCKET_IFNAME=[network_interface_name]: Tell NCCL to use the given network interface name for socket
         communication.
    11. -np [num_processes]: total number of processes to run across all nodes.

    Args:
        training_environment: training environment object containing environment variables,
                              training arguments and hyperparameters.

    Returns:
        str: The mpirun command to run.
    """
    num_gpus = training_environment.available_gpus
    hyperparameters = training_environment.hyperparameters
    process_slots_per_host = int(hyperparameters.get('process_slots_per_host', num_gpus if num_gpus > 0 else 1))

    num_hosts = len(training_environment.hosts)
    num_processes = int(hyperparameters.get('num_processes', process_slots_per_host * num_hosts))

    # By default, use one process per GPU, or one process per node (if training with CPU).
    host_list = training_environment.hosts if process_slots_per_host == 1 else \
        [host + ':{}'.format(process_slots_per_host) for host in training_environment.hosts]

    additional_mpi_options = str(hyperparameters.get('additional_mpi_options', ''))

    mpi_command = 'mpirun --allow-run-as-root --host {}'.format(",".join(host_list)) \
                  + " -mca btl_tcp_if_include {0}".format(training_environment.network_interface_name) \
                  + " -mca oob_tcp_if_include {0}".format(training_environment.network_interface_name) \
                  + " -mca btl ^openib" \
                  + " -x PATH" \
                  + " -x LD_LIBRARY_PATH" \
                  + " -x LD_PRELOAD=".format(_CHANGE_HOSTNAME_LIBRARY) \
                  + " -mca orte_abort_on_non_zero_status 1" \
                  + " -x NCCL_DEBUG=INFO" \
                  + " -x NCCL_SOCKET_IFNAME={}".format(training_environment.network_interface_name) \
                  + " -np {} ".format(num_processes) \
                  + " {} ".format(additional_mpi_options) \
                  + " {}".format(_MPI_SCRIPT)
    return mpi_command


def _start_ssh_daemon():
    subprocess.Popen(["/usr/sbin/sshd", "-D"])


def _wait_for_training_to_finish(training_environment):
    current_host = training_environment.current_host

    logger.info("worker node {} is waiting for MPI to start training process ".format(current_host))
    _wait_for_mpi_to_start_running()

    logger.info("MPI started training process on worker node {}".format(current_host))

    _wait_until_mpi_stops_running()
    logger.info("Training process started by MPI on worker node {} stopped" .format(current_host))


def _wait_for_worker_nodes_to_start_sshd(hosts, interval=1):
    with timeout(seconds=180):
        while len(hosts) > 0:
            logger.info("hosts that aren't SSHable yet: " + str(hosts))
            for host in hosts:
                host_is_sshable = _can_connect(host, 22)
                if host_is_sshable:
                    hosts.remove(host)
            time.sleep(interval)


def _can_connect(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        logger.debug("testing connection to host " + host)
        s.connect((host, port))
        s.close()
        logger.debug("can connect to host " + host)
        return True
    except socket.error:
        logger.debug("can't connect to host " + host)
        return False


def _retry_if_false(result):
    return result is False


@cs.retry(stop_max_delay=30 * 1000,
          wait_fixed=1000,
          retry_on_result=_retry_if_false)
def _wait_for_mpi_to_start_running():
    return os.path.isfile(_MPI_IS_RUNNING)


@cs.retry(wait_fixed=5000,
          retry_on_result=_retry_if_false)
def _wait_until_mpi_stops_running():
    return os.path.isfile(_MPI_IS_FINISHED)


if __name__=="__main__":
    _run_training()

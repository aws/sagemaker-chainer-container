import os, time, subprocess, socket, logging

from chainer_framework import run_training

from container_support.app import TrainingEngine

logger = logging.getLogger(__name__)

engine = TrainingEngine()


@engine.train()
def train(user_module, training_environment):
    """ Runs training on a user supplied module.

    Training is invoked by calling a "train" function in the user supplied module.
    """

    if len(training_environment.hosts) == 1:
        run_training.train(user_module, training_environment)
    else:
        if _is_master_node(training_environment.current_host):
            _wait_for_nodes_to_start_sshd([host for host in training_environment.hosts
                                           if host != training_environment.current_host])

            # See the Dockerfile for more on mpi_script.sh.
            # The master node uses mpirun to run /mpi_script.sh on worker nodes to create a file ('/mpi_is_running')
            # to indicate that training is underway.
            mpi_script_name = "/mpi_script.sh"

            # Use the 'ethwe' network interface. OpenMPI needs this to mpirun or else:
            # [aws:00053] [[41151,0],1] tcp_peer_send_blocking: send() to socket 9 failed: Broken pipe (32)
            # TODO: remove this after EASE VPC launches. Read from resource config in train. Contact Jeffrey.
            mpi_mca_options = "--mca btl_tcp_if_include {0} --mca oob_tcp_if_include {0}"\
                .format(training_environment.network_interface_name)

            # TODO: choose number of processes (for gpu and cpu)?
            # TODO: stdout from all nodes is sent to algo-1 -- instead, capture stdout on worker nodes.
            mpi_command = 'mpirun --allow-run-as-root --host ' + ','.join(training_environment.hosts) + ' ' \
                          + " {} ".format(mpi_mca_options) \
                          + mpi_script_name
            logger.info("mpi_command: " + mpi_command)

            # TODO: sys.exit(1) on called process failure -- fails quickly?
            try:
                subprocess.check_call(mpi_command.split())
            except subprocess.CalledProcessError:
                logger.error("mpi call failed")

            logger.info("mpi call is finished, master is exiting")
        else:
            subprocess.Popen(["/usr/sbin/sshd", "-D"])
            # workers poll their own filesystems to see if training is underway (in another process started by MPI
            # on the master node)
            mpi_running_file = '/mpi_is_running'
            logger.info("worker node {} is waiting for MPI to start training process "
                        .format(training_environment.current_host))
            _wait_for_mpi_to_start_running(mpi_running_file, training_environment.current_host)
            logger.info("MPI started training process on worker node {}".format(training_environment.current_host))

            # TODO: replace this with something more stable: send signal over ssh.
            _wait_until_mpi_stops_running(mpi_running_file)
            logger.info("Training process started by MPI on worker node {} stopped"
                        .format(training_environment.current_host))


def _is_master_node(current_host):
    """ Returns True if this machine should start the training processes on the other nodes."""
    return current_host == 'algo-1'


def _wait_for_nodes_to_start_sshd(hosts, interval=1, timeout=180):
    '''
    "master" node needs to wait for other nodes to be sshable before running MPI.
    '''
    def _check_ssh_daemon(host):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            logger.info("testing ssh connection to host " + host)
            s.connect((host, 22))
            s.close()
            logger.info("can ssh to host" + host)
            return True
        except socket.error:
            logger.info("can't ssh to host " + host)
            return False

    time_elapsed = 0
    while len(hosts) > 0 and time_elapsed < timeout:
        logger.info("hosts that aren't SSHable yet: " + str(hosts))
        for host in hosts:
            host_is_sshable = _check_ssh_daemon(host)
            if host_is_sshable:
                hosts.remove(host)
        time.sleep(interval)
        time_elapsed += interval
        if time_elapsed > timeout:
            raise RuntimeError("Couldn't connect to ssh daemon for all hosts. Hosts: " + str(hosts))


def _wait_for_mpi_to_start_running(mpi_running_file, current_host, interval=10, timeout=60):
    '''
    Workers need to wait for master node to start training on workers with MPI.
    '''
    time_elapsed = 0
    mpi_is_running = _is_mpi_running(mpi_running_file)
    while not mpi_is_running and time_elapsed < timeout:
        mpi_is_running = _is_mpi_running(mpi_running_file)
        time.sleep(interval)
        time_elapsed += interval
        if time_elapsed > timeout:
            raise RuntimeError("MPI didn't start process on host " + current_host)


def _wait_until_mpi_stops_running(mpi_running_file, interval=10):
    '''
    Waits for training process started by MPI to complete so that worker nodes know when to die.

    EASE fails quickly when any container fails, but succeeds only if all containers exit successfully.
    So workers need to know when training on master node finished. /mpi_script.sh removes this file when training
    finishes so workers can exit.

    '''
    mpi_is_running = _is_mpi_running(mpi_running_file)
    while mpi_is_running:
        mpi_is_running = _is_mpi_running(mpi_running_file)
        time.sleep(interval)


def _is_mpi_running(mpi_running_file):
    return os.path.isfile(mpi_running_file)
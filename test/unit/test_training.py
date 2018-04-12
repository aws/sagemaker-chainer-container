import os
import tempfile
import pytest
import shlex
import socket
from mock import MagicMock, patch, call

import chainer
import chainer.links as L
from chainer import serializers

from chainer_framework.training import _CHANGE_HOSTNAME_LIBRARY, _MPI_IS_RUNNING, _MPI_IS_FINISHED, \
    MODEL_FILE_NAME, train, _change_hostname, _get_master_host_name, _run_training, \
    _run_mpi_on_all_nodes, _get_mpi_command, _start_ssh_daemon, _wait_for_training_to_finish, _default_save, \
    _wait_for_worker_nodes_to_start_sshd, _can_connect, _wait_for_mpi_to_start_running, _wait_until_mpi_stops_running, \
    _use_mpi, _num_processes, _process_slots_per_host, _additional_mpi_options
from chainer_framework.timeout import TimeoutError


@pytest.fixture()
def master_node_distributed_training_env():
    env = MagicMock()
    env.current_host = 'algo-1'
    env.hosts = ['algo-1', 'algo-2']
    env.hyperparameters = {}
    env.network_interface_name = "ethmock"
    return env


@pytest.fixture()
def worker_node_distributed_training_env():
    env = MagicMock()
    env.current_host = 'algo-2'
    env.hosts = ['algo-1', 'algo-2']
    env.hyperparameters = {}
    return env


@pytest.fixture()
def single_machine_training_env():
    env = MagicMock()
    env.current_host = 'algo-1'
    env.hosts = ['algo-1']
    env.hyperparameters = {}
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, 'model'))
    env.model_dir = os.path.join(tmp, 'model')
    return env


@pytest.fixture()
def training_state():
    training_state = MagicMock()
    training_state.trained = False
    training_state.saved = False
    return training_state


@pytest.fixture()
def user_module():
    return MagicMock(spec=['train'])


@pytest.fixture()
def user_module_with_save():
    return MagicMock(spec=['train', 'save'])


class DummyModel(chainer.Chain):

    def __init__(self):
        super(DummyModel, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None)

    def __call__(self, x):
        return self.l1()


def test_single_machine_train_and_default_save(single_machine_training_env, user_module, training_state):
    def user_module_train():
        training_state.trained = True
        training_state.model = chainer.Chain()
        return training_state.model
    user_module.train = user_module_train

    train(user_module, single_machine_training_env)

    assert training_state.trained
    assert os.path.exists(os.path.join(single_machine_training_env.model_dir, MODEL_FILE_NAME))


def test_single_machine_train_and_user_module_save(single_machine_training_env, user_module_with_save, training_state):
    def user_module_train():
        training_state.trained = True
        training_state.model = chainer.Chain()
        return training_state.model
    user_module_with_save.train = user_module_train

    train(user_module_with_save, single_machine_training_env)

    assert training_state.trained
    user_module_with_save.save.assert_called_with(training_state.model, single_machine_training_env.model_dir)


def test_default_save(single_machine_training_env):
    model = DummyModel()

    _default_save(single_machine_training_env, model)
    model_path = os.path.join(single_machine_training_env.model_dir, MODEL_FILE_NAME)
    loaded_model = DummyModel()

    serializers.load_npz(os.path.join(model_path), loaded_model)


def test_warn_when_no_model_is_saved(single_machine_training_env, user_module, training_state):
    def user_module_train():
        training_state.trained = True
        training_state.model = None
        return training_state.model
    user_module.train = user_module_train

    with patch('logging.Logger.warn') as mock:
        train(user_module, single_machine_training_env)

        assert training_state.trained
        mock.assert_called_with("Model object is empty. No model was saved! train() should return a model.")


def test_distributed_training_save_model_on_master_node(master_node_distributed_training_env, user_module):
    def user_module_train():
        training_state.trained = True
        training_state.model = chainer.Chain()
        return training_state.model
    user_module_with_save.train = user_module_train

    with patch('chainer_framework.training._default_save') as mock_default_save:
        _run_training(master_node_distributed_training_env, user_module)

        mock_default_save.assert_called_once()


def test_distributed_training_dont_save_model_on_worker_nodes(worker_node_distributed_training_env, user_module):
    def user_module_train():
        training_state.trained = True
        training_state.model = chainer.Chain()
        return training_state.model
    user_module_with_save.train = user_module_train

    with patch('chainer_framework.training._default_save') as mock_default_save:
        _run_training(worker_node_distributed_training_env, user_module)

        mock_default_save.assert_not_called()


def test_distributed_training_from_master_node(master_node_distributed_training_env, user_module):
    with patch('chainer_framework.training._change_hostname') as mock_change_hostname, \
         patch('chainer_framework.training._wait_for_worker_nodes_to_start_sshd') as mock_wait_for_sshd, \
         patch ('chainer_framework.training._run_mpi_on_all_nodes') as mock_run_mpi_on_all_nodes:

        train(user_module, master_node_distributed_training_env)

        mock_change_hostname.assert_called_once_with(master_node_distributed_training_env.current_host)
        mock_wait_for_sshd.assert_called_once_with([host for host in master_node_distributed_training_env.hosts
                                                    if host != master_node_distributed_training_env.current_host])
        mock_run_mpi_on_all_nodes.assert_called_once_with(master_node_distributed_training_env)


def test_distributed_training_from_master_node(master_node_distributed_training_env, user_module):
    with patch('chainer_framework.training._change_hostname') as mock_change_hostname, \
         patch('chainer_framework.training._wait_for_worker_nodes_to_start_sshd') as mock_wait_for_sshd, \
         patch ('chainer_framework.training._run_mpi_on_all_nodes') as mock_run_mpi_on_all_nodes:

        train(user_module, master_node_distributed_training_env)

        mock_change_hostname.assert_called_once_with(master_node_distributed_training_env.current_host)
        mock_wait_for_sshd.assert_called_once_with([host for host in master_node_distributed_training_env.hosts
                                                    if host != master_node_distributed_training_env.current_host])
        mock_run_mpi_on_all_nodes.assert_called_once_with(master_node_distributed_training_env)


def test_distributed_training_from_worker_node(worker_node_distributed_training_env, user_module):
    with patch('chainer_framework.training._change_hostname') as mock_change_hostname, \
         patch('chainer_framework.training._start_ssh_daemon') as mock_start_ssh_daemon, \
         patch('chainer_framework.training._wait_for_training_to_finish') as mock_wait_for_training_to_finish:

        train(user_module, worker_node_distributed_training_env)

        mock_change_hostname.assert_called_once_with(worker_node_distributed_training_env.current_host)
        mock_start_ssh_daemon.assert_called_once()
        mock_wait_for_training_to_finish.assert_called_once_with(worker_node_distributed_training_env)


def test_change_hostname(single_machine_training_env):
    with patch('os.system') as mock_system:
        _change_hostname(single_machine_training_env.current_host)
        mock_system.assert_called_with("change-hostname.sh {}".format(single_machine_training_env.current_host))


def test_run_mpi_on_all_nodes(master_node_distributed_training_env):
    with patch('subprocess.check_call') as mock_check_call:
        _run_mpi_on_all_nodes(master_node_distributed_training_env)
        mock_check_call.assert_called_with(shlex.split(_get_mpi_command(master_node_distributed_training_env)))


def test_get_mpi_command(master_node_distributed_training_env):
    mpi_command = _get_mpi_command(master_node_distributed_training_env)

    network_interface_name = master_node_distributed_training_env.network_interface_name
    assert "mpirun" in mpi_command
    assert "--allow-run-as-root" in mpi_command
    assert "-host algo-1,algo-2" in mpi_command
    assert "-mca btl_tcp_if_include {}".format(network_interface_name) in mpi_command
    assert "-mca oob_tcp_if_include {}".format(network_interface_name) in mpi_command
    assert "-x PATH" in mpi_command
    assert "-x LD_LIBRARY_PATH" in mpi_command
    assert "-x LD_PRELOAD={}".format(_CHANGE_HOSTNAME_LIBRARY) in mpi_command
    assert "-mca orte_abort_on_non_zero_status 1" in mpi_command
    assert "-x NCCL_SOCKET_IFNAME={}".format(network_interface_name) in mpi_command
    assert "-np 2" in mpi_command


def test_get_mpi_command_with_additional_hyperparameters(master_node_distributed_training_env):
    master_node_distributed_training_env.hyperparameters = {'num_processes': 1024,
                                                            'process_slots_per_host':128,
                                                            'additional_mpi_options': '-X MY_ENVIRONMENT_VARIABLE'}

    mpi_command = _get_mpi_command(master_node_distributed_training_env)

    network_interface_name = master_node_distributed_training_env.network_interface_name
    assert "mpirun" in mpi_command
    assert "--allow-run-as-root" in mpi_command
    assert "-host algo-1:128,algo-2:128" in mpi_command
    assert "-mca btl_tcp_if_include {}".format(network_interface_name) in mpi_command
    assert "-mca oob_tcp_if_include {}".format(network_interface_name) in mpi_command
    assert "-x PATH" in mpi_command
    assert "-x LD_LIBRARY_PATH" in mpi_command
    assert "-x LD_PRELOAD={}".format(_CHANGE_HOSTNAME_LIBRARY) in mpi_command
    assert "-mca orte_abort_on_non_zero_status 1" in mpi_command
    assert "-x NCCL_SOCKET_IFNAME={}".format(network_interface_name) in mpi_command
    assert "-np 1024" in mpi_command
    assert "-X MY_ENVIRONMENT_VARIABLE" in mpi_command


def test_get_mpi_command_with_gpus(master_node_distributed_training_env):
    master_node_distributed_training_env.available_gpus = 4

    mpi_command = _get_mpi_command(master_node_distributed_training_env)

    assert "algo-1:4,algo-2:4" in mpi_command


def test_get_mpi_command_with_num_processes(master_node_distributed_training_env):
    master_node_distributed_training_env.hyperparameters['num_processes'] = 8

    mpi_command = _get_mpi_command(master_node_distributed_training_env)

    assert "-np 8" in mpi_command


def test_get_mpi_command_with_process_slots_per_host(master_node_distributed_training_env):
    master_node_distributed_training_env.hyperparameters['process_slots_per_host'] = 16

    mpi_command = _get_mpi_command(master_node_distributed_training_env)

    assert "algo-1:16,algo-2:16" in mpi_command


def test_get_mpi_command_with_additional_mpi_options(master_node_distributed_training_env):
    another_mpi_option = "-x MY_ENVIRONMENT_VARIABLE"
    master_node_distributed_training_env.hyperparameters['additional_mpi_options'] = another_mpi_option

    mpi_command = _get_mpi_command(master_node_distributed_training_env)

    assert another_mpi_option in mpi_command


def test_start_ssh_daemon():
    with patch('subprocess.Popen') as mock_popen:

        _start_ssh_daemon()

        mock_popen.assert_called_with(["/usr/sbin/sshd", "-D"])


def test_wait_for_training_to_finish(worker_node_distributed_training_env):
    with patch('chainer_framework.training._wait_for_mpi_to_start_running') as mock_wait_for_mpi_to_start_running, \
         patch('chainer_framework.training._wait_until_mpi_stops_running') as mock_wait_until_mpi_stops_running:

        _wait_for_training_to_finish(worker_node_distributed_training_env)

        mock_wait_for_mpi_to_start_running.assert_called_once()
        mock_wait_until_mpi_stops_running.assert_called_once()


def test_wait_for_mpi_to_start_running():
    with patch('os.path.isfile') as mock_isfile, patch('time.sleep') as mock_sleep:
        mock_isfile.side_effect = [False, False, True]

        _wait_for_mpi_to_start_running()
        mock_isfile.assert_has_calls([call(_MPI_IS_RUNNING), call(_MPI_IS_RUNNING), call(_MPI_IS_RUNNING)])

        assert len(mock_isfile.call_args_list) == 3


def test_wait_until_mpi_stops_running():
    with patch('os.path.isfile') as mock_isfile, patch('time.sleep') as mock_sleep:
        mock_isfile.side_effect = [False, False, True]

        _wait_until_mpi_stops_running()

        mock_isfile.assert_has_calls([call(_MPI_IS_FINISHED), call(_MPI_IS_FINISHED), call(_MPI_IS_FINISHED)])
        assert mock_isfile.call_count == 3


def test_wait_for_worker_nodes_to_start_sshd(master_node_distributed_training_env):
    with patch('chainer_framework.training._can_connect') as mock_can_connect, patch('time.sleep') as mock_sleep:
        hosts = [host for host in master_node_distributed_training_env.hosts
                 if host != master_node_distributed_training_env.current_host]
        mock_can_connect.side_effect = [False, False, True]

        _wait_for_worker_nodes_to_start_sshd(hosts)

        assert mock_can_connect.call_count == 3


def test_wait_for_worker_nodes_to_start_sshd_timeout(master_node_distributed_training_env):
    with patch('chainer_framework.training._can_connect') as mock_can_connect:
        hosts = [host for host in master_node_distributed_training_env.hosts
                 if host != master_node_distributed_training_env.current_host]
        mock_can_connect.return_value = False

        with pytest.raises(TimeoutError):
            _wait_for_worker_nodes_to_start_sshd(hosts, interval=0.001, timeout_in_seconds=0.0001)


def test_get_master_host_name(master_node_distributed_training_env):

    master_host_name = _get_master_host_name(master_node_distributed_training_env.hosts)

    assert master_host_name == "algo-1"


def test_can_connect():
    mock_socket = MagicMock(spec=['connect', 'close'])
    mock_socket.connect.side_effect = [socket.error('expected'), socket.error('expected'), None]

    first_call = _can_connect('algo-2', 2222, mock_socket)
    second_call = _can_connect('algo-2', 2222, mock_socket)
    third_call = _can_connect('algo-2', 2222, mock_socket)

    assert first_call == False
    assert second_call == False
    assert third_call == True
    assert mock_socket.connect.call_count == 3


def test_use_mpi():
    assert _use_mpi({'use_mpi': 'True'}, [])
    assert _use_mpi({'use_mpi': 'true'}, [])
    assert _use_mpi({}, ['algo-1', 'algo-2'])
    assert not _use_mpi({'use_mpi': 'False'}, ['algo-1'])
    assert not _use_mpi({'use_mpi': 'false'}, [])
    assert not _use_mpi({}, ['algo-1'])


def test_num_processes():
    # hyperparameters, process_slots_per_host, num_hosts
    assert _num_processes({'num_processes': 16}, 2, 3) == 16
    assert _num_processes({}, 2, 3) == 6


def test_process_slots_per_host():
    assert _process_slots_per_host({}, 0) == 1
    assert _process_slots_per_host({}, 1) == 1
    assert _process_slots_per_host({}, 2) == 2
    assert _process_slots_per_host({'process_slots_per_host': 1}, 2) == 1
    assert _process_slots_per_host({'process_slots_per_host': 200}, 1) == 200


def test_additional_mpi_options():
    assert '' == _additional_mpi_options({})
    assert "-X MY_ENVIRONMENT_VARIABLE" == _additional_mpi_options({'additional_mpi_options':
                                                                    "-X MY_ENVIRONMENT_VARIABLE"})

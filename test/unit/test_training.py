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
import os
import shlex
import socket
import tempfile

import chainer
import chainer.links as L
from mock import call, MagicMock, patch
import pytest

from chainer_framework import timeout, training


# pylint: disable=protected-access


@pytest.fixture(name='master_node_distributed_training_env')
def fixture_master_node_distributed_training_env():
    env = MagicMock()
    env.current_host = 'algo-1'
    env.hosts = ['algo-1', 'algo-2']
    env.hyperparameters = {}
    env.network_interface_name = "foonetwork"
    env.num_gpus = 4
    return env


@pytest.fixture(name='worker_node_distributed_training_env')
def fixture_worker_node_distributed_training_env():
    env = MagicMock()
    env.current_host = 'algo-2'
    env.hosts = ['algo-1', 'algo-2']
    env.hyperparameters = {}
    env.num_gpus = 4
    return env


@pytest.fixture(name='single_machine_training_env')
def fixture_single_machine_training_env():
    env = MagicMock()
    env.current_host = 'algo-1'
    env.hosts = ['algo-1']
    env.hyperparameters = {}
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, 'model'))
    env.model_dir = os.path.join(tmp, 'model')
    env.num_gpus = 1
    return env


@pytest.fixture(name='training_state')
def fixture_training_state():
    training_state = MagicMock()
    training_state.trained = False
    training_state.saved = False
    return training_state


@pytest.fixture(name='user_module')
def fixture_user_module():
    return MagicMock(spec=['train'])


class DummyModel(chainer.Chain):
    def __init__(self):
        super(DummyModel, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None)

    def __call__(self, x):
        return self.l1()


def test_single_machine_train(single_machine_training_env, user_module, training_state):
    def user_module_train():
        training_state.trained = True
        training_state.model = chainer.Chain()
        return training_state.model

    user_module.train = user_module_train

    training.train(user_module, single_machine_training_env)

    assert training_state.trained


def test_distributed_training_from_master_node(master_node_distributed_training_env, user_module):
    start_ssh_daemon_import = 'chainer_framework.training._wait_for_worker_nodes_to_start_sshd'
    with patch('chainer_framework.training._change_hostname') as mock_change_hostname, \
            patch('chainer_framework.training._start_ssh_daemon') as mock_start_ssh_daemon, \
            patch(start_ssh_daemon_import) as mock_wait_for_sshd, \
            patch('chainer_framework.training._run_mpi_on_all_nodes') as mock_run_mpi_on_all_nodes:
        training.train(user_module, master_node_distributed_training_env)

        mock_change_hostname.assert_called_once_with(
            master_node_distributed_training_env.current_host)
        mock_start_ssh_daemon.assert_called_once()
        mock_wait_for_sshd.assert_called_once_with(
            master_node_distributed_training_env.hosts)
        mock_run_mpi_on_all_nodes.assert_called_once_with(
            master_node_distributed_training_env)


def test_distributed_training_from_worker_node(worker_node_distributed_training_env, user_module):
    wait_for_training_to_finish_import = 'chainer_framework.training._wait_for_training_to_finish'
    with patch('chainer_framework.training._change_hostname') as mock_change_hostname, \
            patch('chainer_framework.training._start_ssh_daemon') as mock_start_ssh_daemon, \
            patch(wait_for_training_to_finish_import) as mock_wait_for_training_to_finish:
        training.train(user_module, worker_node_distributed_training_env)

        mock_change_hostname.assert_called_once_with(
            worker_node_distributed_training_env.current_host)
        mock_start_ssh_daemon.assert_called_once()
        mock_wait_for_training_to_finish.assert_called_once_with(
            worker_node_distributed_training_env)


def test_change_hostname(single_machine_training_env):
    with patch('os.system') as mock_system:
        training._change_hostname(single_machine_training_env.current_host)
        mock_system.assert_called_with(
            "change-hostname.sh {}".format(single_machine_training_env.current_host))


def test_run_mpi_on_all_nodes(master_node_distributed_training_env):
    with patch('subprocess.check_call') as mock_check_call:
        training._run_mpi_on_all_nodes(master_node_distributed_training_env)
        mock_check_call.assert_called_with(
            shlex.split(training._get_mpi_command(master_node_distributed_training_env)))


def test_get_mpi_command(master_node_distributed_training_env):
    network_interface_name = 'foonetwork'
    master_node_distributed_training_env.resource_config = {
        'network_interface_name': network_interface_name
    }

    mpi_command = training._get_mpi_command(master_node_distributed_training_env)

    assert "mpirun" in mpi_command
    assert "--allow-run-as-root" in mpi_command
    assert "-host algo-1:4,algo-2:4" in mpi_command
    assert "-mca btl_tcp_if_include {}".format(network_interface_name) in mpi_command
    assert "-mca oob_tcp_if_include {}".format(network_interface_name) in mpi_command
    assert "-x PATH" in mpi_command
    assert "-x LD_LIBRARY_PATH" in mpi_command
    assert "-x LD_PRELOAD={}".format(training._CHANGE_HOSTNAME_LIBRARY) in mpi_command
    assert "-mca orte_abort_on_non_zero_status 1" in mpi_command
    assert "-x NCCL_SOCKET_IFNAME={}".format(network_interface_name) in mpi_command
    assert "-np 8" in mpi_command


def test_get_mpi_command_with_gpus(master_node_distributed_training_env):
    master_node_distributed_training_env.num_gpus = 4

    mpi_command = training._get_mpi_command(master_node_distributed_training_env)
    assert "algo-1:4,algo-2:4" in mpi_command


def test_get_mpi_command_with_num_processes(master_node_distributed_training_env):
    master_node_distributed_training_env.hyperparameters['num_processes'] = 8

    mpi_command = training._get_mpi_command(master_node_distributed_training_env)

    assert "-np 8" in mpi_command


def test_get_mpi_command_with_process_slots_per_host(master_node_distributed_training_env):
    master_node_distributed_training_env.hyperparameters['process_slots_per_host'] = 16

    mpi_command = training._get_mpi_command(master_node_distributed_training_env)

    assert "algo-1:16,algo-2:16" in mpi_command


def test_get_mpi_command_with_additional_mpi_options(master_node_distributed_training_env):
    another_mpi_option = "-x MY_ENVIRONMENT_VARIABLE"
    master_node_distributed_training_env.hyperparameters[
        'additional_mpi_options'] = another_mpi_option

    mpi_command = training._get_mpi_command(master_node_distributed_training_env)

    assert another_mpi_option in mpi_command


def test_start_ssh_daemon():
    with patch('subprocess.Popen') as mock_popen:
        training._start_ssh_daemon()

        mock_popen.assert_called_with(["/usr/sbin/sshd", "-D"])


def test_wait_for_training_to_finish(worker_node_distributed_training_env):
    wait_for_mpi_import = 'chainer_framework.training._wait_for_mpi_to_start_running'
    wait_until_mpi_stops_import = 'chainer_framework.training._wait_until_mpi_stops_running'
    with patch(wait_for_mpi_import) as mock_wait_for_mpi_to_start_running, \
            patch(wait_until_mpi_stops_import) as mock_wait_until_mpi_stops_running:
        training._wait_for_training_to_finish(worker_node_distributed_training_env)

        mock_wait_for_mpi_to_start_running.assert_called_once()
        mock_wait_until_mpi_stops_running.assert_called_once()


def test_wait_for_mpi_to_start_running():
    with patch('os.path.isfile') as mock_isfile, patch('time.sleep'):
        mock_isfile.side_effect = [False, False, True]

        training._wait_for_mpi_to_start_running()
        mock_isfile.assert_has_calls(
            [call(training._MPI_IS_RUNNING), call(training._MPI_IS_RUNNING),
             call(training._MPI_IS_RUNNING)])

        assert len(mock_isfile.call_args_list) == 3


def test_wait_until_mpi_stops_running():
    with patch('os.path.isfile') as mock_isfile, patch('time.sleep'):
        mock_isfile.side_effect = [False, False, True]

        training._wait_until_mpi_stops_running()

        mock_isfile.assert_has_calls(
            [call(training._MPI_IS_FINISHED), call(training._MPI_IS_FINISHED),
             call(training._MPI_IS_FINISHED)])
        assert mock_isfile.call_count == 3


def test_wait_for_worker_nodes_to_start_sshd(master_node_distributed_training_env):
    with patch('chainer_framework.training._can_connect') as mock_can_connect, patch('time.sleep'):
        hosts = [host for host in master_node_distributed_training_env.hosts
                 if host != master_node_distributed_training_env.current_host]
        mock_can_connect.side_effect = [False, False, True]

        training._wait_for_worker_nodes_to_start_sshd(hosts)

        assert mock_can_connect.call_count == 3


def test_wait_for_worker_nodes_to_start_sshd_timeout(master_node_distributed_training_env):
    with patch('chainer_framework.training._can_connect') as mock_can_connect:
        hosts = [host for host in master_node_distributed_training_env.hosts
                 if host != master_node_distributed_training_env.current_host]
        mock_can_connect.return_value = False

        with pytest.raises(timeout.TimeoutError):
            training._wait_for_worker_nodes_to_start_sshd(hosts, interval=0.001,
                                                          timeout_in_seconds=0.0001)


def test_get_master_host_name(master_node_distributed_training_env):
    master_host_name = training._get_master_host_name(master_node_distributed_training_env.hosts)

    assert master_host_name == "algo-1"


def test_can_connect():
    mock_socket = MagicMock(spec=['connect', 'close'])
    mock_socket.connect.side_effect = [socket.error('expected'), socket.error('expected'), None]

    first_call = training._can_connect('algo-2', 2222, mock_socket)
    second_call = training._can_connect('algo-2', 2222, mock_socket)
    third_call = training._can_connect('algo-2', 2222, mock_socket)

    assert not first_call
    assert not second_call
    assert third_call
    assert mock_socket.connect.call_count == 3

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
import socket
import sys
import tempfile

import chainer
import chainer.links as L
from mock import call, MagicMock, patch, mock_open
import pytest
from six import PY2

from chainer_framework import timeout, training


# pylint: disable=protected-access


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


def mock_training_env(current_host='algo-1', hosts=None, hyperparameters=None,
                      module_dir='s3://my/script', module_name='imagenet', **kwargs):
    hosts = hosts or ['algo-1']

    hyperparameters = hyperparameters or {}

    return MagicMock(current_host=current_host, hosts=hosts, hyperparameters=hyperparameters,
                     module_dir=module_dir, module_name=module_name, **kwargs)


@patch('sagemaker_containers.beta.framework.modules.run_module_from_s3')
def test_single_machine(run_module_from_s3):

    env = mock_training_env()
    training.train(env, {})

    run_module_from_s3.assert_called_with('s3://my/script', env.to_cmd_args(),
                                          env.to_env_vars(), 'imagenet')


@patch('chainer_framework.training._change_hostname')
@patch('chainer_framework.training._start_ssh_daemon')
@patch('chainer_framework.training._wait_for_worker_nodes_to_start_sshd')
@patch('chainer_framework.training._run_mpi_on_all_nodes')
@patch('chainer_framework.training._create_mpi_script')
def test_distributed_training_from_master_node(
        _create_mpi_script,
        _run_mpi_on_all_nodes,
        _wait_for_worker_nodes_to_start_sshd,
        _start_ssh_daemon,
        _change_hostname):
    hosts = ['algo-1', 'algo-2']
    env = mock_training_env(hosts=hosts)
    training.train(env, {})

    _create_mpi_script.assert_called_with(env)
    _change_hostname.assert_called_once_with('algo-1')
    _start_ssh_daemon.assert_called_once()
    _wait_for_worker_nodes_to_start_sshd.assert_called_once_with(hosts)
    _run_mpi_on_all_nodes.assert_called_once_with(env, {})


builtins_open = '__builtin__.open' if PY2 else 'builtins.open'


@patch('chainer_framework.training._change_hostname')
@patch('chainer_framework.training._start_ssh_daemon')
@patch('chainer_framework.training._wait_for_worker_nodes_to_start_sshd')
@patch('subprocess.check_call')
@patch('sagemaker_containers.beta.framework.modules.download_and_install')
@patch('os.stat')
@patch('os.chmod')
@patch(builtins_open, mock_open())
def test_distributed_training_from_master_node_use_mpi(
        chmod, stat, download_and_install, check_call,
        _wait_for_worker_nodes_to_start_sshd, _start_ssh_daemon, _change_hostname):
    hosts = ['algo-1', 'algo-2']
    env = mock_training_env(hosts=hosts, num_gpus=0, network_interface_name='foonet')
    training.train(env, {'sagemaker_use_mpi': True})

    download_and_install.assert_called_with('s3://my/script')
    _change_hostname.assert_called_once_with('algo-1')
    _start_ssh_daemon.assert_called_once()
    _wait_for_worker_nodes_to_start_sshd.assert_called_once_with(hosts)
    check_call.assert_called_once_with(
        ['mpirun', '--allow-run-as-root', '--host', 'algo-1,algo-2', '-mca', 'btl_tcp_if_include',
         'foonet', '-mca',
         'oob_tcp_if_include', 'foonet',
         '-mca', 'btl', '^openib', '-x', 'PATH', '-x', 'LD_LIBRARY_PATH', '-x',
         'LD_PRELOAD=/libchangehostname.so', '-mca', 'orte_abort_on_non_zero_status', '1', '-x',
         'NCCL_DEBUG=INFO', '-x', 'NCCL_SOCKET_IFNAME=foonet',
         '-np', '2', '/mpi_script.sh'])
    chmod.assert_called_with('/mpi_script.sh', stat().st_mode.__or__())


@patch('chainer_framework.training._change_hostname')
@patch('chainer_framework.training._start_ssh_daemon')
@patch('chainer_framework.training._wait_for_worker_nodes_to_start_sshd')
@patch('subprocess.check_call')
@patch('sagemaker_containers.beta.framework.modules.download_and_install')
@patch('os.stat')
@patch('os.chmod')
@patch(builtins_open, mock_open())
def test_distributed_training_from_master_node_use_mpi_with_gpus(
        chmod, stat, download_and_install, check_call,
        _wait_for_worker_nodes_to_start_sshd, _start_ssh_daemon, _change_hostname):
    hosts = ['algo-1', 'algo-2']
    env = mock_training_env(hosts=hosts, num_gpus=8, network_interface_name='foonet')
    training.train(env, {'sagemaker_use_mpi': True})

    download_and_install.assert_called_with('s3://my/script')
    _change_hostname.assert_called_once_with('algo-1')
    _start_ssh_daemon.assert_called_once()
    _wait_for_worker_nodes_to_start_sshd.assert_called_once_with(hosts)
    check_call.assert_called_once_with(
        ['mpirun', '--allow-run-as-root', '--host', 'algo-1:8,algo-2:8', '-mca',
         'btl_tcp_if_include',
         'foonet', '-mca',
         'oob_tcp_if_include', 'foonet',
         '-mca', 'btl', '^openib', '-x', 'PATH', '-x', 'LD_LIBRARY_PATH', '-x',
         'LD_PRELOAD=/libchangehostname.so', '-mca', 'orte_abort_on_non_zero_status', '1', '-x',
         'NCCL_DEBUG=INFO', '-x', 'NCCL_SOCKET_IFNAME=foonet',
         '-np', '16', '/mpi_script.sh'])
    chmod.assert_called_with('/mpi_script.sh', stat().st_mode.__or__())


@patch('chainer_framework.training._change_hostname')
@patch('chainer_framework.training._start_ssh_daemon')
@patch('chainer_framework.training._wait_for_worker_nodes_to_start_sshd')
@patch('subprocess.check_call')
@patch('sagemaker_containers.beta.framework.modules.download_and_install')
@patch('os.stat')
@patch('os.chmod')
@patch(builtins_open, mock_open())
def test_distributed_training_from_master_node_use_mpi_with_slot_processes_per_host(
        chmod, stat, download_and_install, check_call,
        _wait_for_worker_nodes_to_start_sshd, _start_ssh_daemon, _change_hostname):
    hosts = ['algo-1', 'algo-2']
    env = mock_training_env(hosts=hosts, num_gpus=8, network_interface_name='foonet')
    training.train(env, {'sagemaker_use_mpi': True, 'sagemaker_process_slots_per_host': 16})

    download_and_install.assert_called_with('s3://my/script')
    _change_hostname.assert_called_once_with('algo-1')
    _start_ssh_daemon.assert_called_once()
    _wait_for_worker_nodes_to_start_sshd.assert_called_once_with(hosts)
    check_call.assert_called_once_with(
        ['mpirun', '--allow-run-as-root', '--host', 'algo-1:16,algo-2:16', '-mca',
         'btl_tcp_if_include',
         'foonet', '-mca',
         'oob_tcp_if_include', 'foonet',
         '-mca', 'btl', '^openib', '-x', 'PATH', '-x', 'LD_LIBRARY_PATH', '-x',
         'LD_PRELOAD=/libchangehostname.so', '-mca', 'orte_abort_on_non_zero_status', '1', '-x',
         'NCCL_DEBUG=INFO', '-x', 'NCCL_SOCKET_IFNAME=foonet',
         '-np', '32', '/mpi_script.sh'])
    chmod.assert_called_with('/mpi_script.sh', stat().st_mode.__or__())


@patch('subprocess.Popen')
@patch('sagemaker_containers.beta.framework.modules.download_and_install')
@patch('os.stat')
@patch('os.chmod')
@patch(builtins_open, mock_open())
@patch('os.path.isfile')
def test_distributed_training_from_worker_node_use_mpi(
        isfile, chmod, stat, download_and_install, popen):
    hosts = ['algo-1', 'algo-2']
    env = mock_training_env(current_host='algo-2', hosts=hosts)
    training.train(env, {'sagemaker_use_mpi': True})

    download_and_install.assert_called_with('s3://my/script')
    popen.assert_called_with(['/usr/sbin/sshd', '-D'])

    isfile.assert_called_with('/mpi_is_finished')
    chmod.assert_called_once_with('/mpi_script.sh', stat().st_mode.__or__())


@patch('os.system')
@patch('chainer_framework.training._start_ssh_daemon')
@patch('sagemaker_containers.beta.framework.modules.download_and_install')
@patch('os.stat')
@patch('os.chmod')
@patch(builtins_open, mock_open())
@patch('os.path.isfile')
def test_distributed_training_from_worker_node(
        isfile, chmod, stat, download_and_install,
        _start_ssh_daemon, system):
    hosts = ['algo-1', 'algo-2']
    env = mock_training_env(current_host='algo-2', hosts=hosts)
    training.train(env, {})

    download_and_install.assert_called_with('s3://my/script')
    system.assert_called_once_with('change-hostname.sh algo-2')
    _start_ssh_daemon.assert_called_once()

    isfile.assert_called_with('/mpi_is_finished')
    chmod.assert_called_once_with('/mpi_script.sh', stat().st_mode.__or__())


@patch('os.system')
@patch('subprocess.Popen')
@patch('chainer_framework.training._can_connect', return_value=True)
@patch('time.sleep')
@patch('subprocess.check_call')
@patch('sagemaker_containers.beta.framework.modules.download_and_install')
@patch('os.stat')
@patch('os.chmod')
@patch(builtins_open, mock_open())
@patch('socket.socket')
def test_distributed_training_from_worker_node_use_mpi_with_sagemaker_additional_mpi_options(
        socket, chmod, stat, download_and_install, check_call, sleep, _can_connect, popen, system):
    hosts = ['algo-1', 'algo-2']
    env = mock_training_env(hosts=hosts, num_gpus=8, network_interface_name='foonet')
    training.train(env, {'sagemaker_use_mpi': True, 'sagemaker_process_slots_per_host': 16,
                         'sagemaker_additional_mpi_options': '-x MY_ENVIRONMENT_VARIABLE'})

    download_and_install.assert_called_with('s3://my/script')
    system.assert_called_once_with('change-hostname.sh algo-1')
    popen.assert_called_once_with(["/usr/sbin/sshd", "-D"])
    _can_connect.assert_called_with('algo-2', 22, socket())
    check_call.assert_called_once_with(
        ['mpirun', '--allow-run-as-root', '--host', 'algo-1:16,algo-2:16', '-mca',
         'btl_tcp_if_include',
         'foonet', '-mca',
         'oob_tcp_if_include', 'foonet',
         '-mca', 'btl', '^openib', '-x', 'PATH', '-x', 'LD_LIBRARY_PATH', '-x',
         'LD_PRELOAD=/libchangehostname.so', '-mca', 'orte_abort_on_non_zero_status', '1', '-x',
         'NCCL_DEBUG=INFO', '-x', 'NCCL_SOCKET_IFNAME=foonet',
         '-np', '32', '-x', 'MY_ENVIRONMENT_VARIABLE', '/mpi_script.sh'])

    chmod.assert_called_with('/mpi_script.sh', stat().st_mode.__or__())

    open().write.assert_called_with("""#!/usr/bin/env bash
touch /mpi_is_running
%s -m mpi4py -m imagenet
EXIT_CODE=$?
touch /mpi_is_finished
exit ${EXIT_CODE}
""" % sys.executable)


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


def test_wait_for_worker_nodes_to_start_sshd():
    with patch('chainer_framework.training._can_connect') as mock_can_connect, patch('time.sleep'):
        mock_can_connect.side_effect = [False, False, True]

        training._wait_for_worker_nodes_to_start_sshd(['algo-2'])

        assert mock_can_connect.call_count == 3


def test_wait_for_worker_nodes_to_start_sshd_timeout(master_node_distributed_training_env):
    with patch('chainer_framework.training._can_connect') as mock_can_connect:
        hosts = [host for host in master_node_distributed_training_env.hosts
                 if host != master_node_distributed_training_env.current_host]
        mock_can_connect.return_value = False

        with pytest.raises(timeout.TimeoutError):
            training._wait_for_worker_nodes_to_start_sshd(hosts, interval=0.001,
                                                          timeout_in_seconds=0.0001)


@patch('chainer_framework.training._can_connect', return_value=False)
@patch('time.sleep')
@patch('socket.socket')
def test_wait_for_worker_nodes_to_start_sshd_timeout(socket, sleep, _can_connect):
    hosts = ['algo-1', 'algo-2']
    mock_training_env(hosts=hosts, num_gpus=8, network_interface_name='foonet')

    with pytest.raises(timeout.TimeoutError):
        training._wait_for_worker_nodes_to_start_sshd(hosts, interval=0.001,
                                                      timeout_in_seconds=0.0001)

    socket.assert_called()
    sleep.assert_called()


@patch('chainer_framework.training._can_connect', side_effect=[False, False, True])
@patch('time.sleep')
@patch('socket.socket')
def test_wait_for_worker_nodes_to_start_sshd_timeout(socket, sleep, _can_connect):
    training._wait_for_worker_nodes_to_start_sshd(['algo-2'])

    assert _can_connect.call_count == 3
    socket.assert_called()
    sleep.assert_called()


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

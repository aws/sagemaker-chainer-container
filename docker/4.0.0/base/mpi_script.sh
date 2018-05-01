#!/usr/bin/env bash
# For distributed training: the 'master node' runs mpirun with this script, '/mpi_script.sh'
# This script creates a file '/mpi_is_running' that worker nodes use to determine whether training (started by MPI from
# the master node) is still running. Processes on worker nodes use /mpi_is_finished file to determine when to exit.
touch /mpi_is_running
python -m mpi4py -m chainer_framework.training
EXIT_CODE=$?
touch /mpi_is_finished

exit ${EXIT_CODE}

#!/usr/bin/env bash
touch /mpi_is_running &&
 python -m chainer_framework.run_training &&
 rm /mpi_is_running
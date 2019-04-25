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
from __future__ import absolute_import

import os

import chainermn
import mpi4py.MPI
import sagemaker_containers


if __name__ == '__main__':
    mpi_comm = mpi4py.MPI.COMM_WORLD

    if mpi_comm.rank == 0:
        env = sagemaker_containers.training_env()
        open(os.path.join(env.output_data_dir, 'this_file_is_expected'), 'a').close()
        raise ValueError('this failure is expected')
    chainermn.create_communicator('naive', mpi_comm)

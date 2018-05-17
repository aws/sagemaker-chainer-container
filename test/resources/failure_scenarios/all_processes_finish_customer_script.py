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
import time

import chainermn


def train(hyperparameters, num_gpus, current_host, output_data_dir, hosts):
    communicator = hyperparameters.get('communicator', 'naive' if num_gpus == 0 else 'pure_nccl')
    comm = chainermn.create_communicator(communicator)

    print('process {} on host {} of {} starting'.format(comm.intra_rank, current_host, len(hosts)))

    if comm.intra_rank == 1 and current_host != 'algo-1':
        os.makedirs(output_data_dir)
        # this sleep time must be longer than the polling interval to check if mpi is finished.
        time.sleep(6)
        open(os.path.join(output_data_dir, 'process_could_complete'), 'a').close()

    print('process {} on host {} exiting'.format(comm.intra_rank, current_host))

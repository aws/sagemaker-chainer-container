# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import argparse
import os
import time

import chainermn
import sagemaker_containers

if __name__ == '__main__':
    env = sagemaker_containers.training_env()

    parser = argparse.ArgumentParser()

    parser.add_argument('--num-gpus', type=int, default=env.num_gpus)
    parser.add_argument('--communicator', type=str,
                        default='naive' if env.num_gpus == 0 else 'pure_nccl')
    parser.add_argument('--current_host', type=str, default=env.current_host)
    parser.add_argument('--hosts', type=str, default=env.hosts)
    parser.add_argument('--output-data-dir', type=str, default=env.output_data_dir)

    args = parser.parse_args()

    comm = chainermn.create_communicator(args.communicator)

    num_hosts = len(args.hosts)
    print('process %s on host %s of %s starting' % (comm.intra_rank, args.current_host, num_hosts))

    if comm.intra_rank == 1 and args.current_host != args.hosts[0]:
        host_output_prefix = os.path.join(args.output_data_dir, args.current_host[:args.current_host.rfind('-')])
        os.makedirs(host_output_prefix)

        # this sleep time must be longer than the polling interval to check if mpi is finished.
        print('process %s on host %s of %s sleeping' % (comm.intra_rank, args.current_host, num_hosts))

        time.sleep(20)
        open(os.path.join(host_output_prefix, 'process_could_complete'), 'a').close()

    print('process {} on host {} exiting'.format(comm.intra_rank, args.current_host))

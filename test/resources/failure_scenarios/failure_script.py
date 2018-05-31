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
from __future__ import print_function, absolute_import

import argparse

import chainermn
import sagemaker_containers

if __name__ == '__main__':
    env = sagemaker_containers.training_env()

    parser = argparse.ArgumentParser()

    parser.add_argument('--num-gpus', type=int, default=env.num_gpus)
    parser.add_argument('--communicator', type=str,
                        default='naive' if env.num_gpus == 0 else 'pure_nccl')
    parser.add_argument('--hosts', type=str, default=env.hosts)
    parser.add_argument('--node_to_fail', type=int)

    args = parser.parse_args()

    if len(args.hosts) == 1:
        raise Exception('Exception on a single machine')

    comm = chainermn.create_communicator(args.communicator)

    # When running in local mode, setting rank to 'inter_rank' simulates multi-node training.
    rank = comm.inter_rank

    if args.node_to_fail == rank:
        print('exception from node {}'.format(rank))
        raise Exception('exception from node {}'.format(rank))

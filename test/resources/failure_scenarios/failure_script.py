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
from __future__ import print_function

import chainermn


def train(hyperparameters, num_gpus, hosts, current_host):  # pylint: disable=unused-argument
    if len(hosts) == 1:
        raise Exception('Exception on a single machine')

    communicator = hyperparameters.get(
        'communicator', 'naive' if num_gpus == 0 else 'pure_nccl')
    comm = chainermn.create_communicator(communicator)

    node_to_fail = hyperparameters.get('node_to_fail')

    # When running in local mode, setting rank to 'inter_rank' simulates multi-node training.
    rank = comm.inter_rank

    if node_to_fail == rank:
        raise Exception('exception from node {}'.format(rank))

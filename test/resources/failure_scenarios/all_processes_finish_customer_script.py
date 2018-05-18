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
import argparse
import os
import time

import chainermn
from sagemaker_containers import env

if __name__ == '__main__':
    training_env = env.TrainingEnv()

    parser = argparse.ArgumentParser()

    parser.add_argument('--num-gpus', type=int, default=training_env.num_gpus)
    parser.add_argument('--communicator',  type=str,
                        default='naive' if training_env.num_gpus == 0 else 'pure_nccl')
    parser.add_argument('--current_host', type=str, default=training_env.current_host)
    parser.add_argument('--hosts', type=str, default=training_env.hosts)
    parser.add_argument('--output_data_dir', type=str, default=training_env.output_data_dir)

    args, _ = parser.parse_known_args()

    comm = chainermn.create_communicator(args.communicator)

    num_hosts = len(args.hosts)
    print('process %s on host %s of %s starting' % (comm.intra_rank, args.current_host, num_hosts))

    if comm.intra_rank == 1 and args.current_host != 'algo-1':
        os.makedirs(args.output_data_dir)
        # this sleep time must be longer than the polling interval to check if mpi is finished.
        time.sleep(6)
        open(os.path.join(args.output_data_dir, 'process_could_complete'), 'a').close()

    print('process {} on host {} exiting'.format(comm.intra_rank, args.current_host))

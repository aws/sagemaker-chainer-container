import os
import time

import chainermn


def train(hyperparameters, num_gpus, current_host, output_data_dir, hosts):

    communicator = hyperparameters.get('communicator', 'naive' if num_gpus == 0 else 'pure_nccl')
    comm = chainermn.create_communicator(communicator)

    assert comm.intra_size > 1
    assert len(hosts) > 1

    if comm.intra_rank == 1 and current_host != 'algo-1':
        os.makedirs(output_data_dir)
        # this sleep time must be longer than the polling interval to check if mpi is finished.
        time.sleep(6)
        open(os.path.join(output_data_dir, 'process_could_complete'), 'a').close()

    print('process {} on host {} exiting'.format(comm.intra_rank, current_host))
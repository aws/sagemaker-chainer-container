from __future__ import print_function

import chainermn


def train(hyperparameters, num_gpus, hosts):
    if len(hosts) == 1:
        raise Exception('Exception on a single machine')
    communicator = hyperparameters.get('communicator', 'naive' if num_gpus == 0 else 'pure_nccl')
    comm = chainermn.create_communicator(communicator)
    rank = hyperparameters.get('rank', 'intra_rank')

    node_to_fail = hyperparameters.get('node_to_fail')

    # When running in local mode, setting rank to 'inter_rank' simulates multi-node training.
    if rank == 'inter_rank':
        rank = comm.inter_rank
    else:
        rank = comm.intra_rank

    if node_to_fail == rank:
        raise Exception('exception from node {}'.format(rank))


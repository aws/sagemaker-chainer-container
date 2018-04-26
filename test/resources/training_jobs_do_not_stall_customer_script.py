import mpi4py.MPI

import chainermn


def train():
    mpi_comm = mpi4py.MPI.COMM_WORLD

    if mpi_comm.rank == 0:
        raise ValueError('failure!')
    chainermn.create_communicator('naive', mpi_comm)

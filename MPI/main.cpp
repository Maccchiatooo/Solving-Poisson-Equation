#include "mpi.h"
#include <iostream>
#include <unistd.h>
#include <stdexcept>
#include "poisson.hpp"
#define dim 2
int main(int argc, char *argv[])
{
    int rank, process;
    int px, py;
    int coord[2] = {};
    int cart_num_proc[2];
    int config_e;
    int global_len[2];
    int local_len[2];
    MPI_Comm cart_comm;
    int cmax = 0;
    double start, end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process);

    global_len[0] = 360;
    global_len[1] = 360;

    for (int i = 1; i < process; i++)
    {
        if (process % i == 0)
        {
            px = i;
            py = process / i;
            config_e = 1. / (2. * (global_len[1] * (px - 1) / py + global_len[0] * (py - 1) / px));
        }
        if (config_e >= cmax)
        {
            cmax = config_e;
            cart_num_proc[0] = px;
            cart_num_proc[1] = py;
        }
    }

    int periodic[2] = {1, 1};
    int reorder = 0;
    local_len[0] = global_len[0] / cart_num_proc[0];
    local_len[1] = global_len[1] / cart_num_proc[1];

    MPI_Cart_create(MPI_COMM_WORLD, 2, cart_num_proc, periodic, reorder, &cart_comm);
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Cart_coords(cart_comm, rank, 2, coord);
    MPI_Barrier(MPI_COMM_WORLD);

    Poisson2d p2(rank, global_len, local_len, coord, cart_num_proc);

    p2.init(rank, cart_comm);
    start = MPI_Wtime();
    for (int i = 1; i <= 10000; i++)
    {

        MPI_Barrier(MPI_COMM_WORLD);
        if (i % 1000 == 0)
            p2.error(rank, process);
        p2.exchange();
        p2.update();
        end = MPI_Wtime();

        if (i % 1000 == 0 && rank == 0)
            printf("time=%f\n", end - start);
    }
    MPI_Finalize();
}

#include "mpi.h"
#include <unistd.h>
#include "poisson.hpp"

int main(int argc, char *argv[])
{

    double start, end;
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    {

        Poisson2d p2(MPI_COMM_WORLD);

        p2.init();
        p2.setup_subdomain();

        start = MPI_Wtime();
        for (int i = 1; i <= 10000; i++)
        {

            MPI_Barrier(MPI_COMM_WORLD);
            if (i % 1000 == 0)
                p2.error();
            p2.exchange();
            p2.update();
            end = MPI_Wtime();

            if (i % 1000 == 0 && p2.comm.me == 0)
                printf("time=%f\n", end - start);
        }
    }
    Kokkos::finalize();
    MPI_Finalize();
}
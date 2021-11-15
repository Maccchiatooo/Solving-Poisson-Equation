#include <Kokkos_Core.hpp>
#include <fstream>
#include <iostream>
#include <math.h>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include "ko.hpp"

#define dim 2

#define iter 10000

int main(int argc, char *argv[])
{
    int nx = 359, ny = 359;
    double dx = 1.0 / nx;
    double dy = 1.0 / ny;
    Kokkos::initialize(argc, argv);
    {
        Kokkos::Timer timer;
        double start = timer.seconds();
        Poisson2d p2(nx, ny, dx, dy);
        p2.init();

        for (int i = 1; i <= iter; i++)
        {

            if (i % 1000 == 0)
                p2.error();

            p2.update();
            double end = timer.seconds();
            if (i % 1000 == 0)
            {

                printf("time = %f\n", end - start);
            }
        }
    }

    Kokkos::finalize();
    return 0;
};

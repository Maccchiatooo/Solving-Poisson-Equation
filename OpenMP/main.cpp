#include <math.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include "poisson.hpp"
#include <omp.h>

#define dim 2
#define nx 359
#define ny 359
#define iter 10000
int main(int argc, char *argv[])
{
    double dx = 1.0 / nx;
    double dy = 1.0 / ny;

    double start, end;

    Poisson2d p2(nx, ny, dx, dy);

    p2.init();

    for (int i = 1; i <= iter; i++)
    {
        start = omp_get_wtime();
        if (i % 1000 == 0)
            p2.error();

        p2.update();
        end = omp_get_wtime();

        if (i % 1000 == 0)
            printf("time = %f\n", end - start);
    }
}

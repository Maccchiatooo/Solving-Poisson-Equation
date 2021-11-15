#include <math.h>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include "poisson.hpp"
#include <ctime>

#define dim 2
#define nx 359
#define ny 359
#define iter 10000

int main(int argc, char *argv[])
{
    double dx = 1.0 / nx;
    double dy = 1.0 / ny;
    clock_t start, end;

    Poisson2d p2(nx, ny, dx, dy);
    start = clock();
    p2.init();

    for (int i = 1; i <= iter; i++)
    {

        if (i % 1000 == 0)
            p2.error();

        p2.update();
        end = clock();
        if (i % 1000 == 0)
            printf("time = %f\n", (double)(end - start) / CLOCKS_PER_SEC);
    }
}

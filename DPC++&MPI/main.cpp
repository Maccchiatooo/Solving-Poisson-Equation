#include "mpi.h"
#include <iostream>
#include <unistd.h>
#include <stdexcept>
#include <math.h>
#include <iostream>
#include <ctime>
#include <CL/sycl.hpp>

using namespace sycl;

#define dim 2
#define nx 359
#define ny 359
#define iter 10000

int main(int argc, char *argv[])
{
    // mpi
    int rank, process;
    int px, py;
    int coord[2] = {};
    int cart_num_proc[2];
    int config_e;
    int global_len[2];
    int local_len[2];
    int cmax = 0;
    int m_up, m_down, m_left, m_right;
    MPI_Comm cart_comm;
    MPI_Datatype x_edge_type;
    MPI_Datatype y_edge_type;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process);

    global_len[0] = nx + 1;
    global_len[1] = ny + 1;

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

    printf("local_len=%d,%d\n", local_len[0], local_len[1]);

    MPI_Cart_create(MPI_COMM_WORLD, 2, cart_num_proc, periodic, reorder, &cart_comm);
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Cart_coords(cart_comm, rank, 2, coord);
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "this is " << coord[0] << coord[1] << "rank"
              << "\n";

    printf("px=%d,py=%d\n", cart_num_proc[0], cart_num_proc[1]);

    clock_t start, end;
    size_t lx = local_len[0];
    size_t ly = local_len[1];
    double dx = 1.0 / nx;
    double dy = 1.0 / ny;
    double pi = 3.141592653589793;
    int size = (lx + 2) * (ly + 2);

    // create queue on defult device
    queue Q{};
    std::cout << "Selected device:" << Q.get_device().get_info<info::device::name>() << "\n";

    double *m_cellValue = malloc_shared<double>(size, Q);
    double *m_cellValueNew = malloc_shared<double>(size, Q);
    double *m_fb = malloc_shared<double>(size, Q);
    double *m_exact = malloc_shared<double>(size, Q);
    double *u_exact = malloc_shared<double>(size, Q);
    double *uxxyy_exact = malloc_shared<double>(size, Q);

    start = clock();
    auto ini = Q.submit([&](handler &h)
                        { h.parallel_for(range{lx, ly}, [=](id<2> idx)
                                         {
                                             int id0 = idx[0]+1;
                                             int id1 = idx[1] + 1;

                                             int g_x = coord[0] * local_len[0] + id0-1;
                                             int g_y = coord[1] * local_len[1] + id1-1;
                                             double phi_x = g_x * dx;
                                             double phi_y = g_y * dy;
                                             double u_e = sycl::sin(pi * phi_x * phi_y);
                                             double u_xe = -pi * pi * (pow(phi_x, 2) + pow(phi_y, 2)) * sycl::sin(pi * phi_x * phi_y);
                                             int id = id0 + id1 * (local_len[0] + 2);

                                             if (g_x == 0 || g_x == global_len[0]-1 || g_x == 0 ||  g_y ==global_len[1]-1)
                                             {
                                                 m_fb[id] = u_e;
                                                 m_cellValue[id] = u_e;
                                             }
                                             else
                                             {
                                                 m_fb[id] = -u_xe;
                                             }
                                             m_exact[id] = u_e; }); });
    Q.wait();

    MPI_Cart_shift(cart_comm, 0, 1, &m_left, &m_right);

    MPI_Cart_shift(cart_comm, 1, 1, &m_down, &m_up);

    MPI_Type_vector(local_len[0], 1, 1, MPI_DOUBLE, &x_edge_type);
    MPI_Type_commit(&x_edge_type);

    MPI_Type_vector(local_len[1], 1, (local_len[0] + 2), MPI_DOUBLE, &y_edge_type);
    MPI_Type_commit(&y_edge_type);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("success\n");

    for (int i = 1; i <= 1000; i++)
    {

        if (i % 100 == 0)
        {

            double error = 0.0;

            for (int idx = 1; idx < local_len[0] + 1; idx++)
            {
                for (int idy = 1; idy < local_len[1] + 1; idy++)
                {
                    double g_x = coord[0] * local_len[0] + idx - 1;
                    double g_y = coord[1] * local_len[1] + idy - 1;
                    double phi_x = g_x * dx;
                    double phi_y = g_y * dy;
                    int id = idx + idy * (local_len[0] + 2);

                    error += pow((m_cellValue[id] - sin(pi * phi_x * phi_y)), 2);
                }
            }
            double totalerror = 0;

            MPI_Reduce(&error, &totalerror, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0)
                std::cout << "avg error for is " << sqrt(totalerror / (double)(360 * 360))
                          << std::endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        int tag = 1;
        if (coord[0] != 0)
        {
            MPI_Send(&m_cellValue[local_len[0] + 3], 1, y_edge_type, m_left, tag, MPI_COMM_WORLD);
        }

        if (coord[0] != cart_num_proc[0] - 1)
        {
            MPI_Recv(&m_cellValue[2 * local_len[0] + 3], 1, y_edge_type, m_right, tag, MPI_COMM_WORLD, &status);
        }

        tag = 2;
        if (coord[0] != cart_num_proc[0] - 1)
        {
            MPI_Send(&m_cellValue[2 * local_len[0] + 2], 1, y_edge_type, m_right, tag, MPI_COMM_WORLD);
        }

        if (coord[0] != 0)
        {
            MPI_Recv(&m_cellValue[local_len[0] + 2], 1, y_edge_type, m_left, tag, MPI_COMM_WORLD, &status);
        }

        tag = 3;
        if (coord[1] != cart_num_proc[1] - 1)
        {
            MPI_Send(&m_cellValue[(local_len[0] + 2) * local_len[1] + 1], 1, x_edge_type, m_up, tag, MPI_COMM_WORLD);
        }
        if (coord[1] != 0)
        {
            MPI_Recv(&m_cellValue[1], 1, x_edge_type, m_down, tag, MPI_COMM_WORLD, &status);
        }

        tag = 4;
        if (coord[1] != 0)
        {
            MPI_Send(&m_cellValue[local_len[0] + 3], 1, x_edge_type, m_down, tag, MPI_COMM_WORLD);
        }
        if (coord[1] != cart_num_proc[1] - 1)
        {
            MPI_Recv(&m_cellValue[(local_len[0] + 2) * (local_len[1] + 1) + 1], 1, x_edge_type, m_up, tag, MPI_COMM_WORLD, &status);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        auto upd1 = Q.submit([&](handler &h)
                             { h.parallel_for(range{lx, ly}, [=](id<2> idx)
                                              {

                                          int id0 = idx[0]+1;
                                          int id1 = idx[1] + 1;

                                          int g_x = coord[0] * local_len[0] + id0-1;
                                          int g_y = coord[1] * local_len[1] + id1-1;
                                          double phi_x = g_x * dx;
                                          double phi_y = g_y * dy;
                                          double u_e = sycl::sin(pi * phi_x * phi_y);
                                          double u_xe = -pi * pi * (pow(phi_x, 2) + pow(phi_y, 2)) * sycl::sin(pi * phi_x * phi_y);
                                          int id = id0 + id1 * (local_len[0] + 2);

     if (g_x == 0 || g_x == global_len[0]-1 || g_x == 0 ||  g_y ==global_len[1]-1)
     {
         m_cellValueNew[id] = u_e;
     }else{
          m_cellValueNew[id] = 0.25 * (m_cellValue[id0 - 1 + id1 * (local_len[0] + 2)] + m_cellValue[id0 + 1 + id1 * (local_len[0] + 2)] +
                            m_cellValue[id0 + (id1 - 1) * (local_len[0] + 2)] + m_cellValue[id0 + (id1 + 1) * (local_len[0] + 2)] +
                            m_fb[id] * dx * dy);

     } }); });

        auto upd2 = Q.submit([&](handler &h)
                             {

         h.depends_on(upd1);
         h.parallel_for(range{lx , ly }, [=](id<2> idx)
                        {
                             int id0 = idx[0]+1;
                             int id1 = idx[1] + 1;
                             int id = id0 + id1 * (local_len[0] + 2);
                            m_cellValue[id] = m_cellValueNew[id]; }); });
        upd2.wait();
        end = clock();
        if (rank == 0 && i % 100 == 0)
            printf("time = %f\n", (double)(end - start) / CLOCKS_PER_SEC);
    }
}

#ifndef _poisson_H_
#define _poisson_H_

#include <mpi.h>
#include <fstream>
#include <iostream>
class Poisson2d
{

public:
    Poisson2d(int rank, int *global_len, int *local_len, int *coord, int *cart_num_proc)
        : m_rank(rank),
          g_len(global_len),
          m_len(local_len),
          m_x(coord[0]),
          m_y(coord[1]),
          m_cart(cart_num_proc)
    {

        dx = 1.0 / (g_len[0] - 1);
        dy = 1.0 / (g_len[1] - 1);

        m_cellValue = (double *)malloc(sizeof(double) * (m_len[0] + 2) * (m_len[1] + 2));
        m_cellValueNew = (double *)malloc(sizeof(double) * (m_len[0] + 2) * (m_len[1] + 2));
        m_fb = (double *)malloc(sizeof(double) * (m_len[0] + 2) * (m_len[1] + 2));
        m_exact = (double *)malloc(sizeof(double) * (m_len[0] + 2) * (m_len[1] + 2));

        for (int j = 0; j <= m_len[1] + 1; j++)
        {
            for (int i = 0; i <= m_len[0] + 1; i++)
            {
                m_cellValue[l2i(i, j)] = 0;
                m_cellValueNew[l2i(i, j)] = 0;
                m_fb[l2i(i, j)] = 0;
                m_exact[l2i(i, j)] = 0;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    inline int l2i(int x, int y) const
    {

        return x + y * (this->m_len[0] + 2);
    }

    void init(int rank, MPI_Comm &cart_comm);

    void exchange();
    void update();
    void output();
    void error(int rank, int process);

    int m_rank, m_x, m_y;
    int local_start[2];
    int local_end[2];
    int *m_len;
    int *g_len;
    int *m_cart;
    int m_up, m_down, m_left, m_right;
    double *m_cellValue;
    double *m_cellValueNew;
    double *m_fb;
    double *m_exact;

    MPI_Datatype x_edge_type;
    MPI_Datatype y_edge_type;
    double dx, dy;
};

#endif
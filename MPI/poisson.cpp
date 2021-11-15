#include "poisson.hpp"
#include <math.h>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <fstream>

double u_exact(double x, double y)
{
    double pi = 3.141592653589793;
    double value = sin(pi * x * y);
    return value;
}
double uxxyy_exact(double x, double y)
{
    double pi = 3.141592653589793;
    double value;
    value = -pi * pi * (x * x + y * y) * sin(pi * x * y);
    return value;
}

void Poisson2d::init(int rank, MPI_Comm &cart_comm)
{
    double phi_x;
    double phi_y;
    int g_x, g_y;

    for (int j = 1; j <= m_len[1]; j++)
    {
        for (int i = 1; i <= m_len[0]; i++)
        {
            g_x = m_x * m_len[0] + i - 1;
            g_y = m_y * m_len[1] + j - 1;
            phi_x = g_x * dx;
            phi_y = g_y * dy;

            if (g_x == 0 || g_x == g_len[0] - 1 || g_y == 0 || g_y == g_len[1] - 1)
            {
                m_cellValue[l2i(i, j)] = u_exact(phi_x, phi_y);
                m_fb[l2i(i, j)] = u_exact(phi_x, phi_y);
            }
            else
            {
                m_fb[l2i(i, j)] = -uxxyy_exact(phi_x, phi_y);
            }
            m_exact[l2i(i, j)] = u_exact(phi_x, phi_y);
        }
    }

    MPI_Cart_shift(cart_comm, 0, 1, &m_left, &m_right);

    MPI_Cart_shift(cart_comm, 1, 1, &m_down, &m_up);

    MPI_Type_vector(m_len[0], 1, 1, MPI_DOUBLE, &x_edge_type);
    MPI_Type_commit(&x_edge_type);

    MPI_Type_vector(m_len[1], 1, (m_len[0] + 2), MPI_DOUBLE, &y_edge_type);
    MPI_Type_commit(&y_edge_type);
}

void Poisson2d::exchange()
{
    MPI_Status status;

    int tag = 1;
    if (m_x != 0)
    {
        MPI_Send(&m_cellValue[l2i(1, 1)], 1, y_edge_type, m_left, tag, MPI_COMM_WORLD);
    }

    if (m_x != m_cart[0] - 1)
    {
        MPI_Recv(&m_cellValue[l2i(m_len[0] + 1, 1)], 1, y_edge_type, m_right, tag, MPI_COMM_WORLD, &status);
    }

    tag = 2;
    if (m_x != m_cart[0] - 1)
    {
        MPI_Send(&m_cellValue[l2i(m_len[0], 1)], 1, y_edge_type, m_right, tag, MPI_COMM_WORLD);
    }

    if (m_x != 0)
    {
        MPI_Recv(&m_cellValue[l2i(0, 1)], 1, y_edge_type, m_left, tag, MPI_COMM_WORLD, &status);
    }

    tag = 3;
    if (m_y != m_cart[1] - 1)
    {
        MPI_Send(&m_cellValue[l2i(1, m_len[1])], 1, x_edge_type, m_up, tag, MPI_COMM_WORLD);
    }
    if (m_y != 0)
    {
        MPI_Recv(&m_cellValue[l2i(1, 0)], 1, x_edge_type, m_down, tag, MPI_COMM_WORLD, &status);
    }

    tag = 4;
    if (m_y != 0)
    {
        MPI_Send(&m_cellValue[l2i(1, 1)], 1, x_edge_type, m_down, tag, MPI_COMM_WORLD);
    }
    if (m_y != m_cart[1] - 1)
    {
        MPI_Recv(&m_cellValue[l2i(1, m_len[1] + 1)], 1, x_edge_type, m_up, tag, MPI_COMM_WORLD, &status);
    }
}

void Poisson2d::update()
{
    int g_x, g_y;
    double phi_x, phi_y;

    for (int j = 1; j <= m_len[1]; j++)
    {
        for (int i = 1; i <= m_len[0]; i++)
        {
            g_x = m_x * m_len[0] + i - 1;
            g_y = m_y * m_len[1] + j - 1;
            phi_x = g_x * dx;
            phi_y = g_y * dy;

            if (g_x == 0 || g_x == g_len[0] - 1 || g_y == 0 || g_y == g_len[1] - 1)
            {
                m_cellValueNew[l2i(i, j)] = u_exact(phi_x, phi_y);
            }
            else
            {
                m_cellValueNew[l2i(i, j)] =
                    0.25 * (m_cellValue[l2i(i - 1, j)] + m_cellValue[l2i(i + 1, j)] +
                            m_cellValue[l2i(i, j - 1)] + m_cellValue[l2i(i, j + 1)] +
                            m_fb[l2i(i, j)] * dx * dy);
            }
        }
    }

    for (int j = 1; j <= m_len[1]; j++)
    {
        for (int i = 1; i <= m_len[0]; i++)
        {

            m_cellValue[l2i(i, j)] = m_cellValueNew[l2i(i, j)];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    return;
}
void Poisson2d::error(int rank, int procNum)
{
    double error = 0;
    int g_x, g_y;
    for (int j = 1; j <= m_len[1]; j++)
    {
        for (int i = 1; i <= m_len[0]; i++)
        {
            error += pow((m_cellValue[l2i(i, j)] - m_exact[l2i(i, j)]), 2);
        }
    }
    error = sqrt(error / (double)(m_len[1] * m_len[0]));

    double totalerror = 0;
    MPI_Reduce(&error, &totalerror, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        std::cout << "avg error for is " << sqrt(totalerror / (double)(360 * 360))
                  << std::endl;
    }
}

#include "poisson.hpp"
#include <math.h>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <fstream>

typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>> mdrange_policy;

KOKKOS_FUNCTION
double u_exact(double x, double y)
{
    double pi = 3.141592653589793;
    double value = sin(pi * x * y);
    return value;
};
KOKKOS_FUNCTION
double uxxyy_exact(double x, double y)
{
    double pi = 3.141592653589793;
    double value;
    value = -pi * pi * (x * x + y * y) * sin(pi * x * y);
    return value;
};

void Poisson2d::init()
{

    x_lo = lx * comm.px;
    x_hi = lx * (comm.px + 1);
    y_lo = ly * comm.py;
    y_hi = ly * (comm.py + 1);

    // printf("Me:%i x_lo: %i x_hi: %i y_lo: %i y_hi: %i\n", comm.me, x_lo, x_hi, y_lo, y_hi);
    m_cellValue = Kokkos::View<double **, Kokkos::CudaUVMSpace>("m_cellValue", lx + 2, ly + 2);
    m_cellValueNew = Kokkos::View<double **, Kokkos::CudaUVMSpace>("m_cellValueNew", lx + 2, ly + 2);
    m_fb = Kokkos::View<double **, Kokkos::CudaUVMSpace>("m_fb", lx + 2, ly + 2);
    m_exact = Kokkos::View<double **, Kokkos::CudaUVMSpace>("m_exact", lx + 2, ly + 2);

    int X = m_cellValue.extent(0);
    int Y = m_cellValue.extent(1);

    Kokkos::parallel_for(
        "init", mdrange_policy({0, 0}, {X, Y}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            double phi_x = (lx * comm.px + i - 1) * dx;
            double phi_y = (ly * comm.py + j - 1) * dy;

            if ((lx * comm.px + i == 1) || (lx * comm.px + i == glx) || (ly * comm.py + j == 1) || (ly * comm.py + j == gly))
            {
                m_fb(i, j) = u_exact(phi_x, phi_y);
                m_cellValue(i, j) = u_exact(phi_x, phi_y);
            }
            else
            {
                m_fb(i, j) = -uxxyy_exact(phi_x, phi_y);
            }
            m_exact(i, j) = u_exact(phi_x, phi_y);
        });
}

void Poisson2d::setup_subdomain()
{

    if (x_lo != 0)
        m_left = buffer_t("m_left", y_hi - y_lo + 2);

    if (x_hi != glx)
        m_right = buffer_t("m_right", y_hi - y_lo + 2);

    if (y_lo != 0)
        m_down = buffer_t("m_down", x_hi - x_lo + 2);

    if (y_hi != gly)
        m_up = buffer_t("m_up", x_hi - x_lo + 2);

    if (x_lo != 0)
        m_leftout = buffer_t("m_leftout", ly + 2);

    if (x_hi != glx)
        m_rightout = buffer_t("m_rightout", y_hi - y_lo + 2);

    if (y_lo != 0)
        m_downout = buffer_t("m_downout", x_hi - x_lo + 2);

    if (y_hi != gly)
        m_upout = buffer_t("m_upout", x_hi - x_lo + 2);
}
void Poisson2d::pack()
{
    if (x_lo != 0)
        Kokkos::deep_copy(m_leftout, Kokkos::subview(m_cellValue, 1, Kokkos::ALL));

    if (x_hi != glx)
        Kokkos::deep_copy(m_rightout, Kokkos::subview(m_cellValue, lx, Kokkos::ALL));

    if (y_lo != 0)
        Kokkos::deep_copy(m_downout, Kokkos::subview(m_cellValue, Kokkos::ALL, 1));

    if (y_hi != gly)
        Kokkos::deep_copy(m_upout, Kokkos::subview(m_cellValue, Kokkos::ALL, ly));

    if (x_lo != 0)
        Kokkos::deep_copy(m_left, Kokkos::subview(m_cellValue, 0, Kokkos::ALL));

    if (x_hi != glx)
        Kokkos::deep_copy(m_right, Kokkos::subview(m_cellValue, lx + 1, Kokkos::ALL));

    if (y_lo != 0)
        Kokkos::deep_copy(m_down, Kokkos::subview(m_cellValue, Kokkos::ALL, 0));

    if (y_hi != gly)
        Kokkos::deep_copy(m_up, Kokkos::subview(m_cellValue, Kokkos::ALL, ly + 1));
}

void Poisson2d::exchange()
{

    int mar = 1;

    if (x_lo != 0)
    {
        MPI_Send(m_leftout.data(), m_leftout.size(), MPI_DOUBLE, comm.left, mar, comm.comm);
    }

    if (x_hi != glx)
    {
        MPI_Recv(m_right.data(), m_right.size(), MPI_DOUBLE, comm.right, mar, comm.comm, MPI_STATUSES_IGNORE);
    }

    mar = 2;
    if (x_hi != glx)
    {
        MPI_Send(m_rightout.data(), m_rightout.size(), MPI_DOUBLE, comm.right, mar, comm.comm);
    }
    if (x_lo != 0)
    {
        MPI_Recv(m_left.data(), m_left.size(), MPI_DOUBLE, comm.left, mar, comm.comm, MPI_STATUSES_IGNORE);
    }

    mar = 3;

    if (y_lo != 0)
    {
        MPI_Send(m_downout.data(), m_downout.size(), MPI_DOUBLE, comm.down, mar, comm.comm);
    }

    if (y_hi != gly)
    {
        MPI_Recv(m_up.data(), m_up.size(), MPI_DOUBLE, comm.up, mar, comm.comm, MPI_STATUSES_IGNORE);
    }

    mar = 4;
    if (y_hi != gly)
    {
        MPI_Send(m_upout.data(), m_upout.size(), MPI_DOUBLE, comm.up, mar, comm.comm);
    }
    if (y_lo != 0)
    {
        MPI_Recv(m_down.data(), m_down.size(), MPI_DOUBLE, comm.down, mar, comm.comm, MPI_STATUSES_IGNORE);
    }
}

void Poisson2d::unpack()
{

    if (x_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(m_cellValue, 0, Kokkos::ALL), m_left);

    if (x_hi != glx)
        Kokkos::deep_copy(Kokkos::subview(m_cellValue, lx + 1, Kokkos::ALL), m_right);

    if (y_lo != 0)
        Kokkos::deep_copy(Kokkos::subview(m_cellValue, Kokkos::ALL, 0), m_down);

    if (y_hi != gly)
        Kokkos::deep_copy(Kokkos::subview(m_cellValue, Kokkos::ALL, ly + 1), m_up);
}

void Poisson2d::update()
{

    MPI_Barrier(MPI_COMM_WORLD);
    int X = m_cellValue.extent(0);
    int Y = m_cellValue.extent(1);
    Kokkos::parallel_for(
        "init", mdrange_policy({1, 1}, {X - 1, Y - 1}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            double phi_x = (lx * comm.px + i - 1) * dx;
            double phi_y = (ly * comm.py + j - 1) * dy;

            if ((lx * comm.px + i == 1) || (lx * comm.px + i == glx) || (ly * comm.py + j == 1) || (ly * comm.py + j == gly))
            {
                m_cellValueNew(i, j) = u_exact(phi_x, phi_y);
            }
            else
            {
                m_cellValueNew(i, j) =
                    0.25 * (m_cellValue(i - 1, j) + m_cellValue(i, j + 1) +
                            m_cellValue(i, j - 1) + m_cellValue(i + 1, j) +
                            m_fb(i, j) * dx * dy);
            }
        });

    Kokkos::deep_copy(m_cellValue, m_cellValueNew);
    MPI_Barrier(MPI_COMM_WORLD);
    return;
}
void Poisson2d::error()
{
    double result;
    int X = m_cellValue.extent(0);
    int Y = m_cellValue.extent(1);
    Kokkos::parallel_reduce(
        "error", mdrange_policy({1, 1}, {X - 1, Y - 1}), KOKKOS_CLASS_LAMBDA(int i, int j, double &error) {
            error += pow((m_cellValue(i, j) - m_exact(i, j)), 2);
        },
        result);

    double totalerror = 0;
    MPI_Reduce(&result, &totalerror, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (comm.me == 0)
    {
        std::cout << "avg error for is " << sqrt(totalerror / (glx * gly))
                  << std::endl;
    }
}

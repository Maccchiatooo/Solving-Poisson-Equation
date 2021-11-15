#include "ko.hpp"
#include <math.h>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include "ko.hpp"
#include <Kokkos_Core.hpp>

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
    Kokkos::parallel_for(
        "init", mdrange_policy({0, 0}, {mx + 1, my + 1}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            m_cellValue(i, j) = 0;
            m_cellValueNew(i, j) = 0;
            m_fb(i, j) = 0;
            m_exact(i, j) = 0;
        });
    Kokkos::parallel_for(
        "cal", mdrange_policy({0, 0}, {mx + 1, my + 1}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            double phi_x = i * mdx;
            double phi_y = j * mdy;
            if (i == 0 || i == mx || j == 0 || j == my)
            {
                m_fb(i, j) = u_exact(phi_x, phi_y);
                m_cellValue(i, j) = u_exact(phi_x, phi_y);
            }
            else
            {
                m_fb(i, j) = -uxxyy_exact(phi_x, phi_y);
            }
            m_exact(i, j) =  u_exact(phi_x, phi_y); });
};

void Poisson2d::update()
{
    Kokkos::parallel_for(
        "init", mdrange_policy({0, 0}, {mx + 1, my + 1}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
            double phi_x = i * mdx;
            double phi_y = j * mdy;

            if (i == 0 || i == mx || j == 0 || j == my)
            {
                m_cellValueNew(i, j) = u_exact(phi_x, phi_y);
            }
            else
            {
                m_cellValueNew(i, j) =
                    0.25 * (m_cellValue(i - 1, j) + m_cellValue(i + 1, j) +
                            m_cellValue(i, j - 1) + m_cellValue(i, j + 1) +
                            m_fb(i, j) * mdx * mdy);
            }
        });
    // m_cellValue = m_cellValueNew;
    Kokkos::deep_copy(m_cellValue, m_cellValueNew);
};

void Poisson2d::error()
{
    double result = 0;
    Kokkos::parallel_reduce(
        "error", mdrange_policy({0, 0}, {mx + 1, my + 1}), KOKKOS_CLASS_LAMBDA(int i, int j, double &error) {
            error += pow((m_cellValue(i, j) - m_exact(i, j)), 2);
        },
        result);

    result = sqrt(result / (double)((my + 1) * (mx + 1)));

    printf("avg error for is %g\n", result);
};

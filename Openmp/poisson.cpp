#include "poisson.hpp"
#include <math.h>
#include <cstring>
#include <iostream>
#include <omp.h>

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

void Poisson2d::init()
{
    double phi_x;
    double phi_y;
    int g_x, g_y;
#pragma omp parallel for private(phi_x, phi_y)
    for (int i = 0; i <= mx; i++)
    {
        for (int j = 0; j <= my; j++)
        {

            phi_x = i * mdx;
            phi_y = j * mdy;

            if (i == 0 || i == mx || j == 0 || j == my)
            {
                m_fb[l2i(i, j)] = u_exact(phi_x, phi_y);
                m_cellValue[l2i(i, j)] = u_exact(phi_x, phi_y);
            }
            else
            {
                m_cellValue[l2i(i, j)] = 0.0;
                m_fb[l2i(i, j)] = -uxxyy_exact(phi_x, phi_y);
            }
            m_exact[l2i(i, j)] = u_exact(phi_x, phi_y);
        }
    }
}

void Poisson2d::update()
{
    double phi_x, phi_y;
#pragma omp parallel for private(phi_x, phi_y)
    for (int j = 0; j <= my; j++)
    {
        for (int i = 0; i <= mx; i++)
        {

            phi_x = i * mdx;
            phi_y = j * mdy;

            if (i == 0 || i == mx || j == 0 || j == my)
            {
                m_cellValueNew[l2i(i, j)] = u_exact(phi_x, phi_y);
            }
            else
            {
                m_cellValueNew[l2i(i, j)] =
                    0.25 * (m_cellValue[l2i(i - 1, j)] + m_cellValue[l2i(i + 1, j)] +
                            m_cellValue[l2i(i, j - 1)] + m_cellValue[l2i(i, j + 1)] +
                            m_fb[l2i(i, j)] * mdx * mdy);
            }
        }
    }

    m_cellValue = m_cellValueNew;
    /*#pragma omp parallel for private(phi_x, phi_y)
        for (int j = 0; j <= my; j++)
        {
            for (int i = 0; i <= mx; i++)
            {

                m_cellValue[l2i(i, j)] = m_cellValueNew[l2i(i, j)];
            }
        }*/

    return;
}

void Poisson2d::error()
{
    double error = 0;
#pragma omp parallel for reduction(+ \
                                   : error)
    for (int j = 0; j <= my; j++)
    {
        for (int i = 0; i <= mx; i++)
        {
            error += pow((m_cellValue[l2i(i, j)] - m_exact[l2i(i, j)]), 2);
        }
    }

    error = sqrt(error / (double)((my + 1) * (mx + 1)));

    std::cout << "avg error for is " << error << std::endl;
}
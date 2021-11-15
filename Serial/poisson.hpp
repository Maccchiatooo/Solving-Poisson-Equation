#ifndef _poisson_H_
#define _poisson_H_
#include <fstream>
#include <iostream>
#include <vector>
class Poisson2d
{

public:
    Poisson2d(int nx, int ny, double dx, double dy)
        : mx(nx),
          my(ny),
          mdx(dx),
          mdy(dy)
    {

        m_cellValue = (double *)malloc(sizeof(double) * (mx + 1) * (my + 1));
        m_cellValueNew = (double *)malloc(sizeof(double) * (mx + 1) * (my + 1));
        m_fb = (double *)malloc(sizeof(double) * (mx + 1) * (my + 1));
        m_exact = (double *)malloc(sizeof(double) * (mx + 1) * (my + 1));

        for (int j = 0; j <= my; j++)
        {
            for (int i = 0; i <= mx; i++)
            {
                m_cellValue[l2i(i, j)] = 0;
                m_cellValueNew[l2i(i, j)] = 0;
                m_fb[l2i(i, j)] = 0;
                m_exact[l2i(i, j)] = 0;
            }
        }
    }

    inline int l2i(int x, int y) const
    {

        return x + y * (this->mx + 1);
    }

    void init();

    void update();
    void error();

    int mx, my;
    double mdx, mdy;
    double *m_cellValue;
    double *m_cellValueNew;
    double *m_fb;
    double *m_exact;
};

#endif
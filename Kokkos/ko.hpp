#ifndef _ko_H_
#define _ko_H_
#include <Kokkos_Core.hpp>

class Poisson2d
{
    typedef Kokkos::View<double **> ViewMatrixType;
    typedef Kokkos::RangePolicy<> range_policy;
    typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>> mdrange_policy;

public:
    Poisson2d(const int nx, const int ny, const double dx, const double dy)
        : mx(nx),
          my(ny),
          mdx(dx),
          mdy(dy){

          };

    void init();
    void update();

    void output();

    void error();
    int mx, my;
    double mdx, mdy;

    Kokkos::View<double **> m_cellValue = Kokkos::View<double **>("m_c", mx + 1, my + 1);
    Kokkos::View<double **> m_cellValueNew = Kokkos::View<double **>("m_n", mx + 1, my + 1);
    Kokkos::View<double **> m_fb = Kokkos::View<double **>("m_f", mx + 1, my + 1);
    Kokkos::View<double **> m_exact = Kokkos::View<double **>("m_e", mx + 1, my + 1);
};
#endif
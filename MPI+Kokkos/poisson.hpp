#ifndef _poisson_H_
#define _poisson_H_
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <fstream>
#include <iostream>

struct CommHelper
{

    MPI_Comm comm;
    int rx, ry;
    int me;
    int px, py;
    int up, down, left, right;

    CommHelper(MPI_Comm comm_)
    {
        comm = comm_;
        int nranks;
        MPI_Comm_size(comm, &nranks);
        MPI_Comm_rank(comm, &me);

        rx = std::pow(1.0 * nranks, 1.0 / 2.0);
        while (nranks % rx != 0)
            rx++;

        ry = nranks / rx;

        px = me % rx;
        py = (me / rx) % ry;
        left = px == 0 ? -1 : me - 1;
        right = px == rx - 1 ? -1 : me + 1;
        down = py == 0 ? -1 : me - rx;
        up = py == ry - 1 ? -1 : me + rx;

        printf("Me:%i MyNeibors: %i %i %i %i\n", me, left, right, up, down);
    }
    template <class ViewType>
    void isend_irecv(int partner, ViewType send_buffer, ViewType recv_buffer, MPI_Request *request_send, MPI_Request *request_recv)
    {
        MPI_Irecv(recv_buffer.data(), recv_buffer.size(), MPI_DOUBLE, partner, 1, comm, request_recv);
        MPI_Isend(send_buffer.data(), send_buffer.size(), MPI_DOUBLE, partner, 1, comm, request_send);
    }
};
struct Poisson2d
{

    CommHelper comm;
    MPI_Request mpi_requests_recv[4];
    MPI_Request mpi_requests_send[4];
    int mpi_active_requests;

    int glx, gly;
    double dx, dy;
    int lx, ly;
    int x, y, x_lo, x_hi, y_lo, y_hi;

    using buffer_t = Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::CudaSpace>;
    buffer_t m_left, m_right, m_down, m_up;
    buffer_t m_leftout, m_rightout, m_downout, m_upout;
    Kokkos::View<double **> m_cellValue, m_cellValueNew, m_fb, m_exact;

    Poisson2d(MPI_Comm comm_) : comm(comm_)
    {

        mpi_active_requests = 0;
        glx = 360;
        gly = 360;
        lx = glx / comm.rx;
        ly = gly / comm.ry;
        dx = 1.0 / (glx - 1);
        dy = 1.0 / (gly - 1);
        m_cellValue = Kokkos::View<double **>();
        m_cellValueNew = Kokkos::View<double **>();
        m_fb = Kokkos::View<double **>();
        m_exact = Kokkos::View<double **>();
    }
    void init();
    void setup_subdomain();
    void pack();
    void exchange();
    void update();
    // void output();
    void error();
};

#endif
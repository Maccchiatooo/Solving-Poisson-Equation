#include <math.h>
#include <iostream>
#include <ctime>
#include <CL/sycl.hpp>

using namespace sycl;

#define dim 2
#define nx 359
#define ny 359
#define iter 10000

inline int l2i(int x, int y)
{

    return x + y * (nx + 1);
}

int main(int argc, char *argv[])
{

    double dx = 1.0 / nx;
    double dy = 1.0 / ny;
    double pi = 3.141592653589793;
    const int size = (nx + 1) * (ny + 1);
    clock_t start, end;

    // create queue on defult device
    queue Q{};
    std::cout << "Selected device:" << Q.get_device().get_info<info::device::name>() << "\n";

    std::array<double, size> data;
    for (int id = 0; id < size; id++)
    {
        data[id] = 0;
    }

    buffer<double> m_cellValue{data};
    buffer<double> m_cellValueNew{data};
    buffer<double> m_fb{data};
    buffer<double> m_exact{data};
    buffer<double> u_exact{data};
    buffer<double> uxxyy_exact{data};
    start = clock();
    auto ini = Q.submit([&](handler &h)
                        {
        accessor m_c{m_cellValue, h, write_only};
        accessor m_f{m_fb, h, write_only};
        accessor m_e{m_exact, h, write_only};


        h.parallel_for(range{nx+1, ny+1}, [=](id<2> idx)
        {
            double phi_x = idx[0] * dx;
            double phi_y = idx[1] * dy;
            double u_e = sycl::sin(pi*phi_x*phi_y);
            double u_xe=-pi * pi * (pow(phi_x,2)+ pow(phi_y,2)) * sycl::sin(pi * phi_x*phi_y);
            int id = idx[0] + idx[1] * (nx + 1);

            if (idx[0] == 0 || idx[0] == nx || idx[1] == 0 || idx[1] == ny)
            {
                m_f[id] = u_e;
                m_c[id] = u_e;

            }else{
                m_f[id] = -u_xe;
            }
            m_e[id] = u_e;

            }); });

    for (int i = 1; i <= iter; i++)
    {

        if (i % 1000 == 0)
        {
            auto m_c = m_cellValue.get_access<access::mode::read>();
            // auto m_e = m_exact.get_access<access::mode::read>();

            double error = 0;

            for (int idx = 0; idx < nx +1; idx++)
            {
                for (int idy = 0; idy < ny +1; idy++)
                {
                    double phi_x = idx * dx;
                    double phi_y = idy * dy;
                    int id = idx + idy * (nx + 1);

                    error += pow((m_c[id] - sin(pi * phi_x * phi_y)), 2);
                }
            }
            error = sqrt(error / (double)size);
            std::cout << "avg error for is " << error << std::endl;
        }

        auto upd1 = Q.submit([&](handler &h)
                             {
            accessor m_cn{m_cellValueNew, h, write_only};
            accessor m_c{m_cellValue, h, read_only};
            accessor m_f{m_fb, h, read_only};

            h.parallel_for(range{nx + 1, ny + 1}, [=](id<2> idx)
                           {

        double phi_x = idx[0] * dx;
        double phi_y = idx[1] * dy;
        double u_e = sycl::sin(pi*phi_x*phi_y);
        double u_xe=-pi * pi * (pow(phi_x,2)+ pow(phi_y,2)) * sycl::sin(pi * phi_x*phi_y);
        int id = idx[0] + idx[1] * (nx + 1);

        if (idx[0] == 0 || idx[0] == nx || idx[1] == 0 || idx[1] == ny)
        {
            m_cn[id] = u_e;
        }else{
            m_cn[id] = 0.25 * (m_c[idx[0] - 1 + idx[1] * (nx + 1)] + m_c[idx[0] + 1 + idx[1] * (nx + 1)] +
                               m_c[idx[0] + (idx[1] - 1) * (nx + 1)] + m_c[idx[0] + (idx[1] + 1) * (nx + 1)] +
                               m_f[id] * dx * dy);

        } }); });

        auto upd2 = Q.submit([&](handler &h)
                             {
            accessor m_cn{m_cellValueNew, h, read_only};
            accessor m_c{m_cellValue, h, write_only};
            h.depends_on(upd1);
            h.parallel_for(range{nx + 1, ny + 1}, [=](id<2> idx)
                           {
                               int id = idx[0] + idx[1] * (nx + 1);
                               m_c[id] = m_cn[id]; }); });
        upd2.wait();
        end = clock();
        if (i % 1000 == 0)
            printf("time = %f\n", (double)(end - start) / CLOCKS_PER_SEC);
    }
}

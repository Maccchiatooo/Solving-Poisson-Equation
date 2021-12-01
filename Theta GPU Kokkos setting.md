# Installing Kokkos on login node
```
cd
mkdir kokkos
cd kokkos
git clone https://github.com/kokkos/kokkos.git
git clone https://github.com/kokkos/kokkos-tutorials.git
```
# Log into a ThetaGPU node
```
ssh thetagpusn1
qsub -I --attrs pubnet=true -A IMEXLBM -n 1 -q single-gpu -t 60
```
# Building Kokkos on the ThetaGPU node
```
cd
cd kokkos
mkdir build
cd build
module load cmake
cmake ../kokkos \
    -D CMAKE_CXX_FLAGS=-fopenmp \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX="${PWD}"/install \
    -D Kokkos_ENABLE_CUDA=On \
    -D Kokkos_ENABLE_CUDA_LAMBDA=On \
    -D Kokkos_ENABLE_SERIAL=On \
    -D Kokkos_ENABLE_OPENMP=On \
    -D CMAKE_CXX_STANDARD=17
    
make -j install
export CMAKE_PREFIX_PATH="${PWD}"/install:"${CMAKE_PREFIX_PATH}"
```
# Example build of a kokkos exercise on the ThetaGPU node
```
cd
cd kokkos
cd kokkos-tutorials/Exercises/04/Solution
mkdir build
cd build
cmake ..
make
```

# Solving-Poisson-Equation

2-D Poisson equation problem is solved by central finite difference
qsub -I -n 4 -t 60 -q default --attrs filesystems=home,grand -A IMEXLBM
five directories

# 1. Serial solver
  compile command:
  ```
  g++ poisson.cpp main.cpp -o run
  ```
  run command:
  ```
  ./run
  ```
  
# 2. OpenMP solver
  compile command:
  ```
  g++ poisson.cpp main.cpp -o run -fopenmp 
  ```
  run command: 
  ```
  ./run
  ```
  
# 3. MPI solver
  compile command: 
  ```
  mpicxx poisson.cpp main.cpp -o run 
  ```
  run command: 
  ```
  mpirun -n 12 ./run
  ```
  
# 4. Kokkos solver 
  Make sure the system supports C++17.\
  after config the kokkos on ThetaGPU, 
  make in the local directory: 
  ```
  make -j KOKKOS_DEVICES=OpenMP
  ```
  OR: 
  ```
  make -j KOKKOS_DEVICES=Cuda 
  ```
  to choose OpenMP or Cuda. \
  run command: 
  ```
  ./run.host
  ```
  OR:
  ```
  ./run.cuda
   ```
# 5. MPI+Kokkos solver 
  ```
  export OMPI_CXX=/grand/IMEXLBM/czhao/Kokkos/kokkos/bin/nvcc_wrapper
  ```
  ```
  export OMP_PROC_BIND=spread
  ```
  ```
  export OMP_PLACES=threads
  ```
  make in the local directory: 
  ```
  make -j
  
  module load PrgEnv-gnu/8.3.3 
  cudatoolkit-standalone/11.6.2
  make KOKKOS_CXX_STANDARD=c++17
  ```
  run command: if you want to use n cpu + n gpu
  ```
mpirun -hostfile $COBALT_NODEFILE -n num -npernode 8 LBM.exe
  ```
  


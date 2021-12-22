# Solving-Poisson-Equation

2-D Poisson equation problem is solved by central finite difference

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
  mpiicxx poisson.cpp main.cpp -o run 
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
# 4. MPI+Kokkos solver 
  ```
  export OMPI_CXX=~/Kokkos/kokkos/bin/nvcc_wrapper
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
  ```
  run command: 
  ```
  mpirun -np n LBM.exe --kokkos_num_devices=n
  ```
  


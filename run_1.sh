#!/bin/bash

echo "==== Baseline ===="
./outputfile baseline

echo "==== OpenMP ===="
./outputfile openmp

echo "==== Block Parallel ===="
./outputfile block

echo "==== MPI (4 processes) ===="
mpirun -np 4 ./outputfile mpi

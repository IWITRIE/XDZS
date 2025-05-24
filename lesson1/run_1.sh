#!/bin/bash
mpic++ -fopenmp -o outputfile lesson1_sourcefile.cpp
hipcc lesson1_sourcefile_dcu.cpp -o outputfile_dcu
echo "==== Baseline ===="
./outputfile baseline

echo "==== OpenMP ===="
./outputfile openmp

echo "==== Block Parallel ===="
./outputfile block

echo "==== MPI (4 processes) ===="
mpirun --allow-run-as-root -np 4 ./outputfile mpi

hipcc sourcefile_dcu.cpp -o outputfile_dcu

echo "==== DCU Performance Analysis ===="

echo "1. Build with debug symbols for hipgdb"
hipcc -g lesson1_sourcefile_dcu.cpp -o outputfile_dcu_debug

echo "2. Run rocm-smi to check GPU stats before running"
rocm-smi

echo "3. Run with hipprof for kernel profiling"
hipprof ./outputfile_dcu > hipprof_results.txt
echo "Profiling results saved to hipprof_results.txt"

echo "4. Collecting memory usage with rocm-smi during execution"
rocm-smi --showmeminfo vram > mem_usage_before.txt
./outputfile_dcu
rocm-smi --showmeminfo vram > mem_usage_after.txt
echo "Memory usage before and after execution saved"

echo "5. Debug example with hipgdb (commented out)"
# hipgdb --args ./outputfile_dcu_debug

echo "6. Generate timeline view with Chromium tracing"
HIP_TRACE_API=1 ./outputfile_dcu
echo "Trace file generated for Chrome tracing"

echo "==== Performance Analysis Complete ===="
# 🚀 高性能矩阵乘法并行计算项目

本项目实现了多种高性能矩阵乘法算法，支持单机多线程（OpenMP）、多进程（MPI）、分块优化、以及基于 DCU（HIP）的 GPU 加速。适合并行计算、异构计算、性能优化等课程实验与工程实践。

## ✨ 功能特性 Features

- **Baseline**：标准串行矩阵乘法
- **OpenMP**：多线程并行加速
- **Block Parallel**：分块并行与缓存友好优化
- **MPI**：多进程分布式并行
- **HIP (DCU)**：GPU 加速矩阵乘法

## 🛠️ 编译方法 Build

### CPU/OpenMP/MPI

```bash
mpic++ -fopenmp -o outputfile lesson1_sourcefile.cpp
```

### HIP (DCU)

```bash
hipcc lesson1_sourcefile_dcu.cpp -o outputfile_dcu
```

## 🚦 运行方法 Run

### Baseline / OpenMP / Block

```bash
./outputfile baseline
./outputfile openmp
./outputfile block
```

### MPI 并行

```bash
mpirun -np 4 ./outputfile mpi
```

### HIP (DCU) GPU 加速

```bash
./outputfile_dcu
```

### 一键评测脚本

```bash
bash run_1.sh
```

## 📊 性能评测 Performance

所有模式均内置性能计时，运行后会输出每种算法的执行时间（单位：秒），便于横向对比。

## 🧪 验证正确性

每种并行/加速实现均自动与 baseline 结果对比，输出 `Valid: 1` 表示结果正确。

## 📁 目录结构 Structure

```
lesson1_sourcefile.cpp      # CPU/OpenMP/MPI/Block 主体代码
lesson1_sourcefile_dcu.cpp  # HIP (DCU) GPU 加速代码
run_1.sh                    # 一键评测脚本
README.md                   # 项目说明
```

## 💡 参考/致谢

- OpenMP/MPI/HIP 官方文档
- 各类高性能计算教材

---

# 🚀 High Performance Matrix Multiplication Project

This project implements various high-performance matrix multiplication algorithms, supporting OpenMP, MPI, block optimization, and HIP (DCU) GPU acceleration. Ideal for parallel/heterogeneous computing courses and engineering practice.

See above for build, run, and performance instructions!

---

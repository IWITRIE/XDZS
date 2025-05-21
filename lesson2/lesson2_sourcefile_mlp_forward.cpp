#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

// 编译文件
// hipcc sourcefile_mlp_forward.cpp -o mlp_forward
// 执行文件
// ./mlp_forward 或者 hipprof ./mlp_forward


#define BATCH 1024
#define I 10
#define H 20
#define O 5

// 新增：添加偏置的kernel
__global__ void add_bias_kernel(double* output, const double* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int col = idx % cols;
        output[idx] += bias[col];
    }
}

// 修改：优化矩阵乘法kernel
__global__ void matmul_kernel(const double* A, const double* B, double* C, int M, int N, int K) {
    const int TILE_SIZE = 16;
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && t * TILE_SIZE + threadIdx.x < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;
            
        if (col < K && t * TILE_SIZE + threadIdx.y < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * K + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            
        __syncthreads();
    }
    
    if (row < M && col < K)
        C[row * K + col] = sum;
}

// 新增：朴素矩阵乘法kernel，用于性能对比
__global__ void matmul_naive_kernel(const double* A, const double* B, double* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < K) {
        double sum = 0.0;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

__global__ void relu_kernel(double* A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = fmax(0.0, A[idx]);
    }
    return;
}

void random_init(std::vector<double>& mat) {
    for (auto& val : mat) {
        val = static_cast<double>(rand()) / RAND_MAX * 2 - 1;
    }
    return;
}

int main() {
    std::vector<double> h_X(BATCH * I), h_W1(I * H), h_B1(H), h_W2(H * O), h_B2(O);
    std::vector<double> h_H(BATCH * H), h_Y(BATCH * O);

    random_init(h_X);
    random_init(h_W1);
    random_init(h_B1);
    random_init(h_W2);
    random_init(h_B2);

    // 以下均为主要修改部分
    double *d_X, *d_W1, *d_B1, *d_H, *d_W2, *d_B2, *d_Y;
    

    // 计时事件
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float milliseconds = 0;

    // 内存分配和数据传输
    hipMalloc(&d_X, BATCH * I * sizeof(double));
    hipMalloc(&d_W1, I * H * sizeof(double));
    hipMalloc(&d_B1, H * sizeof(double));
    hipMalloc(&d_H, BATCH * H * sizeof(double));
    hipMalloc(&d_W2, H * O * sizeof(double));
    hipMalloc(&d_B2, O * sizeof(double));
    hipMalloc(&d_Y, BATCH * O * sizeof(double));

    hipMemcpy(d_X, h_X.data(), BATCH * I * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_W1, h_W1.data(), I * H * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B1, h_B1.data(), H * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_W2, h_W2.data(), H * O * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B2, h_B2.data(), O * sizeof(double), hipMemcpyHostToDevice);

    hipEventRecord(start);

    // Hidden layer: H = X * W1 + B1
    dim3 blockDim(16, 16);
    dim3 gridDim1((H + blockDim.x - 1) / blockDim.x, 
                  (BATCH + blockDim.y - 1) / blockDim.y);
    hipLaunchKernelGGL(matmul_kernel, gridDim1, blockDim, 0, 0,
                       d_X, d_W1, d_H, BATCH, I, H);
    
    // Add bias
    dim3 gridDim_bias1((BATCH * H + 255) / 256);
    hipLaunchKernelGGL(add_bias_kernel, gridDim_bias1, 256, 0, 0,
                       d_H, d_B1, BATCH, H);

    // Apply ReLU
    hipLaunchKernelGGL(relu_kernel, gridDim_bias1, 256, 0, 0,
                       d_H, BATCH * H);

    // Output layer: Y = H * W2 + B2
    dim3 gridDim2((O + blockDim.x - 1) / blockDim.x,
                  (BATCH + blockDim.y - 1) / blockDim.y);
    hipLaunchKernelGGL(matmul_kernel, gridDim2, blockDim, 0, 0,
                       d_H, d_W2, d_Y, BATCH, H, O);

    // Add output bias
    dim3 gridDim_bias2((BATCH * O + 255) / 256);
    hipLaunchKernelGGL(add_bias_kernel, gridDim_bias2, 256, 0, 0,
                       d_Y, d_B2, BATCH, O);

    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&milliseconds, start, stop);

    // 复制结果回主机
    hipMemcpy(h_Y.data(), d_Y, BATCH * O * sizeof(double), hipMemcpyDeviceToHost);

    // 打印性能指标
    std::cout << "Forward Pass Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Throughput: " << (BATCH / (milliseconds / 1000.0)) << " samples/s" << std::endl;

    // 打印部分结果用于验证
    for (int i = 0; i < 5; ++i) {
        std::cout << "Output[" << i << "]: ";
        for (int j = 0; j < O; ++j)
            std::cout << h_Y[i * O + j] << " ";
        std::cout << std::endl;
    }

    // 新增：为朴素实现分配中间缓冲
    double *d_H_naive, *d_Y_naive;
    hipMalloc(&d_H_naive, BATCH * H * sizeof(double));
    hipMalloc(&d_Y_naive, BATCH * O * sizeof(double));

    // 拷贝输入和权重（同原逻辑）
    hipMemcpy(d_X, h_X.data(), BATCH * I * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_W1, h_W1.data(), I * H * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B1, h_B1.data(), H * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_W2, h_W2.data(), H * O * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B2, h_B2.data(), O * sizeof(double), hipMemcpyHostToDevice);

    float time_naive = 0, time_opt = 0;
    // 朴素实现测时
    hipEventRecord(start);
    // Hidden layer (naive)
    hipLaunchKernelGGL(matmul_naive_kernel, gridDim1, blockDim, 0, 0,
                       d_X, d_W1, d_H_naive, BATCH, I, H);
    hipLaunchKernelGGL(add_bias_kernel, gridDim_bias1, 256, 0, 0,
                       d_H_naive, d_B1, BATCH, H);
    hipLaunchKernelGGL(relu_kernel, gridDim_bias1, 256, 0, 0,
                       d_H_naive, BATCH * H);
    // Output layer (naive)
    hipLaunchKernelGGL(matmul_naive_kernel, gridDim2, blockDim, 0, 0,
                       d_H_naive, d_W2, d_Y_naive, BATCH, H, O);
    hipLaunchKernelGGL(add_bias_kernel, gridDim_bias2, 256, 0, 0,
                       d_Y_naive, d_B2, BATCH, O);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&time_naive, start, stop);

    // 优化实现测时（原有 tiled + shared memory）
    hipEventRecord(start);
    // Hidden layer (opt)
    hipLaunchKernelGGL(matmul_kernel, gridDim1, blockDim, 0, 0,
                       d_X, d_W1, d_H, BATCH, I, H);
    hipLaunchKernelGGL(add_bias_kernel, gridDim_bias1, 256, 0, 0,
                       d_H, d_B1, BATCH, H);
    hipLaunchKernelGGL(relu_kernel, gridDim_bias1, 256, 0, 0,
                       d_H, BATCH * H);
    // Output layer (opt)
    hipLaunchKernelGGL(matmul_kernel, gridDim2, blockDim, 0, 0,
                       d_H, d_W2, d_Y, BATCH, H, O);
    hipLaunchKernelGGL(add_bias_kernel, gridDim_bias2, 256, 0, 0,
                       d_Y, d_B2, BATCH, O);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&time_opt, start, stop);

    // 复制结果回主机（任选一种）
    hipMemcpy(h_Y.data(), d_Y.data(), BATCH * O * sizeof(double), hipMemcpyDeviceToHost);

    // 打印对比结果
    std::cout << "Naive Forward Time: " << time_naive << " ms, "
              << "Throughput: " << BATCH / (time_naive / 1000.0) << " samples/s\n";
    std::cout << "Optimized Forward Time: " << time_opt << " ms, "
              << "Throughput: " << BATCH / (time_opt / 1000.0) << " samples/s\n";

    hipFree(d_X);
    hipFree(d_W1);
    hipFree(d_B1);
    hipFree(d_H);
    hipFree(d_W2);
    hipFree(d_B2);
    hipFree(d_Y);
    hipFree(d_H_naive);
    hipFree(d_Y_naive);

    return 0;
}
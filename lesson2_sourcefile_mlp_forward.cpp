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

// 主要修改函数
__global__ void matmul_kernel(const double* A, const double* B, double* C, int M, int N, int K) {
    const int TILE_SIZE = 16;
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // 加载数据到共享内存
        if (row < M && t * TILE_SIZE + threadIdx.x < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;
            
        if (col < K && t * TILE_SIZE + threadIdx.y < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * K + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        
        __syncthreads();

        // 计算tile内的乘法累加
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            
        __syncthreads();
    }
    
    if (row < M && col < K)
        C[row * K + col] = sum;
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
    

    // 内存分配
    hipMalloc(&d_X, BATCH * I * sizeof(double));
    hipMalloc(&d_W1, I * H * sizeof(double));
    hipMalloc(&d_B1, H * sizeof(double));
    hipMalloc(&d_H, BATCH * H * sizeof(double));
    hipMalloc(&d_W2, H * O * sizeof(double));
    hipMalloc(&d_B2, O * sizeof(double));
    hipMalloc(&d_Y, BATCH * O * sizeof(double));

    // 数据传输到设备
    hipMemcpy(d_X, h_X.data(), BATCH * I * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_W1, h_W1.data(), I * H * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B1, h_B1.data(), H * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_W2, h_W2.data(), H * O * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_B2, h_B2.data(), O * sizeof(double), hipMemcpyHostToDevice);

    // Hidden layer: H = X * W1
    dim3 blockDim(16, 16);
    dim3 gridDim1((H + blockDim.x - 1) / blockDim.x, 
                 (BATCH + blockDim.y - 1) / blockDim.y);
    hipLaunchKernelGGL(matmul_kernel, gridDim1, blockDim, 0, 0,
                       d_X, d_W1, d_H, BATCH, I, H);

    // Add bias and apply ReLU
    dim3 gridDim2((BATCH * H + 255) / 256);
    hipLaunchKernelGGL(relu_kernel, gridDim2, 256, 0, 0,
                       d_H, BATCH * H);

    // Output layer: Y = H * W2
    dim3 gridDim3((O + blockDim.x - 1) / blockDim.x,
                 (BATCH + blockDim.y - 1) / blockDim.y);
    hipLaunchKernelGGL(matmul_kernel, gridDim3, blockDim, 0, 0,
                       d_H, d_W2, d_Y, BATCH, H, O);

    // 将结果复制回主机
    hipMemcpy(h_Y.data(), d_Y, BATCH * O * sizeof(double), hipMemcpyDeviceToHost);

    // 打印部分结果用于验证
    for (int i = 0; i < 5; ++i) {
        std::cout << "Output[" << i << "]: ";
        for (int j = 0; j < O; ++j)
            std::cout << h_Y[i * O + j] << " ";
        std::cout << std::endl;
    }

    hipFree(d_X);
    hipFree(d_W1);
    hipFree(d_B1);
    hipFree(d_H);
    hipFree(d_W2);
    hipFree(d_B2);
    hipFree(d_Y);

    return 0;
}
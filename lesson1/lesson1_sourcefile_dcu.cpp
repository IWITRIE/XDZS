#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>


// 编译
// hipcc sourcefile_dcu.cpp -o outputfile_dcu
// 执行
// ./outputfile_dcu

#define N 1024
#define M 2024
#define P 512

// 主要修改函数
__global__ void matmul_kernel(const double* A, const double* B, double* C, int n, int m, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < p) {
        double sum = 0.0;
        for (int k = 0; k < m; ++k) {
            sum += A[row * m + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

// 新增：shared memory 优化 kernel
__global__ void matmul_kernel_shared(const double* A, const double* B, double* C, int n, int m, int p) {
    // tile/block size
    const int TILE = 16;
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    double sum = 0.0;

    for (int t = 0; t < (m + TILE - 1) / TILE; ++t) {
        // load tile from A and B
        if (row < n && t * TILE + threadIdx.x < m)
            As[threadIdx.y][threadIdx.x] = A[row * m + t * TILE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;
        if (col < p && t * TILE + threadIdx.y < m)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * p + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < n && col < p)
        C[row * p + col] = sum;
}

void init_matrix(std::vector<double>& mat) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& x : mat)
        x = dist(gen);
    return;
}

void matmul_cpu(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k)
                sum += A[i * M + k] * B[k * P + j];
            C[i * P + j] = sum;
        }
    return;
}

bool validate(const std::vector<double>& ref, const std::vector<double>& test) {
    for (size_t i = 0; i < ref.size(); ++i)
        if (std::abs(ref[i] - test[i]) > 1e-6)
            return false;
    return true;
}

// 新增：shared memory 优化接口
void matmul_dcu_shared(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int n, int m, int p, float& elapsed_ms) {
    double *d_A, *d_B, *d_C;
    size_t size_A = n * m * sizeof(double);
    size_t size_B = m * p * sizeof(double);
    size_t size_C = n * p * sizeof(double);

    hipMalloc(&d_A, size_A);
    hipMalloc(&d_B, size_B);
    hipMalloc(&d_C, size_C);

    hipMemcpy(d_A, A.data(), size_A, hipMemcpyHostToDevice);
    hipMemcpy(d_B, B.data(), size_B, hipMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((p + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0);

    hipLaunchKernelGGL(matmul_kernel_shared, gridDim, blockDim, 0, 0, d_A, d_B, d_C, n, m, p);

    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsed_ms, start, stop);

    hipMemcpy(C.data(), d_C, size_C, hipMemcpyDeviceToHost);

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
}

int main() {
    std::vector<double> A(N * M), B(M * P), C(N * P), C_ref(N * P);
    init_matrix(A);
    init_matrix(B);

    // CPU baseline
    matmul_cpu(A, B, C_ref);

    // 原始 DCU kernel
    double *d_A, *d_B, *d_C;
    size_t size_A = N * M * sizeof(double);
    size_t size_B = M * P * sizeof(double);
    size_t size_C = N * P * sizeof(double);

    float t_copy_h2d = 0, t_copy_d2h = 0, t_kernel = 0, t_total = 0;
    hipEvent_t ev_start, ev_h2d, ev_kernel_start, ev_kernel_stop, ev_d2h, ev_end;
    hipEventCreate(&ev_start);
    hipEventCreate(&ev_h2d);
    hipEventCreate(&ev_kernel_start);
    hipEventCreate(&ev_kernel_stop);
    hipEventCreate(&ev_d2h);
    hipEventCreate(&ev_end);

    hipEventRecord(ev_start, 0);

    hipMalloc(&d_A, size_A);
    hipMalloc(&d_B, size_B);
    hipMalloc(&d_C, size_C);

    hipMemcpy(d_A, A.data(), size_A, hipMemcpyHostToDevice);
    hipMemcpy(d_B, B.data(), size_B, hipMemcpyHostToDevice);

    hipEventRecord(ev_h2d, 0);

    dim3 blockDim(16, 16);
    dim3 gridDim((P + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    hipEventRecord(ev_kernel_start, 0);
    hipLaunchKernelGGL(matmul_kernel, gridDim, blockDim, 0, 0, d_A, d_B, d_C, N, M, P);
    hipEventRecord(ev_kernel_stop, 0);
    hipEventSynchronize(ev_kernel_stop);

    hipMemcpy(C.data(), d_C, size_C, hipMemcpyDeviceToHost);
    hipEventRecord(ev_d2h, 0);
    hipEventSynchronize(ev_d2h);

    hipEventRecord(ev_end, 0);
    hipEventSynchronize(ev_end);

    hipEventElapsedTime(&t_copy_h2d, ev_start, ev_h2d);
    hipEventElapsedTime(&t_kernel, ev_kernel_start, ev_kernel_stop);
    hipEventElapsedTime(&t_copy_d2h, ev_kernel_stop, ev_d2h);
    hipEventElapsedTime(&t_total, ev_start, ev_d2h);

    if (validate(C_ref, C)) {
       std::cout << "[HIP] Valid: 1" << std::endl;
    } else {
       std::cout << "[HIP] Valid: 0" << std::endl;
    }
    std::cout << "[HIP] H2D Copy Time: " << t_copy_h2d / 1000.0 << " s" << std::endl;
    std::cout << "[HIP] Kernel Time: " << t_kernel / 1000.0 << " s" << std::endl;
    std::cout << "[HIP] D2H Copy Time: " << t_copy_d2h / 1000.0 << " s" << std::endl;
    std::cout << "[HIP] Total Time: " << t_total / 1000.0 << " s" << std::endl;

    hipEventDestroy(ev_start);
    hipEventDestroy(ev_h2d);
    hipEventDestroy(ev_kernel_start);
    hipEventDestroy(ev_kernel_stop);
    hipEventDestroy(ev_d2h);
    hipEventDestroy(ev_end);

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    // 新增：shared memory 优化评测
    std::vector<double> C_shared(N * P);
    float elapsed_ms_shared = 0;
    float t_copy_h2d_s = 0, t_copy_d2h_s = 0, t_kernel_s = 0, t_total_s = 0;
    hipEvent_t ev_start_s, ev_h2d_s, ev_kernel_start_s, ev_kernel_stop_s, ev_d2h_s, ev_end_s;
    hipEventCreate(&ev_start_s);
    hipEventCreate(&ev_h2d_s);
    hipEventCreate(&ev_kernel_start_s);
    hipEventCreate(&ev_kernel_stop_s);
    hipEventCreate(&ev_d2h_s);
    hipEventCreate(&ev_end_s);

    hipEventRecord(ev_start_s, 0);

    double *d_As, *d_Bs, *d_Cs;
    hipMalloc(&d_As, size_A);
    hipMalloc(&d_Bs, size_B);
    hipMalloc(&d_Cs, size_C);

    hipMemcpy(d_As, A.data(), size_A, hipMemcpyHostToDevice);
    hipMemcpy(d_Bs, B.data(), size_B, hipMemcpyHostToDevice);

    hipEventRecord(ev_h2d_s, 0);

    hipEventRecord(ev_kernel_start_s, 0);
    hipLaunchKernelGGL(matmul_kernel_shared, gridDim, blockDim, 0, 0, d_As, d_Bs, d_Cs, N, M, P);
    hipEventRecord(ev_kernel_stop_s, 0);
    hipEventSynchronize(ev_kernel_stop_s);

    hipMemcpy(C_shared.data(), d_Cs, size_C, hipMemcpyDeviceToHost);
    hipEventRecord(ev_d2h_s, 0);
    hipEventSynchronize(ev_d2h_s);

    hipEventRecord(ev_end_s, 0);
    hipEventSynchronize(ev_end_s);

    hipEventElapsedTime(&t_copy_h2d_s, ev_start_s, ev_h2d_s);
    hipEventElapsedTime(&t_kernel_s, ev_kernel_start_s, ev_kernel_stop_s);
    hipEventElapsedTime(&t_copy_d2h_s, ev_kernel_stop_s, ev_d2h_s);
    hipEventElapsedTime(&t_total_s, ev_start_s, ev_d2h_s);

    if (validate(C_ref, C_shared)) {
        std::cout << "[HIP-Shared] Valid: 1" << std::endl;
    } else {
        std::cout << "[HIP-Shared] Valid: 0" << std::endl;
    }
    std::cout << "[HIP-Shared] H2D Copy Time: " << t_copy_h2d_s / 1000.0 << " s" << std::endl;
    std::cout << "[HIP-Shared] Kernel Time: " << t_kernel_s / 1000.0 << " s" << std::endl;
    std::cout << "[HIP-Shared] D2H Copy Time: " << t_copy_d2h_s / 1000.0 << " s" << std::endl;
    std::cout << "[HIP-Shared] Total Time: " << t_total_s / 1000.0 << " s" << std::endl;

    hipEventDestroy(ev_start_s);
    hipEventDestroy(ev_h2d_s);
    hipEventDestroy(ev_kernel_start_s);
    hipEventDestroy(ev_kernel_stop_s);
    hipEventDestroy(ev_d2h_s);
    hipEventDestroy(ev_end_s);

    hipFree(d_As);
    hipFree(d_Bs);
    hipFree(d_Cs);

    // 需额外增加性能评测代码或其他工具进行评测
    return 0;
}
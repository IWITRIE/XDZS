#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <sstream>
#include <algorithm>

// 编译文件
// hipcc sourcefile_mlp.cpp -o mlp_full_dcu
// 执行文件
// ./mlp_full_dcu 或者 hipprof ./mlp_full_dcu

// 预定义参数，可根据需求修改
#define INPUT_DIM 10
#define HIDDEN_DIM 32
#define OUTPUT_DIM 1
#define BATCH_SIZE 256
#define EPOCHS 200
#define LEARNING_RATE 1e-4
#define TILE_SIZE 16  // 添加在其他宏定义之后


// 以下函数和main函数均不为固定形式，可自行按照需求修改

// HIP kernels函数形式，需要自行设计
__global__ void matmul(const double* A, const double* B, double* C, int M, int N, int K) {
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    double sum = 0.0;
    
    // 计算需要的tile数量
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < numTiles; ++tile) {
        // 加载数据到shared memory
        if (row < M && tile * TILE_SIZE + tx < K)
            As[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0;
            
        if (col < N && tile * TILE_SIZE + ty < K)
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0;
            
        __syncthreads();
        
        // 计算当前tile的部分和
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // 写回结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void compute_output_grad(const double* pred, const double* target, double* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = 2.0 * (pred[idx] - target[idx]);
    }
}

__global__ void compute_relu_backward(double* delta, const double* activ, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        delta[idx] = activ[idx] > 0 ? delta[idx] : 0;
    }
}

__global__ void compute_mse_loss(const double* pred, const double* target, double* loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        loss[idx] = (pred[idx] - target[idx]) * (pred[idx] - target[idx]);
    }
}

__global__ void sgd_update(double* weights, const double* grad, double lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= lr * grad[idx];
    }
}

// 添加ReLU激活函数的kernel
__global__ void relu_forward(double* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = output[idx] > 0 ? output[idx] : 0;
    }
}

// 加载带宽数据
std::vector<double> load_json_bandwidth(const std::string& filename) {
    std::vector<double> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return data;
    }

    std::string line;
    std::getline(file, line);
    
    // 移除开头的 '[' 和结尾的 ']'
    size_t start = line.find('[');
    size_t end = line.find_last_of(']');
    if (start != std::string::npos && end != std::string::npos) {
        line = line.substr(start + 1, end - start - 1);
    }

    // 按逗号分割并解析数字
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // 移除空格
        token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
        if (!token.empty()) {
            try {
                double value = std::stod(token);
                data.push_back(value);
            } catch (const std::exception& e) {
                std::cerr << "解析错误: " << token << std::endl;
            }
        }
    }
    
    return data;
}

// 创建数据集
void create_dataset(const std::vector<double>& data,
                    std::vector<double>& X,
                    std::vector<double>& y) {
    for (size_t i = 0; i < data.size() - INPUT_DIM; i++) {
        for (int j = 0; j < INPUT_DIM; j++) {
            X.push_back(data[i + j]);
        }
        y.push_back(data[i + INPUT_DIM]);
    }
}

// 数据归一化处理
void normalize_data(std::vector<double>& data, double& min_val, double& max_val) {
    min_val = *std::min_element(data.begin(), data.end());
    max_val = *std::max_element(data.begin(), data.end());
    for (auto& val : data) {
        val = (val - min_val) / (max_val - min_val);
    }
    return;
}

// 数据反归一化处理
void denormalize_data(std::vector<double>& data, double min_val, double max_val) {
    for (auto& val : data) {
        val = val * (max_val - min_val) + min_val;
    }
    return;
}

// ----------------------------- Main -------------------------------
int main() {
    // 初始化权重和偏置
    std::vector<double> w1(INPUT_DIM * HIDDEN_DIM);
    std::vector<double> w2(HIDDEN_DIM * OUTPUT_DIM);
    std::vector<double> b1(HIDDEN_DIM);
    std::vector<double> b2(OUTPUT_DIM);
    
    // 随机初始化权重
    for(auto& w : w1) w = 0.01 * (rand() / (double)RAND_MAX);
    for(auto& w : w2) w = 0.01 * (rand() / (double)RAND_MAX);
    
    // 分配GPU内存
    double *d_w1, *d_w2, *d_b1, *d_b2;
    hipMalloc(&d_w1, w1.size() * sizeof(double));
    hipMalloc(&d_w2, w2.size() * sizeof(double));
    hipMalloc(&d_b1, b1.size() * sizeof(double));
    hipMalloc(&d_b2, b2.size() * sizeof(double));
    
    // 训练循环
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // 前向传播
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, 
                     (M + TILE_SIZE - 1) / TILE_SIZE);
                     
        // 调用优化后的matmul
        matmul<<<gridDim, blockDim>>>(d_input, d_w1, d_hidden, 
                                     BATCH_SIZE, HIDDEN_DIM, INPUT_DIM);
        relu_forward<<<(HIDDEN_DIM + 255) / 256, 256>>>(d_hidden, HIDDEN_DIM * BATCH_SIZE);
        
        gridDim.x = (OUTPUT_DIM + 15) / 16;
        gridDim.y = (BATCH_SIZE + 15) / 16;
        matmul<<<gridDim, blockDim>>>(d_hidden, d_w2, d_output, BATCH_SIZE, OUTPUT_DIM, HIDDEN_DIM);
        
        // 反向传播
        compute_output_grad<<<(BATCH_SIZE + 255) / 256, 256>>>(d_output, d_target, d_output_grad, BATCH_SIZE);
        compute_relu_backward<<<(HIDDEN_DIM * BATCH_SIZE + 255) / 256, 256>>>(
            d_hidden_grad, d_hidden, HIDDEN_DIM * BATCH_SIZE);
            
        // 参数更新
        sgd_update<<<(w1.size() + 255) / 256, 256>>>(d_w1, d_w1_grad, LEARNING_RATE, w1.size());
        sgd_update<<<(w2.size() + 255) / 256, 256>>>(d_w2, d_w2_grad, LEARNING_RATE, w2.size());
    }
    
    // 清理GPU内存
    hipFree(d_w1);
    hipFree(d_w2);
    hipFree(d_b1);
    hipFree(d_b2);
    
    return 0;
}
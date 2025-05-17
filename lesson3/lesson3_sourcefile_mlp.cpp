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

    std::string line, all;
    while (std::getline(file, line)) {
        all += line;
    }

    // 移除开头的 '[' 和结尾的 ']'
    size_t start = all.find('[');
    size_t end = all.find_last_of(']');
    if (start != std::string::npos && end != std::string::npos) {
        all = all.substr(start + 1, end - start - 1);
    }

    // 按逗号分割并解析数字
    std::stringstream ss(all);
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

// 新增：计算 w2 梯度的 kernel（假设 output_grad: [BATCH_SIZE, OUTPUT_DIM], hidden: [BATCH_SIZE, HIDDEN_DIM]）
__global__ void compute_w2_grad(const double* hidden, const double* output_grad, double* w2_grad, int batch, int hidden_dim, int output_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // hidden_dim
    int j = blockIdx.y * blockDim.y + threadIdx.y; // output_dim
    if (i < hidden_dim && j < output_dim) {
        double sum = 0.0;
        for (int b = 0; b < batch; ++b) {
            sum += hidden[b * hidden_dim + i] * output_grad[b * output_dim + j];
        }
        w2_grad[i * output_dim + j] = sum / batch;
    }
}

// 新增：计算 w1 梯度的 kernel（可用类似方式实现）

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

    // 加载数据
    std::vector<double> raw_data = load_json_bandwidth("/root/XDZS/lesson3/starlink_bw.json");
    if (raw_data.empty()) {
        std::cerr << "数据加载失败，程序退出。" << std::endl;
        return -1;
    }

    // 归一化
    double min_val, max_val;
    normalize_data(raw_data, min_val, max_val);

    // 构建数据集
    std::vector<double> X;
    std::vector<double> y;
    create_dataset(raw_data, X, y);

    // 分配输入、隐藏层、输出、目标、梯度等主机和设备内存
    std::vector<double> hidden(BATCH_SIZE * HIDDEN_DIM);
    std::vector<double> output(BATCH_SIZE * OUTPUT_DIM);
    std::vector<double> target(BATCH_SIZE * OUTPUT_DIM);
    std::vector<double> hidden_grad(BATCH_SIZE * HIDDEN_DIM, 0.0);
    std::vector<double> output_grad(BATCH_SIZE * OUTPUT_DIM, 0.0);
    std::vector<double> w1_grad(INPUT_DIM * HIDDEN_DIM, 0.0);
    std::vector<double> w2_grad(HIDDEN_DIM * OUTPUT_DIM, 0.0);

    double *d_input, *d_hidden, *d_output, *d_target;
    double *d_hidden_grad, *d_output_grad, *d_w1_grad, *d_w2_grad;
    double *d_w1, *d_w2, *d_b1, *d_b2;

    hipMalloc(&d_input, BATCH_SIZE * INPUT_DIM * sizeof(double));
    hipMalloc(&d_hidden, BATCH_SIZE * HIDDEN_DIM * sizeof(double));
    hipMalloc(&d_output, BATCH_SIZE * OUTPUT_DIM * sizeof(double));
    hipMalloc(&d_target, BATCH_SIZE * OUTPUT_DIM * sizeof(double));
    hipMalloc(&d_hidden_grad, BATCH_SIZE * HIDDEN_DIM * sizeof(double));
    hipMalloc(&d_output_grad, BATCH_SIZE * OUTPUT_DIM * sizeof(double));
    hipMalloc(&d_w1_grad, INPUT_DIM * HIDDEN_DIM * sizeof(double));
    hipMalloc(&d_w2_grad, HIDDEN_DIM * OUTPUT_DIM * sizeof(double));
    hipMalloc(&d_w1, w1.size() * sizeof(double));
    hipMalloc(&d_w2, w2.size() * sizeof(double));
    hipMalloc(&d_b1, b1.size() * sizeof(double));
    hipMalloc(&d_b2, b2.size() * sizeof(double));

    // 拷贝权重到设备
    hipMemcpy(d_w1, w1.data(), w1.size() * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_w2, w2.data(), w2.size() * sizeof(double), hipMemcpyHostToDevice);
    // ...如有需要，拷贝b1, b2...

    // 分配loss相关内存
    std::vector<double> loss_host(BATCH_SIZE * OUTPUT_DIM);
    double *d_loss;
    hipMalloc(&d_loss, BATCH_SIZE * OUTPUT_DIM * sizeof(double));

    // 训练循环
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // 每次处理BATCH_SIZE个样本
        hipMemcpy(d_input, X.data(), BATCH_SIZE * INPUT_DIM * sizeof(double), hipMemcpyHostToDevice);
        hipMemcpy(d_target, y.data(), BATCH_SIZE * OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice);

        // 前向传播
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim1((HIDDEN_DIM + TILE_SIZE - 1) / TILE_SIZE, 
                      (BATCH_SIZE + TILE_SIZE - 1) / TILE_SIZE);

        matmul<<<gridDim1, blockDim>>>(d_input, d_w1, d_hidden, 
                                       BATCH_SIZE, HIDDEN_DIM, INPUT_DIM);
        relu_forward<<<(HIDDEN_DIM * BATCH_SIZE + 255) / 256, 256>>>(d_hidden, HIDDEN_DIM * BATCH_SIZE);

        dim3 gridDim2((OUTPUT_DIM + TILE_SIZE - 1) / TILE_SIZE, 
                      (BATCH_SIZE + TILE_SIZE - 1) / TILE_SIZE);
        matmul<<<gridDim2, blockDim>>>(d_hidden, d_w2, d_output, BATCH_SIZE, OUTPUT_DIM, HIDDEN_DIM);

        // 反向传播
        // 1. 输出层梯度
        compute_output_grad<<<(BATCH_SIZE * OUTPUT_DIM + 255) / 256, 256>>>(d_output, d_target, d_output_grad, BATCH_SIZE * OUTPUT_DIM);

        // 2. 计算 w2_grad: hidden^T * output_grad / BATCH_SIZE
        // hidden: [BATCH_SIZE, HIDDEN_DIM], output_grad: [BATCH_SIZE, OUTPUT_DIM]
        // w2_grad: [HIDDEN_DIM, OUTPUT_DIM]
        dim3 blockW2(16, 16);
        dim3 gridW2((HIDDEN_DIM + 15) / 16, (OUTPUT_DIM + 15) / 16);
        matmul<<<gridW2, blockW2>>>(d_hidden, d_output_grad, d_w2_grad, HIDDEN_DIM, OUTPUT_DIM, BATCH_SIZE); // 注意：需要转置hidden
        // 由于matmul没有转置功能，这里建议实现转置或写一个转置kernel。简化起见，建议在主机端先转置hidden再拷贝到设备，或实现转置kernel。

        // 3. 反向传播到隐藏层
        // hidden_grad = output_grad * w2^T
        dim3 gridDim3((HIDDEN_DIM + TILE_SIZE - 1) / TILE_SIZE, 
                      (BATCH_SIZE + TILE_SIZE - 1) / TILE_SIZE);
        matmul<<<gridDim3, blockDim>>>(d_output_grad, d_w2, d_hidden_grad, BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM);

        // 4. relu反向
        compute_relu_backward<<<(HIDDEN_DIM * BATCH_SIZE + 255) / 256, 256>>>(d_hidden_grad, d_hidden, HIDDEN_DIM * BATCH_SIZE);

        // 5. 计算 w1_grad: input^T * hidden_grad / BATCH_SIZE
        // input: [BATCH_SIZE, INPUT_DIM], hidden_grad: [BATCH_SIZE, HIDDEN_DIM]
        // w1_grad: [INPUT_DIM, HIDDEN_DIM]
        dim3 blockW1(16, 16);
        dim3 gridW1((INPUT_DIM + 15) / 16, (HIDDEN_DIM + 15) / 16);
        matmul<<<gridW1, blockW1>>>(d_input, d_hidden_grad, d_w1_grad, INPUT_DIM, HIDDEN_DIM, BATCH_SIZE); // 同样需要input转置

        // 6. 参数更新
        sgd_update<<<(w1.size() + 255) / 256, 256>>>(d_w1, d_w1_grad, LEARNING_RATE, w1.size());
        sgd_update<<<(w2.size() + 255) / 256, 256>>>(d_w2, d_w2_grad, LEARNING_RATE, w2.size());

        // 计算并打印loss
        compute_mse_loss<<<(BATCH_SIZE * OUTPUT_DIM + 255) / 256, 256>>>(d_output, d_target, d_loss, BATCH_SIZE * OUTPUT_DIM);
        hipMemcpy(loss_host.data(), d_loss, BATCH_SIZE * OUTPUT_DIM * sizeof(double), hipMemcpyDeviceToHost);
        double loss_sum = 0.0;
        for (int i = 0; i < BATCH_SIZE * OUTPUT_DIM; ++i) loss_sum += loss_host[i];
        double loss_mean = loss_sum / (BATCH_SIZE * OUTPUT_DIM);
        if (epoch % 10 == 0 || epoch == EPOCHS - 1)
            std::cout << "Epoch " << epoch << ", Loss: " << loss_mean << std::endl;
    }

    // 清理GPU内存
    hipFree(d_input);
    hipFree(d_hidden);
    hipFree(d_output);
    hipFree(d_target);
    hipFree(d_hidden_grad);
    hipFree(d_output_grad);
    hipFree(d_w1_grad);
    hipFree(d_w2_grad);
    hipFree(d_w1);
    hipFree(d_w2);
    hipFree(d_b1);
    hipFree(d_b2);
    hipFree(d_loss);

    return 0;
}
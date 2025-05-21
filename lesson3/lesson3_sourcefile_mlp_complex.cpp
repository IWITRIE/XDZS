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
#define H1_DIM    64  // 第一隐藏层
#define H2_DIM    32  // 第二隐藏层
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

// 计算 W2 梯度： hidden^T * output_grad / batch
__global__ void compute_w2_grad(const double* hidden, const double* output_grad, double* w2_grad,
                                int batch, int hidden_dim, int output_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < hidden_dim && j < output_dim) {
        double sum = 0.0;
        for (int b = 0; b < batch; ++b)
            sum += hidden[b * hidden_dim + i] * output_grad[b * output_dim + j];
        w2_grad[i * output_dim + j] = sum / batch;
    }
}

// 计算 W1 梯度： input^T * hidden_grad / batch
__global__ void compute_w1_grad(const double* input, const double* hidden_grad, double* w1_grad,
                                int batch, int input_dim, int hidden_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < input_dim && j < hidden_dim) {
        double sum = 0.0;
        for (int b = 0; b < batch; ++b)
            sum += input[b * input_dim + i] * hidden_grad[b * hidden_dim + j];
        w1_grad[i * hidden_dim + j] = sum / batch;
    }
}

// 计算偏置梯度：各样本梯度之和 / batch
__global__ void compute_bias_grad(const double* grad, double* bias_grad, int batch, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        double sum = 0.0;
        for (int b = 0; b < batch; ++b)
            sum += grad[b * dim + idx];
        bias_grad[idx] = sum / batch;
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
    // 1. 数据准备
    double min_v, max_v;
    auto raw = load_json_bandwidth("./starlink_bw.json ");     // 使用正确文件名
    if (raw.empty()) {
        std::cerr << "加载带宽数据失败，程序退出。" << std::endl;
        return -1;
    }
    normalize_data(raw, min_v, max_v);
    std::vector<double> X, y;
    create_dataset(raw, X, y);
    if (X.empty() || y.empty()) {
        std::cerr << "样本数据不足，程序退出。" << std::endl;
        return -1;
    }
    size_t S = y.size(), train_n = size_t(S * 0.8), test_n = S - train_n;

    // 2. 划分训练/测试集 (按行)
    std::vector<double> hX_train(X.begin(), X.begin() + train_n * INPUT_DIM),
                        hY_train(y.begin(), y.begin() + train_n);
    std::vector<double> hX_test (X.begin() + train_n * INPUT_DIM, X.end()),
                        hY_test (y.begin() + train_n,       y.end());

    // 新增：Host 权重和偏置
    std::vector<double> w1(INPUT_DIM * H1_DIM);
    std::vector<double> w2(H1_DIM * H2_DIM);
    std::vector<double> w3(H2_DIM * OUTPUT_DIM);
    std::vector<double> b1(H1_DIM);
    std::vector<double> b2(H2_DIM);
    std::vector<double> b3(OUTPUT_DIM);

    // 3. 分配 Device 内存
    double *d_input, *d_target;
    double *d_h1, *d_h2, *d_output;  // 两层隐藏层的输出
    double *d_output_grad, *d_h2_grad, *d_h1_grad;
    double *d_w1, *d_w2, *d_w3, *d_b1, *d_b2, *d_b3; // 权重和偏置
    double *d_w1_grad, *d_w2_grad, *d_w3_grad;       // 权重梯度
    double *d_b1_grad, *d_b2_grad, *d_b3_grad;       // 偏置梯度

    // 分配内存
    hipMalloc(&d_input, train_n*INPUT_DIM*sizeof(double));
    hipMalloc(&d_target, train_n*OUTPUT_DIM*sizeof(double));
    hipMalloc(&d_h1, train_n*H1_DIM*sizeof(double));
    hipMalloc(&d_h2, train_n*H2_DIM*sizeof(double));
    hipMalloc(&d_output, train_n*OUTPUT_DIM*sizeof(double));
    hipMalloc(&d_output_grad, train_n*OUTPUT_DIM*sizeof(double));
    hipMalloc(&d_h2_grad, train_n*H2_DIM*sizeof(double));
    hipMalloc(&d_h1_grad, train_n*H1_DIM*sizeof(double));
    
    // 权重内存
    hipMalloc(&d_w1, INPUT_DIM*H1_DIM*sizeof(double));
    hipMalloc(&d_w2, H1_DIM*H2_DIM*sizeof(double));
    hipMalloc(&d_w3, H2_DIM*OUTPUT_DIM*sizeof(double));
    hipMalloc(&d_b1, H1_DIM*sizeof(double));
    hipMalloc(&d_b2, H2_DIM*sizeof(double));
    hipMalloc(&d_b3, OUTPUT_DIM*sizeof(double));
    
    // 梯度内存
    hipMalloc(&d_w1_grad, INPUT_DIM*H1_DIM*sizeof(double));
    hipMalloc(&d_w2_grad, H1_DIM*H2_DIM*sizeof(double));
    hipMalloc(&d_w3_grad, H2_DIM*OUTPUT_DIM*sizeof(double));
    hipMalloc(&d_b1_grad, H1_DIM*sizeof(double));
    hipMalloc(&d_b2_grad, H2_DIM*sizeof(double));
    hipMalloc(&d_b3_grad, OUTPUT_DIM*sizeof(double));

    // 4. 初始化权重并拷贝到设备
    for(auto& w : w1) w = 0.01 * (rand() / (double)RAND_MAX);
    for(auto& w : w2) w = 0.01 * (rand() / (double)RAND_MAX);
    for(auto& w : w3) w = 0.01 * (rand() / (double)RAND_MAX);

    hipMemcpy(d_w1, w1.data(), w1.size()*sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_w2, w2.data(), w2.size()*sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_w3, w3.data(), w3.size()*sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_b1, b1.data(), b1.size()*sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_b2, b2.data(), b2.size()*sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_b3, b3.data(), b3.size()*sizeof(double), hipMemcpyHostToDevice);
    
    // 新增：损失存储
    double *d_loss;
    hipMalloc(&d_loss, train_n * OUTPUT_DIM * sizeof(double));
    std::vector<double> loss_host(train_n * OUTPUT_DIM);

    // 5. 训练
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // 前向传播: Input→H1
        dim3 blk(TILE_SIZE, TILE_SIZE);
        dim3 grd1((H1_DIM+TILE_SIZE-1)/TILE_SIZE, (train_n+TILE_SIZE-1)/TILE_SIZE);
        matmul<<<grd1,blk>>>(d_input, d_w1, d_h1, train_n, H1_DIM, INPUT_DIM);
        relu_forward<<<(train_n*H1_DIM+255)/256,256>>>(d_h1, train_n*H1_DIM);
        
        // H1→H2
        dim3 grd2((H2_DIM+TILE_SIZE-1)/TILE_SIZE, (train_n+TILE_SIZE-1)/TILE_SIZE);
        matmul<<<grd2,blk>>>(d_h1, d_w2, d_h2, train_n, H2_DIM, H1_DIM);
        relu_forward<<<(train_n*H2_DIM+255)/256,256>>>(d_h2, train_n*H2_DIM);
        
        // H2→Output
        dim3 grd3((OUTPUT_DIM+TILE_SIZE-1)/TILE_SIZE, (train_n+TILE_SIZE-1)/TILE_SIZE);
        matmul<<<grd3,blk>>>(d_h2, d_w3, d_output, train_n, OUTPUT_DIM, H2_DIM);

        // 反向传播: 输出层梯度
        compute_output_grad<<<(train_n+255)/256,256>>>(d_output, d_target, d_output_grad, train_n*OUTPUT_DIM);
        
        // H2层梯度
        compute_relu_backward<<<(train_n*H2_DIM+255)/256,256>>>(d_h2_grad, d_h2, train_n*H2_DIM);
        
        // H1层梯度
        compute_relu_backward<<<(train_n*H1_DIM+255)/256,256>>>(d_h1_grad, d_h1, train_n*H1_DIM);
        
        // 计算权重梯度
        // W3梯度
        dim3 blockW3(16,16), gridW3((H2_DIM+15)/16, (OUTPUT_DIM+15)/16);
        compute_w2_grad<<<gridW3,blockW3>>>(d_h2, d_output_grad, d_w3_grad, train_n, H2_DIM, OUTPUT_DIM);
        compute_bias_grad<<<(OUTPUT_DIM+255)/256,256>>>(d_output_grad, d_b3_grad, train_n, OUTPUT_DIM);
        
        // W2梯度
        dim3 blockW2(16,16), gridW2((H1_DIM+15)/16, (H2_DIM+15)/16);
        compute_w2_grad<<<gridW2,blockW2>>>(d_h1, d_h2_grad, d_w2_grad, train_n, H1_DIM, H2_DIM);
        compute_bias_grad<<<(H2_DIM+255)/256,256>>>(d_h2_grad, d_b2_grad, train_n, H2_DIM);
        
        // W1梯度
        dim3 blockW1(16,16), gridW1((INPUT_DIM+15)/16, (H1_DIM+15)/16);
        compute_w1_grad<<<gridW1,blockW1>>>(d_input, d_h1_grad, d_w1_grad, train_n, INPUT_DIM, H1_DIM);
        compute_bias_grad<<<(H1_DIM+255)/256,256>>>(d_h1_grad, d_b1_grad, train_n, H1_DIM);

        // SGD更新
        sgd_update<<<(w1.size()+255)/256,256>>>(d_w1, d_w1_grad, LEARNING_RATE, w1.size());
        sgd_update<<<(w2.size()+255)/256,256>>>(d_w2, d_w2_grad, LEARNING_RATE, w2.size());
        sgd_update<<<(w3.size()+255)/256,256>>>(d_w3, d_w3_grad, LEARNING_RATE, w3.size());
        sgd_update<<<(b1.size()+255)/256,256>>>(d_b1, d_b1_grad, LEARNING_RATE, b1.size());
        sgd_update<<<(b2.size()+255)/256,256>>>(d_b2, d_b2_grad, LEARNING_RATE, b2.size());
        sgd_update<<<(b3.size()+255)/256,256>>>(d_b3, d_b3_grad, LEARNING_RATE, b3.size());

        // 计算损失
        compute_mse_loss<<<(train_n*OUTPUT_DIM+255)/256,256>>>(d_output, d_target, d_loss,
                                                               train_n*OUTPUT_DIM);
        hipMemcpy(loss_host.data(), d_loss, train_n*OUTPUT_DIM*sizeof(double), hipMemcpyDeviceToHost);
        double sum_loss = 0.0;
        for (double v : loss_host) sum_loss += v;
        double mean_loss = sum_loss / (train_n * OUTPUT_DIM);
        std::cout << "Epoch " << (epoch+1) << " Loss: " << mean_loss << std::endl;
    }

    hipFree(d_loss);
    
    // 打印网络结构
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "神经网络结构：" << std::endl;
    std::cout << "输入层: " << INPUT_DIM << " 维特征" << std::endl;
    std::cout << "隐藏层1: " << H1_DIM << " 个神经元 (ReLU激活)" << std::endl;
    std::cout << "隐藏层2: " << H2_DIM << " 个神经元 (ReLU激活)" << std::endl;
    std::cout << "输出层: " << OUTPUT_DIM << " 个神经元 (线性输出)" << std::endl;
    std::cout << "总参数量: " << w1.size() + w2.size() + w3.size() + b1.size() + b2.size() + b3.size() << " 个" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    // 清理内存
    hipFree(d_input); hipFree(d_target);
    hipFree(d_h1); hipFree(d_h2); hipFree(d_output);
    hipFree(d_output_grad); hipFree(d_h2_grad); hipFree(d_h1_grad);
    hipFree(d_w1); hipFree(d_w2); hipFree(d_w3);
    hipFree(d_b1); hipFree(d_b2); hipFree(d_b3);
    hipFree(d_w1_grad); hipFree(d_w2_grad); hipFree(d_w3_grad);
    hipFree(d_b1_grad); hipFree(d_b2_grad); hipFree(d_b3_grad);
    
    return 0;
}
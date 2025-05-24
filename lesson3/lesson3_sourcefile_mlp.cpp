#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <sstream>
#include <algorithm>
#include <iomanip>  // 添加这个头文件用于 std::setprecision

// 编译文件
// hipcc sourcefile_mlp.cpp -o mlp_full_dcu
// 执行文件
// ./mlp_full_dcu 或者 hipprof ./mlp_full_dcu

// 预定义参数，可根据需求修改
#define INPUT_DIM 10
#define HIDDEN_DIM 32
#define OUTPUT_DIM 1
#define BATCH_SIZE 256
#define EPOCHS 10000
#define LEARNING_RATE 3e-3
#define TILE_SIZE 16  // 修改为16以避免超出限制


// 以下函数和main函数均不为固定形式，可自行按照需求修改

// HIP kernels函数形式，需要自行设计
__global__ void __launch_bounds__(256) matmul(const double* A, const double* B, double* C, int M, int N, int K) {
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

__global__ void __launch_bounds__(256) compute_output_grad(const double* pred, const double* target, double* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = 2.0 * (pred[idx] - target[idx]);
    }
}

__global__ void __launch_bounds__(256) compute_relu_backward(double* delta, const double* activ, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        delta[idx] = activ[idx] > 0 ? delta[idx] : 0;
    }
}

__global__ void __launch_bounds__(256) compute_mse_loss(const double* pred, const double* target, double* loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        loss[idx] = (pred[idx] - target[idx]) * (pred[idx] - target[idx]);
    }
}

__global__ void __launch_bounds__(256) sgd_update(double* weights, const double* grad, double lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= lr * grad[idx];
    }
}

// 添加ReLU激活函数的kernel
__global__ void __launch_bounds__(256) relu_forward(double* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = output[idx] > 0 ? output[idx] : 0;
    }
}

// 计算 W2 梯度： hidden^T * output_grad / batch
__global__ void __launch_bounds__(256) compute_w2_grad(const double* hidden, const double* output_grad, double* w2_grad,
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
__global__ void __launch_bounds__(256) compute_w1_grad(const double* input, const double* hidden_grad, double* w1_grad,
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
__global__ void __launch_bounds__(256) compute_bias_grad(const double* grad, double* bias_grad, int batch, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        double sum = 0.0;
        for (int b = 0; b < batch; ++b)
            sum += grad[b * dim + idx];
        bias_grad[idx] = sum / batch;
    }
}

// 添加偏置加法kernel
__global__ void __launch_bounds__(256) add_bias(double* output, const double* bias, int batch_size, int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * output_dim) {
        int bias_idx = idx % output_dim;
        output[idx] += bias[bias_idx];
    }
}

// 修正反向传播中的hidden_grad计算
__global__ void __launch_bounds__(256) compute_hidden_grad_from_output(const double* output_grad, const double* w2, 
                                               double* hidden_grad, int batch, int hidden_dim, int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * hidden_dim) {
        int batch_idx = idx / hidden_dim;
        int hidden_idx = idx % hidden_dim;
        
        double sum = 0.0;
        for (int o = 0; o < output_dim; ++o) {
            sum += output_grad[batch_idx * output_dim + o] * w2[hidden_idx * output_dim + o];
        }
        hidden_grad[idx] = sum;
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
    auto raw = load_json_bandwidth("./starlink_bw.json");
    if (raw.empty()) {
        std::cerr << "加载带宽数据失败，程序退出。" << std::endl;
        return -1;
    }
    
    std::cout << "原始数据范围: [" << *std::min_element(raw.begin(), raw.end()) 
              << ", " << *std::max_element(raw.begin(), raw.end()) << "]" << std::endl;
    std::cout << "原始数据大小: " << raw.size() << std::endl;
    
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

    // 新增：Host 权重和偏置 - 使用更好的初始化
    std::vector<double> w1(INPUT_DIM * HIDDEN_DIM);
    std::vector<double> w2(HIDDEN_DIM * OUTPUT_DIM);
    std::vector<double> b1(HIDDEN_DIM, 0.0);  // 偏置初始化为0
    std::vector<double> b2(OUTPUT_DIM, 0.0);

    // Xavier初始化
    double xavier_w1 = sqrt(2.0 / (INPUT_DIM + HIDDEN_DIM));
    double xavier_w2 = sqrt(2.0 / (HIDDEN_DIM + OUTPUT_DIM));
    
    for(auto& w : w1) w = xavier_w1 * (2.0 * rand() / RAND_MAX - 1.0);
    for(auto& w : w2) w = xavier_w2 * (2.0 * rand() / RAND_MAX - 1.0);

    // 3. 分配 Device 内存
    double *d_input, *d_target;
    double *d_hidden, *d_output;
    double *d_output_grad, *d_hidden_grad;
    double *d_w1, *d_w2, *d_b1, *d_b2;
    double *d_w1_grad, *d_w2_grad, *d_b1_grad, *d_b2_grad;
    hipMalloc(&d_input,      train_n*INPUT_DIM*sizeof(double));
    hipMalloc(&d_target,     train_n*OUTPUT_DIM*sizeof(double));
    hipMalloc(&d_hidden,     train_n*HIDDEN_DIM *sizeof(double));
    hipMalloc(&d_output,     train_n*OUTPUT_DIM  *sizeof(double));
    hipMalloc(&d_output_grad,train_n*OUTPUT_DIM  *sizeof(double));
    hipMalloc(&d_hidden_grad,train_n*HIDDEN_DIM *sizeof(double));
    hipMalloc(&d_w1, INPUT_DIM*HIDDEN_DIM*sizeof(double));
    hipMalloc(&d_w2, HIDDEN_DIM*OUTPUT_DIM*sizeof(double));
    hipMalloc(&d_b1, HIDDEN_DIM*sizeof(double));
    hipMalloc(&d_b2, OUTPUT_DIM*sizeof(double));
    hipMalloc(&d_w1_grad, INPUT_DIM*HIDDEN_DIM*sizeof(double));
    hipMalloc(&d_w2_grad, HIDDEN_DIM*OUTPUT_DIM*sizeof(double));
    hipMalloc(&d_b1_grad, HIDDEN_DIM*sizeof(double));
    hipMalloc(&d_b2_grad, OUTPUT_DIM*sizeof(double));

    // 4. 拷贝训练数据 & 随机初始化并拷贝参数
    hipMemcpy(d_input, hX_train.data(),  train_n*INPUT_DIM*sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_target,hY_train.data(),  train_n*OUTPUT_DIM*sizeof(double), hipMemcpyHostToDevice);
    // 随机初始化权重
    for(auto& w : w1) w = 0.01 * (rand() / (double)RAND_MAX);
    for(auto& w : w2) w = 0.01 * (rand() / (double)RAND_MAX);
    hipMemcpy(d_w1, w1.data(), w1.size()*sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_w2, w2.data(), w2.size()*sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_b1, b1.data(), b1.size()*sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_b2, b2.data(), b2.size()*sizeof(double), hipMemcpyHostToDevice);

    // 新增：损失存储
    double *d_loss;
    hipMalloc(&d_loss, train_n * OUTPUT_DIM * sizeof(double));
    std::vector<double> loss_host(train_n * OUTPUT_DIM);
    
    // 新增：训练损失记录
    std::vector<double> epoch_losses;
    std::vector<int> epoch_numbers;

    // 5. 训练：正/反向、梯度下降
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // 前向：Input→Hidden
        dim3 blk(TILE_SIZE, TILE_SIZE), grd((HIDDEN_DIM+TILE_SIZE-1)/TILE_SIZE, (train_n+TILE_SIZE-1)/TILE_SIZE);
        matmul<<<grd,blk>>>(d_input, d_w1, d_hidden, train_n, HIDDEN_DIM, INPUT_DIM);
        add_bias<<<(train_n*HIDDEN_DIM+255)/256,256>>>(d_hidden, d_b1, train_n, HIDDEN_DIM);
        relu_forward<<<(train_n*HIDDEN_DIM+255)/256,256>>>(d_hidden, train_n*HIDDEN_DIM);
        
        // Hidden→Output
        grd.x = (OUTPUT_DIM+TILE_SIZE-1)/TILE_SIZE;
        grd.y = (train_n +TILE_SIZE-1)/TILE_SIZE;
        matmul<<<grd,blk>>>(d_hidden, d_w2, d_output, train_n, OUTPUT_DIM, HIDDEN_DIM);
        add_bias<<<(train_n*OUTPUT_DIM+255)/256,256>>>(d_output, d_b2, train_n, OUTPUT_DIM);

        // 误差&梯度
        compute_output_grad<<<(train_n*OUTPUT_DIM+255)/256,256>>>(d_output, d_target, d_output_grad, train_n*OUTPUT_DIM);
        
        // 修正反向传播：先计算hidden_grad，再应用ReLU反向
        compute_hidden_grad_from_output<<<(train_n*HIDDEN_DIM+255)/256,256>>>(
            d_output_grad, d_w2, d_hidden_grad, train_n, HIDDEN_DIM, OUTPUT_DIM);
        compute_relu_backward<<<(train_n*HIDDEN_DIM+255)/256,256>>>(d_hidden_grad, d_hidden, train_n*HIDDEN_DIM);
        
        // 计算权重梯度（W2, B2）
        // W2_grad = H^T * output_grad
        // B2_grad = reduce_sum(output_grad)
        // 同理计算 W1_grad, B1_grad → 省略细节，可调用 matmul 及自定义 reduce kernel
        dim3 blockW2(16,16), gridW2((HIDDEN_DIM+15)/16,(OUTPUT_DIM+15)/16);
        compute_w2_grad<<<gridW2,blockW2>>>(d_hidden, d_output_grad, d_w2_grad,
                                           train_n, HIDDEN_DIM, OUTPUT_DIM);
        compute_bias_grad<<<(OUTPUT_DIM+255)/256,256>>>(d_output_grad, d_b2_grad,
                                                       train_n, OUTPUT_DIM);

        dim3 blockW1(16,16), gridW1((INPUT_DIM+15)/16,(HIDDEN_DIM+15)/16);
        compute_w1_grad<<<gridW1,blockW1>>>(d_input, d_hidden_grad, d_w1_grad,
                                           train_n, INPUT_DIM, HIDDEN_DIM);
        compute_bias_grad<<<(HIDDEN_DIM+255)/256,256>>>(d_hidden_grad, d_b1_grad,
                                                       train_n, HIDDEN_DIM);

        // SGD 更新
        sgd_update<<<(w1.size()+255)/256,256>>>(d_w1, d_w1_grad, LEARNING_RATE, w1.size());
        sgd_update<<<(w2.size()+255)/256,256>>>(d_w2, d_w2_grad, LEARNING_RATE, w2.size());
        sgd_update<<<(b1.size()+255)/256,256>>>(d_b1, d_b1_grad, LEARNING_RATE, b1.size());
        sgd_update<<<(b2.size()+255)/256,256>>>(d_b2, d_b2_grad, LEARNING_RATE, b2.size());

        // 计算本轮 loss (每10轮记录一次以减少开销)
        if (epoch % 10 == 0 || epoch < 2000) {
            compute_mse_loss<<<(train_n*OUTPUT_DIM+255)/256,256>>>(d_output, d_target, d_loss,
                                                                   train_n*OUTPUT_DIM);
            hipMemcpy(loss_host.data(), d_loss, train_n*OUTPUT_DIM*sizeof(double), hipMemcpyDeviceToHost);
            double sum_loss = 0.0;
            for (double v : loss_host) sum_loss += v;
            double mean_loss = sum_loss / (train_n * OUTPUT_DIM);
            
            // 保存损失值和轮数
            epoch_losses.push_back(mean_loss);
            epoch_numbers.push_back(epoch + 1);
            
            // 打印进度 (每100轮打印一次)
            if (epoch % 100 == 0) {
                std::cout << "Epoch " << (epoch+1) << " Loss: " << mean_loss << std::endl;
            }
        }
    }

    // 保存训练损失历史到CSV文件
    std::ofstream loss_file("training_loss_history.csv");
    loss_file << "Epoch,Loss\n";
    for (size_t i = 0; i < epoch_losses.size(); ++i) {
        loss_file << epoch_numbers[i] << "," << std::fixed << std::setprecision(8) << epoch_losses[i] << "\n";
    }
    loss_file.close();
    std::cout << "训练损失历史已保存到 training_loss_history.csv" << std::endl;

    hipFree(d_loss);
    
    // 6. 改进的滑动窗口预测
    std::cout << "\n开始滑动窗口预测..." << std::endl;
    std::cout << "训练样本数: " << train_n << ", 测试样本数: " << test_n << std::endl;
    
    // 使用滑动窗口进行逐步预测
    std::vector<double> sliding_predictions;
    std::vector<double> sliding_true_values;
    std::vector<size_t> sliding_positions;
    
    // 分配单个样本的GPU内存
    double *d_single_input, *d_single_hidden, *d_single_output;
    hipMalloc(&d_single_input, INPUT_DIM*sizeof(double));
    hipMalloc(&d_single_hidden, HIDDEN_DIM*sizeof(double));
    hipMalloc(&d_single_output, OUTPUT_DIM*sizeof(double));
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 从训练集的最后INPUT_DIM个点开始
    std::vector<double> current_window(raw.begin() + train_n, raw.begin() + train_n + INPUT_DIM);
    
    for (size_t i = 0; i < test_n; ++i) {
        // 准备当前窗口的输入（已归一化）
        hipMemcpy(d_single_input, current_window.data(), INPUT_DIM*sizeof(double), hipMemcpyHostToDevice);
        
        // 前向传播
        dim3 blk_single(TILE_SIZE, TILE_SIZE);
        dim3 grd_single((HIDDEN_DIM+TILE_SIZE-1)/TILE_SIZE, (1+TILE_SIZE-1)/TILE_SIZE);
        matmul<<<grd_single,blk_single>>>(d_single_input, d_w1, d_single_hidden, 1, HIDDEN_DIM, INPUT_DIM);
        add_bias<<<(HIDDEN_DIM+255)/256,256>>>(d_single_hidden, d_b1, 1, HIDDEN_DIM);
        relu_forward<<<(HIDDEN_DIM+255)/256,256>>>(d_single_hidden, HIDDEN_DIM);
        
        grd_single.x = (OUTPUT_DIM+TILE_SIZE-1)/TILE_SIZE;
        matmul<<<grd_single,blk_single>>>(d_single_hidden, d_w2, d_single_output, 1, OUTPUT_DIM, HIDDEN_DIM);
        add_bias<<<(OUTPUT_DIM+255)/256,256>>>(d_single_output, d_b2, 1, OUTPUT_DIM);
        
        // 获取预测结果
        double prediction_normalized;
        hipMemcpy(&prediction_normalized, d_single_output, sizeof(double), hipMemcpyDeviceToHost);
        
        // 保存结果
        sliding_predictions.push_back(prediction_normalized);
        sliding_true_values.push_back(hY_test[i]);  // 已归一化的真实值
        sliding_positions.push_back(train_n + INPUT_DIM + i);
        
        // 更新滑动窗口：移除第一个元素，添加真实值（用于下一次预测）
        if (i < test_n - 1) {
            current_window.erase(current_window.begin());
            current_window.push_back(raw[train_n + INPUT_DIM + i]);  // 添加真实的归一化值
        }
    }
    
    hipDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto inference_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // 反归一化预测结果和真实值
    std::vector<double> denorm_pred = sliding_predictions;
    std::vector<double> denorm_test = sliding_true_values;
    denormalize_data(denorm_pred, min_v, max_v);
    denormalize_data(denorm_test, min_v, max_v);
    
    // 计算性能指标
    double test_mse = 0.0;
    for (size_t i = 0; i < test_n; ++i) {
        double diff = denorm_pred[i] - denorm_test[i];
        test_mse += diff * diff;
    }
    test_mse /= test_n;
    
    std::cout << "滑动窗口预测完成！" << std::endl;
    std::cout << "预测时间: " << inference_time.count() << " 微秒" << std::endl;
    std::cout << "测试集MSE: " << test_mse << std::endl;
    
    // 检查预测值范围
    double norm_pred_min = *std::min_element(sliding_predictions.begin(), sliding_predictions.end());
    double norm_pred_max = *std::max_element(sliding_predictions.begin(), sliding_predictions.end());
    std::cout << "归一化预测值范围: [" << norm_pred_min << ", " << norm_pred_max << "]" << std::endl;
    std::cout << "反归一化后预测值范围: [" << *std::min_element(denorm_pred.begin(), denorm_pred.end()) 
              << ", " << *std::max_element(denorm_pred.begin(), denorm_pred.end()) << "]" << std::endl;
    
    // 保存结果到CSV文件
    std::ofstream csv_file("inference_results.csv");
    csv_file << "Position,True_Value,Predicted_Value,Error,Abs_Error,Normalized_True,Normalized_Pred\n";
    
    for (size_t i = 0; i < test_n; ++i) {
        double error = denorm_pred[i] - denorm_test[i];
        csv_file << sliding_positions[i] << "," 
                << denorm_test[i] << ","
                << denorm_pred[i] << ","
                << error << ","
                << std::abs(error) << ","
                << sliding_true_values[i] << ","
                << sliding_predictions[i] << "\n";
    }
    csv_file.close();
    std::cout << "滑动窗口预测结果已保存到 inference_results.csv" << std::endl;
    
    // 输出前几个样本进行调试
    std::cout << "\n前5个测试样本详情:" << std::endl;
    std::cout << "位置\t真实值\t\t预测值\t\t误差\t\t归一化真实值\t归一化预测值" << std::endl;
    for (size_t i = 0; i < std::min(5UL, test_n); ++i) {
        double error = denorm_pred[i] - denorm_test[i];
        std::cout << sliding_positions[i] << "\t" 
                  << std::fixed << std::setprecision(6) << denorm_test[i] << "\t"
                  << denorm_pred[i] << "\t"
                  << error << "\t"
                  << sliding_true_values[i] << "\t\t"
                  << sliding_predictions[i] << std::endl;
    }

    // 清理内存
    hipFree(d_single_input);
    hipFree(d_single_hidden);
    hipFree(d_single_output);

    // 7. 清理
    hipFree(d_input); hipFree(d_target);
    hipFree(d_hidden); hipFree(d_output);
    hipFree(d_output_grad); hipFree(d_hidden_grad);
    hipFree(d_w1); hipFree(d_w2); hipFree(d_b1); hipFree(d_b2);
    hipFree(d_w1_grad); hipFree(d_w2_grad); hipFree(d_b1_grad); hipFree(d_b2_grad);

    return 0;
}
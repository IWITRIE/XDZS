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

// 新增：MLP 网络封装
struct MLPNetwork {
    int in_dim, hid_dim, out_dim;
    double lr;
    // Device 指针
    double *d_input, *d_hidden, *d_output, *d_target;
    double *d_w1, *d_b1, *d_w2, *d_b2;
    double *d_w1g, *d_b1g, *d_w2g, *d_b2g;
    double *d_og, *d_hg;  // output_grad, hidden_grad

    MLPNetwork(int in, int hid, int out, double lr):
        in_dim(in), hid_dim(hid), out_dim(out), lr(lr) {}

    void init(size_t batch) {
        // 分配参数与中间量
        hipMalloc(&d_input,  batch*in_dim*sizeof(double));
        hipMalloc(&d_hidden, batch*hid_dim*sizeof(double));
        hipMalloc(&d_output, batch*out_dim*sizeof(double));
        hipMalloc(&d_target, batch*out_dim*sizeof(double));
        hipMalloc(&d_w1, in_dim*hid_dim*sizeof(double));
        hipMalloc(&d_b1, hid_dim*sizeof(double));
        hipMalloc(&d_w2, hid_dim*out_dim*sizeof(double));
        hipMalloc(&d_b2, out_dim*sizeof(double));
        hipMalloc(&d_w1g, in_dim*hid_dim*sizeof(double));
        hipMalloc(&d_b1g, hid_dim*sizeof(double));
        hipMalloc(&d_w2g, hid_dim*out_dim*sizeof(double));
        hipMalloc(&d_b2g, out_dim*sizeof(double));
        hipMalloc(&d_hg,  batch*hid_dim*sizeof(double));
        hipMalloc(&d_og,  batch*out_dim*sizeof(double));
        // 随机初始化主机参数并拷贝，可自行替换为自定义 init
        std::vector<double> h1(in_dim*hid_dim), hb1(hid_dim),
                            h2(hid_dim*out_dim), hb2(out_dim);
        for(auto& v:h1) v=0.01*(rand()/(double)RAND_MAX);
        for(auto& v:hb1) v=0;
        for(auto& v:h2) v=0.01*(rand()/(double)RAND_MAX);
        for(auto& v:hb2) v=0;
        hipMemcpy(d_w1,h1.data(),h1.size()*sizeof(double),hipMemcpyHostToDevice);
        hipMemcpy(d_b1,hb1.data(),hb1.size()*sizeof(double),hipMemcpyHostToDevice);
        hipMemcpy(d_w2,h2.data(),h2.size()*sizeof(double),hipMemcpyHostToDevice);
        hipMemcpy(d_b2,hb2.data(),hb2.size()*sizeof(double),hipMemcpyHostToDevice);
    }

    void forward(size_t batch) {
        dim3 blk(TILE_SIZE,TILE_SIZE),
             grd1((hid_dim+TILE_SIZE-1)/TILE_SIZE,(batch+TILE_SIZE-1)/TILE_SIZE);
        matmul<<<grd1,blk>>>(d_input,d_w1,d_hidden,batch,hid_dim,in_dim);
        relu_forward<<<(batch*hid_dim+255)/256,256>>>(d_hidden,batch*hid_dim);
        dim3 grd2((out_dim+TILE_SIZE-1)/TILE_SIZE,(batch+TILE_SIZE-1)/TILE_SIZE);
        matmul<<<grd2,blk>>>(d_hidden,d_w2,d_output,batch,out_dim,hid_dim);
    }

    void backward(size_t batch) {
        // output_grad
        compute_output_grad<<<(batch*out_dim+255)/256,256>>>(d_output,d_target,d_og,batch*out_dim);
        // hidden_grad
        compute_relu_backward<<<(batch*hid_dim+255)/256,256>>>(d_hg,d_hidden,batch*hid_dim);
        // weights/bias 梯度
        dim3 b2(16,16), g2((hid_dim+15)/16,(out_dim+15)/16);
        compute_w2_grad<<<g2,b2>>>(d_hidden,d_og,d_w2g,batch,hid_dim,out_dim);
        compute_bias_grad<<<(out_dim+255)/256,256>>>(d_og,d_b2g,batch,out_dim);
        dim3 b1(16,16), g1((in_dim+15)/16,(hid_dim+15)/16);
        compute_w1_grad<<<g1,b1>>>(d_input,d_hg,d_w1g,batch,in_dim,hid_dim);
        compute_bias_grad<<<(hid_dim+255)/256,256>>>(d_hg,d_b1g,batch,hid_dim);
    }

    void update() {
        int sz1=in_dim*hid_dim, sz2=hid_dim*out_dim;
        sgd_update<<<(sz1+255)/256,256>>>(d_w1,d_w1g,lr,sz1);
        sgd_update<<<(hid_dim+255)/256,256>>>(d_b1,d_b1g,lr,hid_dim);
        sgd_update<<<(sz2+255)/256,256>>>(d_w2,d_w2g,lr,sz2);
        sgd_update<<<(out_dim+255)/256,256>>>(d_b2,d_b2g,lr,out_dim);
    }

    double compute_loss(size_t batch, std::vector<double>& loss_host) {
        double *d_loss; hipMalloc(&d_loss,batch*out_dim*sizeof(double));
        compute_mse_loss<<<(batch*out_dim+255)/256,256>>>(d_output,d_target,d_loss,batch*out_dim);
        hipMemcpy(loss_host.data(),d_loss,batch*out_dim*sizeof(double),hipMemcpyDeviceToHost);
        hipFree(d_loss);
        double sum=0; for(double v:loss_host) sum+=v;
        return sum/(batch*out_dim);
    }

    void destroy() {
        hipFree(d_input); hipFree(d_hidden); hipFree(d_output); hipFree(d_target);
        hipFree(d_w1); hipFree(d_b1); hipFree(d_w2); hipFree(d_b2);
        hipFree(d_w1g);hipFree(d_b1g);hipFree(d_w2g);hipFree(d_b2g);
        hipFree(d_hg); hipFree(d_og);
    }
};

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
    std::vector<double> w1(INPUT_DIM * HIDDEN_DIM);
    std::vector<double> w2(HIDDEN_DIM * OUTPUT_DIM);
    std::vector<double> b1(HIDDEN_DIM);
    std::vector<double> b2(OUTPUT_DIM);

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

    // 5. 训练：正/反向、梯度下降
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // 前向：Input→Hidden
        dim3 blk(TILE_SIZE, TILE_SIZE), grd((HIDDEN_DIM+TILE_SIZE-1)/TILE_SIZE, (train_n+TILE_SIZE-1)/TILE_SIZE);
        matmul<<<grd,blk>>>(d_input, d_w1, d_hidden, train_n, HIDDEN_DIM, INPUT_DIM);
        relu_forward<<<(train_n*HIDDEN_DIM+255)/256,256>>>(d_hidden, train_n*HIDDEN_DIM);
        // Hidden→Output
        grd.x = (OUTPUT_DIM+TILE_SIZE-1)/TILE_SIZE;
        grd.y = (train_n +TILE_SIZE-1)/TILE_SIZE;
        matmul<<<grd,blk>>>(d_hidden, d_w2, d_output, train_n, OUTPUT_DIM, HIDDEN_DIM);

        // 误差&梯度
        compute_output_grad<<<(train_n*OUTPUT_DIM+255)/256,256>>>(d_output, d_target, d_output_grad, train_n*OUTPUT_DIM);
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

        // 计算并打印本轮 loss
        compute_mse_loss<<<(train_n*OUTPUT_DIM+255)/256,256>>>(d_output, d_target, d_loss,
                                                               train_n*OUTPUT_DIM);
        hipMemcpy(loss_host.data(), d_loss, train_n*OUTPUT_DIM*sizeof(double), hipMemcpyDeviceToHost);
        double sum_loss = 0.0;
        for (double v : loss_host) sum_loss += v;
        double mean_loss = sum_loss / (train_n * OUTPUT_DIM);
        std::cout << "Epoch " << (epoch+1) << " Loss: " << mean_loss << std::endl;
    }

    hipFree(d_loss);
    // 6. 推理 & 性能评测（同训练前向部分，但输入为 hX_test）
    //    - 复用 d_input, d_hidden, d_output
    //    - 计时 front Prop
    //    - 拷回结果到 hY_pred
    //    - de-normalize 后计算 test MSE

    // 7. 清理
    hipFree(d_input); hipFree(d_target);
    hipFree(d_hidden); hipFree(d_output);
    hipFree(d_output_grad); hipFree(d_hidden_grad);
    hipFree(d_w1); hipFree(d_w2); hipFree(d_b1); hipFree(d_b2);
    hipFree(d_w1_grad); hipFree(d_w2_grad); hipFree(d_b1_grad); hipFree(d_b2_grad);

    return 0;
}
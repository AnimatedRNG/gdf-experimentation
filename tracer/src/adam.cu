#include "helper_math.h"
#include "cuda_matmul.hpp"

template <typename T>
__global__
void adam(size_t num_elements,
          T* params,
          const T* gradient,
          T* exp_avg,
          T* exp_avg_sq,
          const float lr,
          const float beta_1,
          const float beta_2,
          const float weight_decay,
          const float eps,
          const int iteration) {
          
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index >= num_elements) {
        return;
    }
    
    float grad = gradient[index] + weight_decay * params[index];
    exp_avg[index] = exp_avg[index] * beta_1 + (1.0f - beta_1) * grad;
    exp_avg_sq[index] = exp_avg_sq[index] * beta_2 + (1.0f - beta_2) * grad * grad;
    
    float denom = sqrtf(exp_avg_sq[index]) + eps;
    
    float bias_correction_1 = 1.0f / (1.0f - powf(beta_1, (float) iteration));
    float bias_correction_2 = 1.0f / (1.0f - powf(beta_2, (float) iteration));
    
    float adapted_learning_rate = lr * bias_correction_1 / sqrtf(bias_correction_2);
    params[index] = params[index] - adapted_learning_rate * exp_avg[index] / denom;
}

template <typename T, size_t N>
class AdamOptimizer {
  public:
  
    explicit AdamOptimizer(T* params_device,
                           T* gradient_device,
                           cuda_array<T, N>* params_host,
                           cuda_array<T, N>* gradient_host,
                           T zero_value,
                           const float& lr = 1e-3f,
                           const float& beta_1 = 0.9f,
                           const float& beta_2 = 0.99f,
                           const float& weight_decay = 0.0f,
                           const float& eps = 1e-8f) :
        params_device_(params_device),
        gradient_device_(gradient_device),
        params_host_(params_host),
        gradient_host_(gradient_host),
        lr_(lr), beta_1_(beta_1), beta_2_(beta_2),
        weight_decay_(weight_decay), eps_(eps),
        iterations(0),
        num_elements(params_host->num_elements) {
        
        for (int i = 0; i < N; i++) {
            assert(params_host->shape[i] == gradient_host->shape[i]);
        }
        
        std::vector<T> zeros(params_host->num_elements, zero_value);
        
        size_t buf_size = params_host->num_elements * sizeof(T);
        cudaMalloc(&exp_avg_device_, buf_size);
        cudaMalloc(&exp_avg_sq_device_, buf_size);
        cudaMemcpy(exp_avg_device_, &(zeros[0]), buf_size, cudaMemcpyHostToDevice);
        cudaMemcpy(exp_avg_sq_device_, &(zeros[0]), buf_size, cudaMemcpyHostToDevice);
    }
    
    ~AdamOptimizer() {
        cudaFree(exp_avg_device_);
        cudaFree(exp_avg_sq_device_);
    }
    
    void step() {
        size_t block_size = 64;
        
        size_t grid_size = (size_t) ceil((float) num_elements /
                                         (float) block_size);
                                         
        adam<T> <<< grid_size, block_size >>> (num_elements,
                                               params_device_,
                                               gradient_device_,
                                               exp_avg_device_,
                                               exp_avg_sq_device_,
                                               
                                               lr_,
                                               beta_1_,
                                               beta_2_,
                                               weight_decay_,
                                               eps_,
                                               ++iterations);
    }
    
  private:
    float lr_;
    float beta_1_;
    float beta_2_;
    float weight_decay_;
    float eps_;
    int iterations;
    
    size_t num_elements;
    
    T* params_device_;
    T* gradient_device_;
    
    cuda_array<T, N>* params_host_;
    cuda_array<T, N>* gradient_host_;
    
    T* exp_avg_device_;
    T* exp_avg_sq_device_;
};

int main() {
    size_t dims[2] = {5, 2};
    size_t* dims_device;
    
    float expected_params_m[2][5] = {
        {0.9990, 1.0010, 0.9990, 0.9990, 1.0010},
        {1.0010, 0.9990, 0.9990, 0.9990, 1.0010}
    };
    float initial_params_m[2][5] = {
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f}
    };
    
    float initial_grad_m[2][5] = {
        {0.02574085, -0.26188788, 0.5158403, 0.5158403, -10.2624},
        {-0.26188788, 0.02574085, 0.5158403, 0.5158403, -10.2624}
    };
    
    cuda_array<float, 2> initial_params;
    assign(&initial_params, (float*)initial_params_m, dims);
    
    cuda_array<float, 2> initial_grad;
    assign(&initial_grad, (float*)initial_grad_m, dims);
    
    float* initial_params_device = to_device<float, 2>(&initial_params,
                                   &dims_device);
    float* initial_grad_device = to_device<float, 2>(&initial_grad, &dims_device);
    
    AdamOptimizer<float, 2> adam(initial_params_device, initial_grad_device,
                                 &initial_params, &initial_grad,
                                 0.0f);
    adam.step();
    adam.step();
    
    to_host(initial_params_device, &initial_params);
    
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 5; j++) {
            std::cout << index(&initial_params, j, i) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
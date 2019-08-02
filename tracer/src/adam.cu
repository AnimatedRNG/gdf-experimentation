#include "helper_math.h"
#include "cuda_matmul.hpp"
#include "adam.hpp"

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
    
    to_host(initial_params_device, &initial_params);
    
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 5; j++) {
            std::cout << index(&initial_params, j, i) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    }
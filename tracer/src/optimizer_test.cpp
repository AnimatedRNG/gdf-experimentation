#include <iostream>
#include <chrono>
#include <math.h>

#include "HalideBuffer.h"
#include "halide_image_io.h"
#include "HalideRuntime.h"
#include "HalideRuntimeCuda.h"

#include "sdf_gen.h"
#include "optimizer_gen.h"
#include "optimizer.hpp"
#include "buffer_utils.hpp"

using namespace Halide::Runtime;
using namespace Halide::Tools;

int main() {
    float expected_params_m[2][5] = {{0.9990, 1.0010, 0.9990, 0.9990, 1.0010}};
    Buffer<float> expected_params(expected_params_m);

    float initial_params_m[2][5] = {
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f}
    };
    Buffer<float> initial_params(initial_params_m);

    float initial_grad_m[2][5] = {
        {0.02574085, -0.26188788, 0.5158403, 0.5158403, -10.2624},
        {-0.26188788, 0.02574085, -10.2624, 0.5158403, 0.5158403}
    };
    Buffer<float> initial_grad(initial_grad_m);

    initial_params.set_host_dirty(true);
    initial_params.set_device_dirty(false);
    initial_params.copy_to_device(halide_cuda_device_interface());

    initial_grad.set_host_dirty();
    initial_grad.copy_to_device(halide_cuda_device_interface());

    ADAM adam(initial_params, initial_grad);

    auto start = std::chrono::steady_clock::now();

    adam.step();

    auto end = std::chrono::steady_clock::now();

    auto diff = end - start;

    std::cout
            << "adam step took "
            << std::chrono::duration <float, std::milli> (diff).count()
            << " ms"
            << std::endl;

    initial_params.copy_to_host();

    std::cout << "final parameters are " << std::endl;
    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 5; i++) {
            std::cout << initial_params(i, j) << " ";
            //assert(std::abs(initial_params(i, 0) - expected_params(i)) < 1e-5f);
        }
        std::cout << std::endl;
    }
}

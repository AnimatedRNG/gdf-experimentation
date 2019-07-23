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

using namespace Halide::Runtime;
using namespace Halide::Tools;

int main() {
    float expected_params_m[] = {0.9990, 1.0010, 0.9990, 0.9990, 1.0010};
    Buffer<float> expected_params(expected_params_m);

    float initial_params_m[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    Buffer<float> initial_params(initial_params_m);

    float initial_grad_m[] = {0.02574085, -0.26188788, 0.5158403, 0.5158403, -10.2624};
    Buffer<float> initial_grad(initial_grad_m);

    float exp_avg_m[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    Buffer<float> exp_avg(exp_avg_m);

    float exp_avg_sq_m[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    Buffer<float> exp_avg_sq(exp_avg_sq_m);

    Buffer<float> exp_avg_out(5);
    Buffer<float> exp_avg_sq_out(5);

    Buffer<float> final_parameters(5);

    initial_params.set_host_dirty();
    initial_params.copy_to_device(halide_cuda_device_interface());

    initial_grad.set_host_dirty();
    initial_grad.copy_to_device(halide_cuda_device_interface());

    exp_avg.set_host_dirty();
    exp_avg.copy_to_device(halide_cuda_device_interface());

    exp_avg_sq.set_host_dirty();
    exp_avg_sq.copy_to_device(halide_cuda_device_interface());

    auto start = std::chrono::steady_clock::now();
    optimizer_gen(initial_params, initial_grad,
                  1e-3f, 0.9f, 0.99f, 0.0f, 1e-8f,
                  1,
                  exp_avg, exp_avg_sq,
                  exp_avg_out, exp_avg_sq_out,
                  final_parameters);
    auto end = std::chrono::steady_clock::now();

    auto diff = end - start;

    std::cout
            << "adam step took "
            << std::chrono::duration <float, std::milli> (diff).count()
            << " ms"
            << std::endl << std::endl;

    exp_avg_out.copy_to_host();
    exp_avg_sq_out.copy_to_host();

    final_parameters.copy_to_host();

    //std::cout << "final parameters are ";
    for (int i = 0; i < 5; i++) {
        //std::cout << final_parameters(i) << " ";
        assert(std::abs(final_parameters(i) - expected_params(i)) < 1e-5f);
    }
}

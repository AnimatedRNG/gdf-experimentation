#include <iostream>
#include <chrono>
#include <math.h>

#include "HalideBuffer.h"
#include "halide_image_io.h"
#include "HalideRuntime.h"
#include "HalideRuntimeCuda.h"

#include "sdf_gen.h"
#include "fmm_gen.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

int main(int argc, char** argv) {
    const int32_t n_matrix[3] = {
        128, 128, 128
    };

    Buffer<float> sdf(n_matrix[0], n_matrix[1], n_matrix[2]);
    Buffer<float> p0(3);
    Buffer<float> p1(3);

    sdf.set_host_dirty();
    sdf.copy_to_device(halide_cuda_device_interface());

    p0.set_host_dirty();
    p1.copy_to_device(halide_cuda_device_interface());

    auto start = std::chrono::steady_clock::now();
    sdf_gen(n_matrix[0], n_matrix[1], n_matrix[2], sdf, p0, p1);

    auto end = std::chrono::steady_clock::now();
}

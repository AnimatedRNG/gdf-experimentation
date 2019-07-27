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
        32, 32, 32
    };

    Buffer<float> sdf(n_matrix[0], n_matrix[1], n_matrix[2]);
    Buffer<float> sdf_output(n_matrix[0], n_matrix[1], n_matrix[2]);
    Buffer<float> p0(3);
    Buffer<float> p1(3);

    sdf.set_host_dirty();
    sdf.copy_to_device(halide_cuda_device_interface());

    p0.set_host_dirty();
    p1.copy_to_device(halide_cuda_device_interface());

    sdf_output.set_host_dirty();
    sdf_output.copy_to_device(halide_cuda_device_interface());

    auto start = std::chrono::steady_clock::now();
    sdf_gen(n_matrix[0], n_matrix[1], n_matrix[2], sdf, p0, p1);

    auto end = std::chrono::steady_clock::now();

    fmm_gen(sdf, p0, p1, n_matrix[0], n_matrix[1], n_matrix[2], sdf_output);

    sdf_output.copy_to_host();

    for (int i = 0; i < n_matrix[0]; i++) {
        for (int j = 0; j < n_matrix[1]; j++) {
            printf("%.2f\t", sdf_output(i, j, n_matrix[2] / 2));
        }
        std::cout << std::endl;
    }

    Buffer<float> n_64 = sdf_output.sliced(2, n_matrix[2] / 2);
    convert_and_save_image(n_64, "n_" + std::to_string(n_matrix[2] / 2) + ".png");

    Buffer<float> n_orig_64 = sdf.sliced(2, n_matrix[2] / 2);
    convert_and_save_image(n_orig_64, "orig_n_" + std::to_string(n_matrix[2] / 2) + ".png");
}

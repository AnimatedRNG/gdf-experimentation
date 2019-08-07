#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>

#include "fmm_gen.h"

#include "HalideBuffer.h"
#include "HalideRuntime.h"
#include "HalideRuntimeCuda.h"
#include "math.h"
#include "helper_math.h"
#include "cuda_matmul.hpp"

template <typename T, size_t N>
__host__ inline cuda_array<T, N>* from_buffer(Halide::Runtime::Buffer<T>& buf) {
    cuda_array<T, N>* output = new cuda_array<T, N>;
    int product = 1;

    assert(buf.dimensions() == N);

    for (int i = 0; i < buf.dimensions(); i++) {
        output->shape[i] = buf.dim(i).extent();
        output->stride[i] = buf.dim(i).stride();
        product *= buf.dim(i).extent();
    }

    output->num_elements = product;
    output->data = buf.data();

    return output;
}

__host__ void write_img(const char* filename, cuda_array<float, 3>* arr) {
    unsigned char* arr_bytes = (unsigned char*) malloc(arr->num_elements * sizeof(
                                   float));
    int n = 0;

    for (int k = 0; k < arr->shape[2]; k++) {
        for (int j = 0; j < arr->shape[1]; j++) {
            for (int i = arr->shape[0] - 1; i >= 0; i--) {
                float p = index(arr, i, j, k);
                //std::cout << p << std::endl;
                arr_bytes[n++] = ((p < 0 ? 0 : (p > 1.0 ? 255 : p)) * 255.0);
            }
        }
    }

    stbi_write_bmp(filename, arr->shape[1], arr->shape[2], 3, arr_bytes);
}

__host__ void write_sdf(const std::string& filename,
                        cuda_array<float, 3>* arr,
                        cuda_array<float, 1>* p0,
                        cuda_array<float, 1>* p1) {
    std::ofstream outfile(filename);

    if (!outfile.is_open()) {
        throw std::runtime_error("unable to open " + filename);
    }

    outfile << arr->shape[0] << " " << arr->shape[1] << " " << arr->shape[2] <<
            std::endl;

    outfile << index(p0, 0) << " " << index(p0, 1) << " " << index(p0,
            2) << std::endl;

    // our SDFs can have rectangular grid cells
    // which aren't representable in the .sdf format
    float dx = (index(p1, 0) - index(p0, 0)) / arr->shape[0];

    outfile << dx << std::endl;

    for (int z = 0; z < arr->shape[2]; z++) {
        for (int y = 0; y < arr->shape[1]; y++) {
            for (int x = 0; x < arr->shape[0]; x++) {
                outfile << std::setw(11)
                        << std::setprecision(5)
                        << index(arr, x, y, z)
                        << std::endl;
            }
        }
    }

    outfile.close();
}

void call_fmm(cuda_array<float, 3>* sdf,
              cuda_array<float, 1>* p0_,
              cuda_array<float, 1>* p1_) {
    std::vector<int> sizes;
    for (auto size : sdf->shape) {
        sizes.push_back(size);
    }

    auto interface = halide_cuda_device_interface();

    Buffer<float> p0(p0_->data, {3});
    Buffer<float> p1(p1_->data, {3});

    Buffer<float> sdf_buf(sdf->data, sizes);
    sdf_buf.set_host_dirty();
    sdf_buf.copy_to_device(interface);

    Buffer<float> sdf_output(sizes);

    fmm_gen(sdf_buf, p0, p1,
            sdf->shape[0], sdf->shape[1], sdf->shape[2],
            sdf_output);

    sdf_output.copy_to_host();

    sdf_buf.set_host_dirty(false);
    sdf_buf.copy_from(sdf_output);
}

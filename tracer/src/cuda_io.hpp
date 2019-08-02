#pragma once

#include "HalideBuffer.h"
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

#pragma once

#include <iostream>

#include "HalideBuffer.h"
#include "math.h"
#include "helper_math.h"

using namespace Halide::Runtime;

//#define BOUNDS_CHECKING

template<typename T>
inline __host__ __device__ T reflect_edge(T a, T start, T end) {
    return a + max(start - a, 0) - max(a - end + 1, 0);
}

template <typename T, size_t N>
struct cuda_array {
    size_t shape[N];
    size_t stride[N];
    size_t num_elements;
    T* data;
};

template <typename T, size_t N>
__host__ cuda_array<T, N>* create(size_t dims[N]) {
    cuda_array<T, N>* arr = new cuda_array<T, N>;

    size_t product = 1;
    for (int i = 0; i < N; i++) {
        arr->shape[i] = dims[i];
        arr->stride[i] = product;

        product *= dims[i];
    }
    arr->num_elements = product;
    arr->data = new T[product];

    return arr;
}

template <typename T, size_t N>
__host__ __device__ void assign(cuda_array<T, N>* arr, T* ptr, size_t dims[N],
                                const bool& delete_mem = false) {
    size_t product = 1;
    for (int i = 0; i < N; i++) {
        arr->shape[i] = dims[i];
        arr->stride[i] = product;

        product *= dims[i];
    }
    arr->num_elements = product;

    if (delete_mem) {
        delete[] arr->data;
    }

    arr->data = ptr;
}

template <typename T, typename U, size_t N, size_t M>
__host__ __device__ void reinterpret(cuda_array<T, N>* arr,
                                     cuda_array<U, M>* new_arr,
                                     size_t dims[M]) {
    size_t product_t = sizeof(T);
    size_t product_u = sizeof(U);
    for (int i = 0; i < N; i++) {
        product_t *= arr->shape[i];
    }

    for (int i = 0; i < M; i++) {
        product_u *= dims[i];
    }

    assert(product_t == product_u);

    assign(new_arr, (U*)(arr->data), dims, false);
}

template <typename T, size_t N>
__host__ void delete_array(cuda_array<T, N>* array) {
    delete[] array->data;
    delete array;
}

template <typename T>
__host__ __device__ T& index(cuda_array<T, 1>* arr, const int& i) {
#ifdef BOUNDS_CHECKING
    assert(i < arr->shape[0] && i >= 0);
#endif // BOUNDS_CHECKING
    return arr->data[i * arr->stride[0]];
}

template <typename T>
__host__ __device__ T& index(cuda_array<T, 2>* arr,
                             const int& i,
                             const int& j) {
#ifdef BOUNDS_CHECKING
    assert(i < arr->shape[0] && i >= 0);
    assert(j < arr->shape[1] && j >= 0);
#endif // BOUNDS_CHECKING
    return arr->data[i * arr->stride[0] + j * arr->stride[1]];
}

template <typename T>
__host__ __device__ T& index(cuda_array<T, 3>* arr,
                             const int& i,
                             const int& j,
                             const int& k) {
#ifdef BOUNDS_CHECKING
    assert(i < arr->shape[0] && i >= 0);
    assert(j < arr->shape[1] && j >= 0);
    assert(k < arr->shape[2] && k >= 0);
#endif // BOUNDS_CHECKING
    return arr->data[i * arr->stride[0] + j * arr->stride[1] + k * arr->stride[2]];
}

template <typename T>
__host__ __device__ T& index_clamp(cuda_array<T, 3>* arr,
                                   const int& i,
                                   const int& j,
                                   const int& k) {
    return index(arr,
                 clamp(i, 0, arr->shape[0] - 1),
                 clamp(j, 0, arr->shape[1] - 1),
                 clamp(k, 0, arr->shape[2] - 1));
}

template <typename T>
__host__ __device__ T& index_reflect(cuda_array<T, 3>* arr,
                                     const int& i,
                                     const int& j,
                                     const int& k) {
    return index(arr,
                 reflect_edge(i, 0, (int) arr->shape[0] - 1),
                 reflect_edge(j, 0, (int) arr->shape[1] - 1),
                 reflect_edge(k, 0, (int) arr->shape[2] - 1));
}

template <typename T>
__host__ __device__ T& index_debug(cuda_array<T, 3>* arr,
                                   const int& i,
                                   const int& j,
                                   const int& k) {
#ifdef BOUNDS_CHECKING
    assert(i < arr->shape[0] && i >= 0);
    assert(j < arr->shape[1] && j >= 0);
    assert(k < arr->shape[2] && k >= 0);
#endif // BOUNDS_CHECKING
#ifdef DEBUG_GRADIENT
    return arr->data[i * arr->stride[0] + j * arr->stride[1] + k * arr->stride[2]];
#endif
}

template <typename T>
__host__ __device__ T& index_off(cuda_array<T, 3>* arr,
                                 const int& i,
                                 const int& j,
                                 const int& k,
                                 const int& off) {
#ifdef BOUNDS_CHECKING
    assert(i + off < arr->shape[0] && i + off >= 0);
    assert(j + off < arr->shape[1] && j + off >= 0);
    assert(k + off < arr->shape[2] && k + off >= 0);
#endif // BOUNDS_CHECKING
    return arr->data[(off + i) * arr->stride[0] + (off + j) * arr->stride[1] +
                               (off + k) * arr->stride[2]];
}

template <typename T>
__host__ __device__ T& index(cuda_array<T, 4>* arr,
                             const int& i,
                             const int& j,
                             const int& k,
                             const int& l) {
#ifdef BOUNDS_CHECKING
    assert(i < arr->shape[0] && i >= 0);
    assert(j < arr->shape[1] && j >= 0);
    assert(k < arr->shape[2] && k >= 0);
    assert(l < arr->shape[3] && l >= 0);
#endif //BOUNDS_CHECKING
    return arr->data[i * arr->stride[0] + j * arr->stride[1] + k * arr->stride[2]
                       + l * arr->stride[3]];
}

template <typename T>
__host__ __device__ T& index(cuda_array<T, 5>* arr,
                             const int& a0,
                             const int& a1,
                             const int& a2,
                             const int& a3,
                             const int& a4) {
#ifdef BOUNDS_CHECKING
    assert(a0 < arr->shape[0] && a0 >= 0);
    assert(a1 < arr->shape[1] && a1 >= 0);
    assert(a2 < arr->shape[2] && a2 >= 0);
    assert(a3 < arr->shape[3] && a3 >= 0);
    assert(a4 < arr->shape[4] && a4 >= 0);
#endif //BOUNDS_CHECKING
    return arr->data[
            a0 * arr->stride[0] +
            a1 * arr->stride[1] +
            a2 * arr->stride[2] +
            a3 * arr->stride[3] +
            a4 * arr->stride[4]
        ];
}

template <typename T>
__host__ __device__ T& index(cuda_array<T, 6>* arr,
                             const int& a0,
                             const int& a1,
                             const int& a2,
                             const int& a3,
                             const int& a4,
                             const int& a5) {
#ifdef BOUNDS_CHECKING
    assert(a0 < arr->shape[0] && a0 >= 0);
    assert(a1 < arr->shape[1] && a1 >= 0);
    assert(a2 < arr->shape[2] && a2 >= 0);
    assert(a3 < arr->shape[3] && a3 >= 0);
    assert(a4 < arr->shape[4] && a4 >= 0);
    assert(a5 < arr->shape[4] && a5 >= 0);
#endif //BOUNDS_CHECKING
    return arr->data[
            a0 * arr->stride[0] +
            a1 * arr->stride[1] +
            a2 * arr->stride[2] +
            a3 * arr->stride[3] +
            a4 * arr->stride[4] +
            a5 * arr->stride[5]
        ];
}

inline float example_sphere(float x, float y, float z) {
    float xi = x;
    float yi = y;
    float zi = z;
    return sqrtf(xi * xi + yi * yi + zi * zi) - 3.0f;
}

inline float example_box(float x, float y, float z) {
    float xi = x;
    float yi = y;
    float zi = z;
    return sqrtf(xi * xi + yi * yi + zi * zi) - 3.0f;
}

/*template <typename T>
__device__ __host__ void fill(cuda_array<T, 2>* arr, const T& val) {
    for (int j = 0; j < arr->shape[1]; j++) {
        for (int i = 0; i < arr->shape[0]; i++) {
            index(arr, i, j) = val;
        }
    }
}

template <typename T>
__device__ __host__ void fill(cuda_array<T, 3>* arr, const T& val) {
    for (int k = 0; k < arr->shape[2]; k++) {
        for (int j = 0; j < arr->shape[1]; j++) {
            for (int i = 0; i < arr->shape[0]; i++) {
                index(arr, i, j, k) = val;
            }
        }
    }
}*/

template <typename T, size_t N>
__device__ __host__ void fill(cuda_array<T, N>* arr, const T& val) {
    for (int i = 0; i < arr->num_elements; i++) {
        arr->data[i] = val;
    }
}

template <typename T>
__host__ void gen_sdf(std::function<float(float, float, float)> func,
                      float p0_x, float p0_y, float p0_z,
                      float p1_x, float p1_y, float p1_z,
                      cuda_array<T, 3>* arr) {
    for (int k = 0; k < arr->shape[2]; k++) {
        for (int j = 0; j < arr->shape[1]; j++) {
            for (int i = 0; i < arr->shape[0]; i++) {
                float x = ((float) i) / ((float) arr->shape[0]) * (p1_x - p0_x) + p0_x;
                float y = ((float) j) / ((float) arr->shape[1]) * (p1_y - p0_y) + p0_y;
                float z = ((float) k) / ((float) arr->shape[2]) * (p1_z - p0_z) + p0_z;

                index(arr, i, j, k) = func(x, y, z);
            }
        }
    }
}

template <typename T>
inline __host__ void print(cuda_array<T, 2>* arr) {
    for (int i = 0; i < arr->shape[0]; i++) {
        for (int j = 0; j < arr->shape[1]; j++) {
            std::cout << index(arr, i, j) << " ";
        }
        std::cout << std::endl;
    }
}

inline __host__ void print(cuda_array<float3, 3>* arr) {
    for (int i = 0; i < arr->shape[0]; i++) {
        for (int j = 0; j < arr->shape[1]; j++) {
            for (int k = 0; k < arr->shape[2]; k++) {
                if (abs(index(arr, i, j, k).x) <= 1e-5f &&
                    abs(index(arr, i, j, k).y) <= 1e-5f &&
                    abs(index(arr, i, j, k).z) <= 1e-5f) {
                    printf("%.1f %.1f %.1f   \t",
                           index(arr, i, j, k).x,
                           index(arr, i, j, k).y,
                           index(arr, i, j, k).z);
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

template <typename T, size_t N>
__host__ T* to_device(const cuda_array<T, N>* arr, size_t** shape) {
    T* device_ptr;

    std::cout << "copying " << arr->num_elements << " to device" << std::endl;
    cudaMalloc(&device_ptr, arr->num_elements * sizeof(T));
    cudaMemcpy(device_ptr, arr->data, arr->num_elements * sizeof(T),
               cudaMemcpyHostToDevice);

    cudaMalloc(shape, N * sizeof(size_t));
    cudaMemcpy(*shape, arr->shape, N * sizeof(size_t),
               cudaMemcpyHostToDevice);

    return device_ptr;
}

template <typename T, size_t N>
__host__ void zero(cuda_array<T, N>* host, T* device, T zero_val) {
    fill(host, zero_val);
    cudaMemcpy(device, host->data, host->num_elements * sizeof(T),
               cudaMemcpyHostToDevice);
}

template <typename T, size_t N>
__host__ void to_host(T* arr, cuda_array<T, N>* host,
                      bool delete_device_buffer = false) {
    std::cout << "copying " << host->num_elements << " to host" << std::endl;

    int ret = cudaMemcpy(host->data, arr,
                         host->num_elements * sizeof(T),
                         cudaMemcpyDeviceToHost);
    std::cout << "ret " << ret << std::endl;
    if (delete_device_buffer) {
        cudaFree((void*) arr);
    }
}

// from MESA GLU
inline __device__ bool invert_matrix(float* m, float* invOut) {
    float inv[16], det;
    int i;

    inv[0] = m[5]  * m[10] * m[15] -
             m[5]  * m[11] * m[14] -
             m[9]  * m[6]  * m[15] +
             m[9]  * m[7]  * m[14] +
             m[13] * m[6]  * m[11] -
             m[13] * m[7]  * m[10];

    inv[4] = -m[4]  * m[10] * m[15] +
             m[4]  * m[11] * m[14] +
             m[8]  * m[6]  * m[15] -
             m[8]  * m[7]  * m[14] -
             m[12] * m[6]  * m[11] +
             m[12] * m[7]  * m[10];

    inv[8] = m[4]  * m[9] * m[15] -
             m[4]  * m[11] * m[13] -
             m[8]  * m[5] * m[15] +
             m[8]  * m[7] * m[13] +
             m[12] * m[5] * m[11] -
             m[12] * m[7] * m[9];

    inv[12] = -m[4]  * m[9] * m[14] +
              m[4]  * m[10] * m[13] +
              m[8]  * m[5] * m[14] -
              m[8]  * m[6] * m[13] -
              m[12] * m[5] * m[10] +
              m[12] * m[6] * m[9];

    inv[1] = -m[1]  * m[10] * m[15] +
             m[1]  * m[11] * m[14] +
             m[9]  * m[2] * m[15] -
             m[9]  * m[3] * m[14] -
             m[13] * m[2] * m[11] +
             m[13] * m[3] * m[10];

    inv[5] = m[0]  * m[10] * m[15] -
             m[0]  * m[11] * m[14] -
             m[8]  * m[2] * m[15] +
             m[8]  * m[3] * m[14] +
             m[12] * m[2] * m[11] -
             m[12] * m[3] * m[10];

    inv[9] = -m[0]  * m[9] * m[15] +
             m[0]  * m[11] * m[13] +
             m[8]  * m[1] * m[15] -
             m[8]  * m[3] * m[13] -
             m[12] * m[1] * m[11] +
             m[12] * m[3] * m[9];

    inv[13] = m[0]  * m[9] * m[14] -
              m[0]  * m[10] * m[13] -
              m[8]  * m[1] * m[14] +
              m[8]  * m[2] * m[13] +
              m[12] * m[1] * m[10] -
              m[12] * m[2] * m[9];

    inv[2] = m[1]  * m[6] * m[15] -
             m[1]  * m[7] * m[14] -
             m[5]  * m[2] * m[15] +
             m[5]  * m[3] * m[14] +
             m[13] * m[2] * m[7] -
             m[13] * m[3] * m[6];

    inv[6] = -m[0]  * m[6] * m[15] +
             m[0]  * m[7] * m[14] +
             m[4]  * m[2] * m[15] -
             m[4]  * m[3] * m[14] -
             m[12] * m[2] * m[7] +
             m[12] * m[3] * m[6];

    inv[10] = m[0]  * m[5] * m[15] -
              m[0]  * m[7] * m[13] -
              m[4]  * m[1] * m[15] +
              m[4]  * m[3] * m[13] +
              m[12] * m[1] * m[7] -
              m[12] * m[3] * m[5];

    inv[14] = -m[0]  * m[5] * m[14] +
              m[0]  * m[6] * m[13] +
              m[4]  * m[1] * m[14] -
              m[4]  * m[2] * m[13] -
              m[12] * m[1] * m[6] +
              m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] +
             m[1] * m[7] * m[10] +
             m[5] * m[2] * m[11] -
             m[5] * m[3] * m[10] -
             m[9] * m[2] * m[7] +
             m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] -
             m[0] * m[7] * m[10] -
             m[4] * m[2] * m[11] +
             m[4] * m[3] * m[10] +
             m[8] * m[2] * m[7] -
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] +
              m[0] * m[7] * m[9] +
              m[4] * m[1] * m[11] -
              m[4] * m[3] * m[9] -
              m[8] * m[1] * m[7] +
              m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] -
              m[0] * m[6] * m[9] -
              m[4] * m[1] * m[10] +
              m[4] * m[2] * m[9] +
              m[8] * m[1] * m[6] -
              m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (det == 0)
        return false;

    det = 1.0 / det;

    for (i = 0; i < 16; i++)
        invOut[i] = inv[i] * det;

    return true;
}

template <typename T>
__host__ __device__ void mat4(cuda_array<T, 2>* m, T data[4][4]) {
    size_t dims[2] = {4, 4};
    assign<float, 2>(m, (T*) data, dims);
}

__host__ __device__ inline float norm_sq(float3 v) {
    return dot(v, v);
}

template <typename T, size_t N>
__host__ __device__ void matmul_sq(cuda_array<T, N>* a, cuda_array<T, N>* b,
                                   cuda_array<T, N>* c) {
    for (int i = 0; i < a->shape[0]; i++) {
        for (int j = 0; j < a->shape[0]; j++) {
            index(c, i, j) = 0.0f;
            for (int k = 0; k < a->shape[0]; k++) {
                index(c, i, j) += index(a, i, k) * index(b, k, j);
            }
        }
    }
}

template <typename T>
__host__ __device__ float3 matvec(cuda_array<T, 2>* a, const float3& b) {
    return make_float3(index(a, 0, 0) * b.x + index(a, 0, 1) * b.y + index(a, 0,
                       2) * b.z,
                       index(a, 1, 0) * b.x + index(a, 1, 1) * b.y + index(a, 1, 2) * b.z,
                       index(a, 2, 0) * b.x + index(a, 2, 1) * b.y + index(a, 2, 2) * b.z);
}

template <typename T>
__host__ __device__ float4 matvec(cuda_array<T, 2>* a, const float4& b) {
    float i0 = index(a, 0, 0) * b.x + index(a, 0, 1) * b.y + index(a, 0,
               2) * b.z + index(a, 0, 3) * b.w;
    float i1 = index(a, 1, 0) * b.x + index(a, 1, 1) * b.y + index(a, 1,
               2) * b.z + index(a, 1, 3) * b.w;
    float i2 = index(a, 2, 0) * b.x + index(a, 2, 1) * b.y + index(a, 2,
               2) * b.z + index(a, 2, 3) * b.w;
    float i3 = index(a, 3, 0) * b.x + index(a, 3, 1) * b.y + index(a, 3,
               2) * b.z + index(a, 3, 3) * b.w;
    return make_float4(i0, i1, i2, i3);
}

template <typename T>
__host__ __device__ float3 apply_affine(cuda_array<T, 2>* a, const float3& b) {
    float4 l2 = make_float4(b.x, b.y, b.z, 1.0f);

    float4 homogeneous = matvec(a, l2);
    float4 affine_divide = homogeneous / homogeneous.w;

    return make_float3(affine_divide.x, affine_divide.y, affine_divide.z);
}

// based on GLKMatrix4MakeLookAt from GLKit
inline __host__ __device__ void look_at(cuda_array<float, 2>* out,
                                        float3 eye,
                                 float3 center,
                                 float3 up) {
    const float3 n = normalize(center - eye);
    const float3 u = normalize(cross(up, n));
    const float3 v = cross(n, u);

    index(out, 0, 0) = u.x;
    index(out, 1, 0) = v.x;
    index(out, 2, 0) = n.x;
    index(out, 3, 0) = 0.0f;

    index(out, 0, 1) = u.y;
    index(out, 1, 1) = v.y;
    index(out, 2, 1) = n.y;
    index(out, 3, 1) = 0.0f;

    index(out, 0, 2) = u.z;
    index(out, 1, 2) = v.z;
    index(out, 2, 2) = n.z;
    index(out, 3, 2) = 0.0f;

    index(out, 0, 3) = dot(-1.0f * u, eye);
    index(out, 1, 3) = dot(-1.0f * v, eye);
    index(out, 2, 3) = dot(-1.0f * n, eye);
    index(out, 3, 3) = 1.0f;
}

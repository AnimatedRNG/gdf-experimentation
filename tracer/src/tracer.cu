#include <iostream>
#include <functional>
#include "math.h"

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
__host__ __device__ void assign(cuda_array<T, N>* arr, T* ptr, size_t dims[N]) {
    size_t product = 1;
    for (int i = 0; i < N; i++) {
        arr->shape[i] = dims[i];
        arr->stride[i] = product;
        
        product *= dims[i];
    }
    arr->num_elements = product;
    arr->data = ptr;
}

template <typename T, size_t N>
__host__ void delete_array(cuda_array<T, N>* array) {
    delete[] array->data;
    delete array;
}

template <typename T, size_t N>
__host__ __device__ T& index(cuda_array<T, N>* arr, const int& i) {
    return arr->data[i * arr->stride[0]];
}

template <typename T, size_t N>
__host__ __device__ T& index(cuda_array<T, N>* arr,
                             const int& i,
                             const int& j) {
    return arr->data[i * arr->stride[0] + j * arr->stride[1]];
}

template <typename T, size_t N>
__host__ __device__ T& index(cuda_array<T, N>* arr,
                             const int& i,
                             const int& j,
                             const int& k) {
    return arr->data[i * arr->stride[0] + j * arr->stride[1] + k * arr->stride[1]];
}

float example_sphere(float x, float y, float z) {
    return sqrtf(x * x + y * y + z * z) - 3.0f;
}

template <typename T>
__host__ void fill(cuda_array<T, 2>* arr, const T& val) {
    for (int j = 0; j < arr->shape[1]; j++) {
        for (int i = 0; i < arr->shape[0]; i++) {
            index(arr, i, j) = val;
        }
    }
}

template <typename T>
__host__ void fill(cuda_array<T, 3>* arr, const T& val) {
    for (int k = 0; k < arr->shape[2]; k++) {
        for (int j = 0; j < arr->shape[1]; j++) {
            for (int i = 0; i < arr->shape[0]; i++) {
                index(arr, i, j, k) = val;
            }
        }
    }
}

template <typename T>
__host__ void gen_sdf(std::function<float(float, float, float)> func,
                      cuda_array<T, 3>* arr) {
    for (int k = 0; k < arr->shape[2]; k++) {
        for (int j = 0; j < arr->shape[1]; j++) {
            for (int i = 0; i < arr->shape[0]; i++) {
                index(arr, i, j, k) = func(i, j, k);
            }
        }
    }
}

template <typename T>
__host__ void print(cuda_array<T, 2>* arr) {
    for (int i = 0; i < arr->shape[0]; i++) {
        for (int j = 0; j < arr->shape[1]; j++) {
            std::cout << index(arr, i, j) << " ";
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
__host__ void to_host(T* arr, cuda_array<T, N>* host) {
    std::cout << "copying " << host->num_elements << " to host" << std::endl;
    
    int ret = cudaMemcpy(host->data, arr,
                         host->num_elements * sizeof(T),
                         cudaMemcpyDeviceToHost);
    std::cout << "ret " << ret << std::endl;
    cudaFree((void*) arr);
}

__global__
void render(float* projection_matrix_, size_t shape[2]) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    cuda_array<float, 2> projection_matrix;
    assign<float, 2>(&projection_matrix, projection_matrix_, shape);
    
    if (i < 16) {
        index(&projection_matrix, i % 4, i / 4) = 10.0f;
    }
}

void trace() {
    size_t dims[2] = {4, 4};
    cuda_array<float, 2>* host_matrix = create<float, 2>(dims);
    fill(host_matrix, 2.0f);
    //gen_sdf(example_sphere, host_matrix);
    
    size_t* shape;
    float* device_matrix = to_device<float, 2>(host_matrix, &shape);
    
    render <<< 64, 64>>> (device_matrix, shape);
    
    to_host<float, 2>(device_matrix, host_matrix);
    
    print<float>(host_matrix);
    
    delete_array(host_matrix);
}

int main() {
    trace();
}
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
void render(float* projection_matrix_,
            float* view_matrix_,
            float* transform_matrix_,
            float* sdf_, size_t sdf_shape[3],
            float* p0_,
            float* p1_,
            float* target_,
            size_t width,
            size_t height,
            
            float* loss_,
            float* forward_,
            float* dLossdSDF_,
            float* dLossdTransform_) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    size_t mat4_size[2] = {4, 4};
    size_t vec3_size[1] = {3};
    size_t image_size[2] = {width, height};
    
    cuda_array<float, 2> projection_matrix;
    assign<float, 2>(&projection_matrix,
                     projection_matrix_,
                     mat4_size);
                     
    cuda_array<float, 2> view_matrix;
    assign<float, 2>(&view_matrix,
                     view_matrix_,
                     mat4_size);
                     
    cuda_array<float, 2> transform_matrix;
    assign<float, 2>(&transform_matrix,
                     transform_matrix_,
                     mat4_size);
                     
    cuda_array<float, 1> p0;
    assign<float, 1>(&p0,
                     p0_,
                     vec3_size);
                     
    cuda_array<float, 1> p1;
    assign<float, 1>(&p1,
                     p1_,
                     vec3_size);
                     
    cuda_array<float, 2> target;
    assign<float, 2>(&target,
                     target_,
                     image_size);
                     
    if (i < 16) {
        index(&projection_matrix, i % 4, i / 4) += 10.0f;
    }
}

void trace() {
    size_t n_matrix[3] = {
        64, 64, 64
    };
    
    const float projection_matrix[4][4] = {
        {0.75, 0.0, 0.0,  0.0},
        {0.0,  1.0, 0.0,  0.0},
        {0.0,  0.0, 1.0002,  1.0},
        {0.0,  0.0, -0.2,  0.0}
    };
    
    const float view_matrix[4][4] = {
        {-0.9825, 0.1422, 0.1206, 0.0},
        {0.0, 0.6469, -0.7626, 0.0},
        {0.1865, 0.7492, 0.6356, 0.0},
        {-0.3166, 1.1503, 8.8977, 1.0}
    };
    
    const float transform_matrix[4][4] = {
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0},
        {0.0f, 0.0f, 0.0f, 1.0}
    };
    
    const float p0_matrix[3] = {
        -4.0f, -4.0f, -4.0f
        };
        
    const float p1_matrix[3] = {
        4.0f, 4.0f, 4.0f
    };
    
    const int width = 100;
    const int height = 100;
    
    size_t mat4_dims[2] = {4, 4};
    size_t vec3_dims[1] = {3};
    size_t img_dims[3] = {width, height, 3};
    size_t single_dim[1] = {1};
    
    size_t* mat4_dims_device;
    size_t* vec3_dims_device;
    size_t* n_matrix_device;
    size_t* img_dims_device;
    size_t* single_dim_device;
    
    cuda_array<float, 2>* projection_host = create<float, 2>(mat4_dims);
    assign(projection_host, (float*) projection_matrix, mat4_dims, true);
    float* projection_device = to_device<float, 2>(projection_host,
                               &mat4_dims_device);
                               
    cuda_array<float, 2>* view_host = create<float, 2>(mat4_dims);
    assign(view_host, (float*) view_matrix, mat4_dims, true);
    float* view_device = to_device<float, 2>(view_host, &mat4_dims_device);
    
    cuda_array<float, 2>* transform_host = create<float, 2>(mat4_dims);
    assign(transform_host, (float*) transform_matrix, mat4_dims, true);
    float* transform_device = to_device<float, 2>(transform_host,
                              &mat4_dims_device);
                              
    cuda_array<float, 3>* sdf_host = create<float, 3>(n_matrix);
    gen_sdf<float>(example_sphere, sdf_host);
    float* sdf_device = to_device<float, 3>(sdf_host, &n_matrix_device);
    
    cuda_array<float, 1>* p0_host = create<float, 1>(vec3_dims);
    assign(p0_host, (float*) p0_matrix, vec3_dims, true);
    float* p0_device = to_device<float, 1>(p0_host, &vec3_dims_device);
    
    cuda_array<float, 1>* p1_host = create<float, 1>(vec3_dims);
    assign(p1_host, (float*) p1_matrix, vec3_dims, true);
    float* p1_device = to_device<float, 1>(p1_host, &vec3_dims_device);
    
    cuda_array<float, 3>* target_host = create<float, 3>(img_dims);
    float* target_device = to_device<float, 3>(target_host, &img_dims_device);
    
    cuda_array<float, 1>* loss_host = create<float, 1>(single_dim);
    float* loss_device = to_device<float, 1>(loss_host, &single_dim_device);
    
    cuda_array<float, 3>* forward_host = create<float, 3>(img_dims);
    float* forward_device = to_device<float, 3>(forward_host, &img_dims_device);
    
    cuda_array<float, 3>* dloss_dsdf_host = create<float, 3>(n_matrix);
    float* dloss_dsdf_device = to_device<float, 3>(dloss_dsdf_host,
                               &n_matrix_device);
                               
    cuda_array<float, 2>* dloss_dtransform_host = create<float, 2>(mat4_dims);
    float* dloss_dtransform_device = to_device<float, 2>(dloss_dtransform_host,
                                     &mat4_dims_device);
                                     
    render <<< 64, 64>>> (projection_device,
                          view_device,
                          transform_device,
                          
                          sdf_device, n_matrix_device,
                          p0_device, p1_device,
                          
                          target_device,
                          
                          width, height,
                          
                          // outputs
                          loss_device,
                          forward_device,
                          dloss_dsdf_device,
                          dloss_dtransform_device
                         );
                         
    to_host<float, 2>(projection_device, projection_host);
    
    print<float>(projection_host);
}

int main() {
    trace();
}
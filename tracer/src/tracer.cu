#include <iostream>
#include <functional>
#include "math.h"
#include "helper_math.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "cuda_matmul.hpp"

__device__ void projection_gen(int x,
                               int y,
                               cuda_array<float, 2>* projection,
                               cuda_array<float, 2>* view,
                               size_t width,
                               size_t height,
                               float3& ray_pos,
                               float3& ray_vec,
                               float3& origin,
                               float near = 0.1f
                              ) {
    float2 ss_norm = make_float2(((float) x) / ((float) width),
                                 ((float) y) / ((float) height));
    float2 clip_space = ss_norm * 2.0f - 1.0f;
    
    cuda_array<float, 2> proj_view;
    float proj_view_m[4][4];
    mat4<float>(&proj_view, proj_view_m);
    
    cuda_array<float, 2> viewproj_inv;
    float viewproj_inv_m[4][4];
    mat4<float>(&viewproj_inv, viewproj_inv_m);
    
    cuda_array<float, 2> view_inv;
    float view_inv_m[4][4];
    mat4<float>(&view_inv, view_inv_m);
    
    matmul_sq<float, 2>(projection, view, &proj_view);
    invert_matrix(proj_view.data, viewproj_inv.data);
    invert_matrix(view->data, view_inv.data);
    
    
    float4 homogeneous = make_float4(
                             index(&viewproj_inv, 0, 0) * clip_space.x
                             + index(&viewproj_inv, 0, 1) * clip_space.y,
                             index(&viewproj_inv, 1, 0) * clip_space.x
                             + index(&viewproj_inv, 1, 1) * clip_space.y,
                             0.0f,
                             0.0f
                         );
                         
    origin =
        make_float3(
            index(&view_inv, 0, 3), index(&view_inv, 1, 3), index(&view_inv, 2, 3)
        );
        
    float4 h_div = (homogeneous / homogeneous.w);
    float3 projected = make_float3(h_div.x, h_div.y, h_div.z) - origin;
    
    ray_vec = normalize(projected);
    ray_pos = origin + ray_vec * near;
}

__device__ float3 forward_pass(int x,
                               int y,
                               cuda_array<float, 3>* sdf,
                               cuda_array<float, 1>* p0,
                               cuda_array<float, 1>* p1,
                               size_t sdf_shape[3],
                               
                               
                               cuda_array<float, 2>* projection,
                               cuda_array<float, 2>* view,
                               cuda_array<float, 2>* ray_transform,
                               
                               int width,
                               int height
                              ) {
    float3 ray_pos = make_float3(0.0f, 0.0f, 0.0f);
    float3 ray_vec = make_float3(0.0f, 0.0f, 0.0f);
    float3 origin = make_float3(0.0f, 0.0f, 0.0f);
    
    projection_gen(x, y, projection, view, width, height, ray_pos, ray_vec, origin);
    
    return ray_pos;
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
    
    if (i >= width || j >= height) {
        return;
    }
    
    size_t mat4_size[2] = {4, 4};
    size_t vec3_size[1] = {3};
    size_t image_size[3] = {3, width, height};
    
    cuda_array<float, 2> projection;
    assign<float, 2>(&projection,
                     projection_matrix_,
                     mat4_size);
                     
    cuda_array<float, 2> view;
    assign<float, 2>(&view,
                     view_matrix_,
                     mat4_size);
                     
    cuda_array<float, 2> transform;
    assign<float, 2>(&transform,
                     transform_matrix_,
                     mat4_size);
                     
    cuda_array<float, 3> sdf;
    assign<float, 3>(&sdf,
                     sdf_,
                     sdf_shape);
                     
    cuda_array<float, 1> p0;
    assign<float, 1>(&p0,
                     p0_,
                     vec3_size);
                     
    cuda_array<float, 1> p1;
    assign<float, 1>(&p1,
                     p1_,
                     vec3_size);
                     
    cuda_array<float, 3> target;
    assign<float, 3>(&target,
                     target_,
                     image_size);
                     
    cuda_array<float, 3> forward;
    assign<float, 3>(&forward,
                     forward_,
                     image_size);
                     
    float3 c = forward_pass(i, j,
                            &sdf, &p0, &p1, sdf_shape,
                            &projection, &view, &transform,
                            width, height);
                            
    index(&forward, 0, i, j) = (float) i / width;
    index(&forward, 1, i, j) = (float) j / height;
    index(&forward, 2, i, j) = 0.0f;
    //index(&forward, 0, i, j) = c.x;
    //index(&forward, 1, i, j) = c.y;
    //index(&forward, 2, i, j) = c.z;
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
    size_t img_dims[3] = {3, width, height};
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
                                     
    const size_t block_size = 8;
    const size_t grid_size_x = (int)(ceil((float) width / (float) block_size));
    const size_t grid_size_y = (int)(ceil((float) height / (float) block_size));
    
    dim3 blocks(grid_size_x, grid_size_y);
    dim3 threads(block_size, block_size);
    
    render <<< blocks, threads>>> (projection_device,
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
    to_host<float, 3>(forward_device, forward_host);
    
    print<float>(projection_host);
    write_img("forward_cuda.bmp", forward_host);
}

int main() {
    trace();
}
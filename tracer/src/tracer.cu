#include <iostream>
#include <chrono>
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
    float2 clip_space_xy = ss_norm * 2.0f - 1.0f;
    float4 clip_space = make_float4(clip_space_xy.x, clip_space_xy.y, 1.0f, 1.0f);
    
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
    
    float4 homogeneous = matvec<float>(&viewproj_inv, clip_space);
    
    origin =
        make_float3(
            index(&view_inv, 0, 3), index(&view_inv, 1, 3), index(&view_inv, 2, 3)
        );
        
    float4 h_div = (homogeneous / homogeneous.w);
    float3 projected = make_float3(h_div.x, h_div.y, h_div.z) - origin;
    
    ray_vec = normalize(projected);
    ray_pos = origin + ray_vec * near;
}

template <typename T>
__device__ T trilinear(cuda_array < T, 3>* f,
                       float3 p0,
                       float3 p1,
                       int3 sdf_shape,
                       
                       float3 position,
                       T exterior
                      ) {
    float3 grid_space = ((position - p0) / (p1 - p0)) *
                        make_float3(sdf_shape.x, sdf_shape.y, sdf_shape.z);
                        
    if (grid_space.x < 0.0f || grid_space.x < 1.0f
            || grid_space.y < 0.0f || grid_space.y < 1.0f
            || grid_space.z < 0.0f || grid_space.z < 1.0f) {
        return exterior;
    }
    
    int3 lp = make_int3(
                  clamp(int(grid_space.x), 0, sdf_shape.x - 1),
                  clamp(int(grid_space.y), 0, sdf_shape.y - 1),
                  clamp(int(grid_space.z), 0, sdf_shape.z - 1)
              );
    int3 up = make_int3(
                  clamp(int(ceil(grid_space.x)), 0, sdf_shape.x - 1),
                  clamp(int(ceil(grid_space.y)), 0, sdf_shape.y - 1),
                  clamp(int(ceil(grid_space.z)), 0, sdf_shape.z - 1)
              );
              
    float3 alpha = grid_space - make_float3(lp);
    
    T c000 = index(f, lp.x, lp.y, lp.z);
    T c001 = index(f, lp.x, lp.y, up.z);
    T c010 = index(f, lp.x, up.y, lp.z);
    T c011 = index(f, lp.x, up.y, up.z);
    T c100 = index(f, up.x, lp.y, lp.z);
    T c101 = index(f, up.x, lp.y, up.z);
    T c110 = index(f, up.x, up.y, lp.z);
    T c111 = index(f, up.x, up.y, up.z);
    
    T c00 = lerp(c000, c100, alpha.x);
    T c01 = lerp(c001, c101, alpha.x);
    T c10 = lerp(c010, c110, alpha.x);
    T c11 = lerp(c011, c111, alpha.x);
    
    T c0 = lerp(c00, c10, alpha.y);
    T c1 = lerp(c01, c11, alpha.y);
    
    T c = lerp(c0, c1, alpha.z);
    
    return c;
}

__device__ float h(cuda_array<float, 3>* sdf,
                   uint3 sdf_shape,
                   uint3 pos,
                   unsigned int dim) {
    float h_kern[3] = {1.f, 2.f, 1.f};
    
    uint3 c_ = clamp(pos, make_uint3(0), sdf_shape - 1);
    
    switch (dim) {
        case 0:
            return index(sdf, c_.x - 1, c_.y, c_.z) * h_kern[0] +
                   index(sdf, c_.x, c_.y, c_.z) * h_kern[1] +
                   index(sdf, c_.x + 1, c_.y, c_.z) * h_kern[2];
        case 1:
            return index(sdf, c_.x, c_.y - 1, c_.z) * h_kern[0] +
                   index(sdf, c_.x, c_.y, c_.z) * h_kern[1] +
                   index(sdf, c_.x, c_.y + 1, c_.z) * h_kern[2];
        case 2:
            return index(sdf, c_.x, c_.y, c_.z - 1) * h_kern[0] +
                   index(sdf, c_.x, c_.y, c_.z) * h_kern[1] +
                   index(sdf, c_.x, c_.y, c_.z + 1) * h_kern[2];
        default:
            assert(false);
            return -1.0f;
    }
}

__device__ float h_p(cuda_array<float, 3>* sdf,
                     uint3 sdf_shape,
                     uint3 pos,
                     unsigned int dim) {
    float h_kern[2] = {1.f, -1.f};
    
    uint3 c_ = clamp(pos, make_uint3(0), sdf_shape - 1);
    
    switch (dim) {
        case 0:
            return index(sdf, c_.x - 1, c_.y, c_.z) * h_kern[0] +
                   index(sdf, c_.x + 1, c_.y, c_.z) * h_kern[1];
        case 1:
            return index(sdf, c_.x, c_.y - 1, c_.z) * h_kern[0] +
                   index(sdf, c_.x, c_.y + 1, c_.z) * h_kern[1];
        case 2:
            return index(sdf, c_.x, c_.y, c_.z - 1) * h_kern[0] +
                   index(sdf, c_.x, c_.y, c_.z + 1) * h_kern[1];
        default:
            assert(false);
            return -1.0f;
    }
}

__device__ float3 sobel_at(cuda_array<float, 3>* sdf,
                           uint3 sdf_shape,
                           uint3 pos) {
    float h_x = h(sdf, sdf_shape, pos, 0);
    float h_y = h(sdf, sdf_shape, pos, 1);
    float h_z = h(sdf, sdf_shape, pos, 2);
    
    float h_p_x = h(sdf, sdf_shape, pos, 0);
    float h_p_y = h(sdf, sdf_shape, pos, 1);
    float h_p_z = h(sdf, sdf_shape, pos, 2);
    
    
    // TODO: properly handle degenerate case where
    // normal equals zero? or should we just add an offset
    return normalize(make_float3(
                         h_p_x * h_y * h_z,
                         h_p_y * h_z * h_x,
                         h_p_z * h_x * h_y
                     ));
}

__global__
void sobel(
    float* sdf_, size_t sdf_shape[3],
    float* normals_) {
    
    size_t normals_shape[4] = {3, sdf_shape[0], sdf_shape[1], sdf_shape[2]};
    
    cuda_array<float, 3> sdf;
    assign<float, 3>(&sdf,
                     sdf_,
                     sdf_shape);
                     
    cuda_array<float, 4> normals;
    assign<float, 4>(&normals,
                     normals_,
                     normals_shape);
                     
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint j = blockIdx.y * blockDim.y + threadIdx.y;
    uint k = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= sdf_shape[0] || j >= sdf_shape[1] || k >= sdf_shape[2]) {
        return;
    }
    
    float3 sb = sobel_at(&sdf,
                         make_uint3((uint) sdf_shape[0],
                                    (uint) sdf_shape[1],
                                    (uint) sdf_shape[2]),
                         make_uint3(i, j, k));
    index(&normals, 0, i, j, k) = sb.x;
    index(&normals, 1, i, j, k) = sb.y;
    index(&normals, 2, i, j, k) = sb.z;
}

__device__ float3 forward_pass(int x,
                               int y,
                               cuda_array<float, 3>* sdf,
                               float3 p0,
                               float3 p1,
                               uint3 sdf_shape,
                               
                               
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

    float3 pos = ray_pos;
    for (int tr = 0; tr < 50; tr++) {
        pos += trilinear<float>(sdf, p0, p1, make_int3(sdf_shape), pos, 1.0f) * ray_vec;
    }
    
    //return ray_vec;
    return pos;
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

    float3 p0_p = make_float3(index(&p0, 0), index(&p0, 1), index(&p0, 2));
    float3 p1_p = make_float3(index(&p1, 0), index(&p1, 1), index(&p1, 2));
    uint3 sdf_shape_p = make_uint3(sdf_shape[0],
                                   sdf_shape[1],
                                   sdf_shape[2]);
                     
    float3 c = forward_pass(i, j,
                            &sdf, p0_p, p1_p, sdf_shape_p,
                            &projection, &view, &transform,
                            width, height);

    index(&forward, 0, i, j) = c.x;
    index(&forward, 1, i, j) = c.y;
    index(&forward, 2, i, j) = c.z;
}

void trace() {
    size_t n_matrix[3] = {
        64, 64, 64
    };
    
    size_t n_sobel_matrix[4] = {
        3, n_matrix[0], n_matrix[1], n_matrix[2]
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
    
    const int width = 1000;
    const int height = 1000;
    
    size_t mat4_dims[2] = {4, 4};
    size_t vec3_dims[1] = {3};
    size_t img_dims[3] = {3, width, height};
    size_t single_dim[1] = {1};
    
    size_t* mat4_dims_device;
    size_t* vec3_dims_device;
    size_t* n_matrix_device;
    size_t* n_sobel_matrix_device;
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
    gen_sdf<float>(example_sphere,
                   p0_matrix[0], p0_matrix[1], p0_matrix[2],
                   p1_matrix[0], p1_matrix[1], p1_matrix[2],
                   sdf_host);
    float* sdf_device = to_device<float, 3>(sdf_host, &n_matrix_device);
    
    cuda_array<float, 4>* normals_host = create<float, 4>(n_sobel_matrix);
    float* normals_device = to_device<float, 4>(normals_host,
                            &n_sobel_matrix_device);
                            
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
                                     
    const size_t sobel_block_size = 4;
    const size_t sobel_grid_size_x = (int)(ceil((float) n_sobel_matrix[0] /
                                           (float) sobel_block_size));
    const size_t sobel_grid_size_y = (int)(ceil((float) n_sobel_matrix[2] /
                                           (float) sobel_block_size));
    const size_t sobel_grid_size_z = (int)(ceil((float) n_sobel_matrix[3] /
                                           (float) sobel_block_size));
                                           
    dim3 sobel_blocks(sobel_grid_size_x, sobel_grid_size_y, sobel_grid_size_z);
    dim3 sobel_threads(sobel_block_size, sobel_block_size, sobel_block_size);
    
    sobel <<< sobel_blocks, sobel_threads>>> (sdf_device,
            n_matrix_device,
            normals_device);

    cudaThreadSynchronize();
            
    const size_t block_size = 32;
    const size_t grid_size_x = (int)(ceil((float) width / (float) block_size));
    const size_t grid_size_y = (int)(ceil((float) height / (float) block_size));
    
    dim3 blocks(grid_size_x, grid_size_y);
    dim3 threads(block_size, block_size);

    auto start = std::chrono::steady_clock::now();
    
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
    cudaThreadSynchronize();

    auto end = std::chrono::steady_clock::now();

    auto diff = end - start;

    std::cout << "Rendered image in "
              << std::chrono::duration <float, std::milli> (diff).count()
              << " ms"
              << std::endl << std::endl;
                                  
    to_host<float, 2>(projection_device, projection_host);
    to_host<float, 3>(forward_device, forward_host);
    
    print<float>(projection_host);
    write_img("forward_cuda.bmp", forward_host);
}

int main() {
    trace();
}
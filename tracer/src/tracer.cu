#include <iostream>
#include <chrono>
#include <functional>

#include "math.h"
#include "helper_math.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "read_sdf.hpp"

#include "cuda_io.hpp"
#include "cuda_matmul.hpp"
#include "cuda_trilinear.hpp"
#include "adam.hpp"

#define ITERATIONS 900

#define SQUARE(a) (a * a)

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

__device__ float h(cuda_array<float, 3>* sdf,
                   uint3 sdf_shape,
                   uint3 pos,
                   unsigned int dim) {
    float h_kern[3] = {1.f, 2.f, 1.f};
    
    uint3 c_ = clamp(pos, make_uint3(0, 0, 0), sdf_shape - 1);
    
    auto clamped_index = [sdf, sdf_shape](int x, int y, int z) {
        return index(sdf,
                     clamp(x, 0, int(sdf_shape.x - 1)),
                     clamp(y, 0, int(sdf_shape.y - 1)),
                     clamp(z, 0, int(sdf_shape.z - 1)));
    };
    
    switch (dim) {
        case 0:
            return clamped_index(c_.x - 1, c_.y, c_.z) * h_kern[0] +
                   clamped_index(c_.x, c_.y, c_.z) * h_kern[1] +
                   clamped_index(c_.x + 1, c_.y, c_.z) * h_kern[2];
        case 1:
            return clamped_index(c_.x, c_.y - 1, c_.z) * h_kern[0] +
                   clamped_index(c_.x, c_.y, c_.z) * h_kern[1] +
                   clamped_index(c_.x, c_.y + 1, c_.z) * h_kern[2];
        case 2:
            return clamped_index(c_.x, c_.y, c_.z - 1) * h_kern[0] +
                   clamped_index(c_.x, c_.y, c_.z) * h_kern[1] +
                   clamped_index(c_.x, c_.y, c_.z + 1) * h_kern[2];
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
    
    uint3 c_ = clamp(pos, make_uint3(0, 0, 0), sdf_shape - 1);
    
    auto clamped_index = [sdf, sdf_shape](int x, int y, int z) {
        return index(sdf,
                     clamp(x, 0, int(sdf_shape.x - 1)),
                     clamp(y, 0, int(sdf_shape.y - 1)),
                     clamp(z, 0, int(sdf_shape.z - 1)));
    };
    
    switch (dim) {
        case 0:
            return clamped_index(c_.x - 1, c_.y, c_.z) * h_kern[0] +
                   clamped_index(c_.x + 1, c_.y, c_.z) * h_kern[1];
        case 1:
            return clamped_index(c_.x, c_.y - 1, c_.z) * h_kern[0] +
                   clamped_index(c_.x, c_.y + 1, c_.z) * h_kern[1];
        case 2:
            return clamped_index(c_.x, c_.y, c_.z - 1) * h_kern[0] +
                   clamped_index(c_.x, c_.y, c_.z + 1) * h_kern[1];
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
    
    float h_p_x = h_p(sdf, sdf_shape, pos, 0);
    float h_p_y = h_p(sdf, sdf_shape, pos, 1);
    float h_p_z = h_p(sdf, sdf_shape, pos, 2);
    
    // TODO: properly handle degenerate case where
    // normal equals zero? or should we just add an offset
    return normalize(make_float3(
                         h_p_x * h_y * h_z + 1e-7,
                         h_p_y * h_z * h_x + 1e-7,
                         h_p_z * h_x * h_y + 1e-7
                     ));
}

__global__
void sobel(
    float* sdf_, size_t sdf_shape[3],
    float3* normals_) {
    
    cuda_array<float, 3> sdf;
    assign<float, 3>(&sdf,
                     sdf_,
                     sdf_shape);
                     
    cuda_array<float3, 3> normals;
    assign<float3, 3>(&normals,
                      normals_,
                      sdf_shape);
                      
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint j = blockIdx.y * blockDim.y + threadIdx.y;
    uint k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= sdf_shape[0] || j >= sdf_shape[1] || k >= sdf_shape[2]) {
        return;
    }
    
    float3 sb = -1.0f * sobel_at(&sdf,
                                 make_uint3((uint) sdf_shape[0],
                                            (uint) sdf_shape[1],
                                            (uint) sdf_shape[2]),
                                 make_uint3(i, j, k));
    index(&normals, i, j, k) = sb;
    //index(&normals, i, j, k) = make_float3(i, j, k);
    //normals.data[i + j * 64 + k * 64 * 64] = make_float3(i, j, k);
}

typedef struct {
    float3 pos[ITERATIONS + 1];
    float dist[ITERATIONS + 1];
    float3 normal[ITERATIONS + 1];
    float g_d[ITERATIONS + 1];
    float opc[ITERATIONS + 1];
    float3 intensity[ITERATIONS + 1];
    float3 volumetric_shaded[ITERATIONS + 1];
    
    float3 ray_pos;
    float3 ray_vec;
    float3 origin;
    
    float step;
} chk;

__device__ inline float norm_sq(float3 a) {
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

__device__ inline float step_f(float a) {
    return (a < 0.0f) ? 0.0f : 1.0f;
}

__device__ inline float mesa(float a, float low = 0.0f, float high = 1.0f) {
    return (a < low || a > high) ? 0.0f : 1.0f;
}

__device__ void create_chk(chk& c) {
    c.opc[0] = 0.0f;
    c.volumetric_shaded[0] = make_float3(0.0f, 0.0f, 0.0f);
}

__device__ float3 light_source(float3 light_color,
                               float3 position,
                               float3 light_position,
                               float3 normal,
                               float kd = 0.7f,
                               float ks = 0.3f,
                               float ka = 100.0f) {
    float3 light_vec = normalize(light_position - position);
    float3 diffuse = kd * clamp(dot(normal, light_vec), 0.0f, 1.0f) * light_color;
    return diffuse;
}

__device__ float3 shade(float3 position, float3 origin, float3 normal,
                        float3 top_light_color = make_float3(0.6f, 0.6f, 0.0f),
                        float3 self_light_color = make_float3(0.4f, 0.0f, 0.4f),
                        float3 top_light_pos = make_float3(10.0f, 30.0f, 0.0f)) {
    float3 self_light_pos = origin;
    
    float3 top_light = light_source(top_light_color, position, top_light_pos,
                                    normal);
    float3 self_light = light_source(self_light_color, position, self_light_pos,
                                     normal);
                                     
    float3 total_light = top_light + self_light;
    
    //return total_light;
    // I disabled the shading for now, the image has constant intensity
    return make_float3(1.0f, 1.0f, 1.0f);
}

__device__ float3 light_source_d(float3 light_color,
                                 float3 position,
                                 float3 light_position,
                                 float3 normal,
                                 float3 dNormaldSDF,
                                 float kd = 0.7f,
                                 float ks = 0.3f,
                                 float ka = 100.0f) {
    float3 light_vec = normalize(light_position - position);
    return kd * mesa(dot(normal, light_vec), 0.0f, 1.0f) *
           dot(dNormaldSDF, light_vec)
           * light_color;
}

__device__ float3 shade_d(float3 position, float3 origin, float3 normal,
                          float3 dNormaldSDF,
                          float3 top_light_color = make_float3(0.6f, 0.6f, 0.0f),
                          float3 self_light_color = make_float3(0.4f, 0.0f, 0.4f),
                          float3 top_light_pos = make_float3(10.0f, 30.0f, 0.0f)) {
    float3 self_light_pos = origin;
    
    float3 top_light_d = light_source_d(top_light_color, position, top_light_pos,
                                        normal, dNormaldSDF);
    float3 self_light_d = light_source_d(self_light_color, position, self_light_pos,
                                         normal, dNormaldSDF);
                                         
    float3 total_light_d = top_light_d + self_light_d;
    
    return total_light_d;
}

__device__ inline float to_render_dist(float dist, float scale_factor = 1.0f) {
    return scale_factor / (10.0f + (1.0f - clamp(abs(dist), 0.0f, 1.0f)) * 90.0f);
}

__device__ inline float normal_pdf(float x, float sigma = 1e-7f,
                                   float mean = 0.0f) {
    return (1.0f / sqrtf(2.0f * (float) M_PI * sigma * sigma)) *
           expf((x - mean) * (x - mean) / (-2.0f * sigma * sigma));
}

__device__ inline float relu(float a) {
    return max(a, 0.0f);
}

__device__ inline float normal_pdf_rectified(float x, float sigma = 1e-2f,
        float mean = 0.0f) {
    return normal_pdf(relu(x), sigma, mean);
}

__device__ float3 forward_pass(int x,
                               int y,
                               cuda_array<float, 3>* sdf,
                               float3 p0,
                               float3 p1,
                               uint3 sdf_shape,
                               cuda_array<float3, 3>* normals,
                               
                               
                               cuda_array<float, 2>* projection,
                               cuda_array<float, 2>* view,
                               cuda_array<float, 2>* ray_transform,
                               
                               int width,
                               int height,
                               chk& ch
                              ) {
    ch.ray_pos = make_float3(0.0f, 0.0f, 0.0f);
    ch.ray_vec = make_float3(0.0f, 0.0f, 0.0f);
    ch.origin = make_float3(0.0f, 0.0f, 0.0f);
    
    ch.step = 1.0f / 100.0f;
    
    const float u_s = 1.0f;
    const float k = -1.0f;
    
    projection_gen(x, y, projection, view, width, height, ch.ray_pos, ch.ray_vec,
                   ch.origin);
                   
    float3 transformed_ray_pos = apply_affine<float>(ray_transform, ch.ray_pos);
    ch.pos[0] = transformed_ray_pos;
    //#pragma unroll
    for (int tr = 0; tr < ITERATIONS; tr++) {
        ch.dist[tr] = trilinear<float>(sdf, p0, p1, make_int3(sdf_shape), ch.pos[tr],
                                       1.0f);
        // on iteration tr, because the pos was from iteration tr
        //float ds = to_render_dist(ch.dist[tr]);
        
        float step = ch.step;
        //float step = 1.0f / 100.0f;
        //float step = ds;
        // uncomment for sphere tracing
        //float step = ch.dist[tr];
        
        ch.pos[tr + 1] = ch.pos[tr] + step * ch.ray_vec;
        
        // also on iteration tr
        ch.g_d[tr] = normal_pdf_rectified(ch.dist[tr]);
        
        ch.normal[tr] = trilinear<float3>(normals, p0, p1,
                                          make_int3(sdf_shape),
                                          ch.pos[tr],
                                          make_float3(0.0f, 0.0f, 0.0f));
                                          
        ch.intensity[tr] = shade(ch.pos[tr], ch.origin, ch.normal[tr]);
        
        ch.opc[tr + 1] = ch.opc[tr] + ch.g_d[tr] * step;
        
        float scattering = ch.g_d[tr] * u_s;
        
        ch.volumetric_shaded[tr + 1] =
            ch.volumetric_shaded[tr] + scattering * expf(k * ch.opc[tr + 1]) *
            ch.intensity[tr] * step;
    }
    
    //return ch.intensity[ITERATIONS / 2];
    return ch.volumetric_shaded[ITERATIONS];
    //return index(normals, x / (1000 / 64), y / (1000 / 64), 30);
}

__device__
void backwards_pass(
    int x,
    int y,
    cuda_array<float, 3>* sdf,
    float3 p0,
    float3 p1,
    uint3 sdf_shape,
    cuda_array<float3, 3>* normals,
    
    cuda_array<float, 3>* target,
    
    cuda_array<float, 2>* projection,
    cuda_array<float, 2>* view,
    cuda_array<float, 2>* ray_transform,
    
    cuda_array<float, 1>* loss,
    cuda_array<float, 3>* dLossdSDF,
    cuda_array<float, 2>* dLossdTransform,
    
    int width,
    int height,
    chk& ch
) {
    float3 target_color = make_float3(index(target, 0, x, y),
                                      index(target, 1, x, y),
                                      index(target, 2, x, y));
    atomicAdd(&index(loss, 0),
              norm_sq(target_color - ch.volumetric_shaded[ITERATIONS]));
              
    // reset each iteration
    // represents the derivative of trilinear w.r.t the SDF
    cuda_array<float, 3> dsdf;
    float dsdf_data[2][2][2];
    size_t dsdf_data_dims[3] = {2, 2, 2};
    assign(&dsdf, (float*) dsdf_data, dsdf_data_dims);
    
    // reset each iteration
    // represents the derivative of the trilinear of normals w.r.t the SDF
    cuda_array<float3, 3> dnormals;
    float3 dnormals_data[4][4][4];
    size_t dnormals_dims[3] = {4, 4, 4};
    assign(&dnormals, (float3*) dnormals_data, dnormals_dims);
    
    // not reset each iteration
    // represents the accumulation of the coefficient of opc
    cuda_array<float3, 3> opc_accumulator;
    float3 opc_accumulator_data[4][4][4];
    size_t opc_accumulator_dims[3] = {4, 4, 4};
    assign(&opc_accumulator, (float3*) opc_accumulator_data, opc_accumulator_dims);
    fill(&opc_accumulator, make_float3(0.0f));
    
    // TODO: set these somewhere else
    const float u_s = 1.0f;
    const float k = -1.0f;
    
    //#pragma unroll
    for (int tr = ITERATIONS; tr >= 1; tr--) {
        float3 pos = ch.pos[tr - 1];
        float dist = ch.dist[tr - 1];
        float g_d = ch.g_d[tr - 1];
        float opc_t1 = ch.opc[tr];
        float3 vs_t1 = ch.volumetric_shaded[tr];
        float3 intensity = ch.intensity[tr - 1];
        float scattering = g_d * u_s;
        
        int3 lp;
        int3 up;
        bool oob;
        
        float3 grid_space;
        float3 alpha = populate_trilinear_pos(p0, p1, make_int3(sdf_shape),
                                              pos, grid_space, lp, up, oob);
                                              
        dTrilinear_dSDF(lp, up,
                        &dsdf,
                        p0, p1, make_int3(sdf_shape),
                        
                        alpha, grid_space,
                        
                        pos, false);
                        
        dTrilinear_dNormals(lp, up,
                            sdf,
                            &dnormals,
                            p0, p1, make_int3(sdf_shape),
                            
                            alpha, grid_space,
                            
                            pos, false);
                            
        float expf_k_opc1 = expf(k * opc_t1);
        
        // see below for some explanation
        float3 t1 = k * SQUARE(ch.step) * intensity * scattering * expf_k_opc1;
        float3 t2 = k * ch.step * intensity * scattering * expf_k_opc1;
        float3 t3 = ch.step * intensity * expf_k_opc1;
        float t4 = ch.step * scattering * expf_k_opc1;
        float t5 = 1.0f;
        
        // the ijk coordinate system is centered at the ray evaluation position - 1
        for (int i = -1; i < 3; i++) {
            for (int j = -1; j < 3; j++) {
                for (int k = -1; k < 3; k++) {
                    // sdf_pos starts at (-1, -1, -1)
                    // sdf_location gives us the location of the ray evaluation
                    // in grid space
                    int3 sdf_location = make_int3(clamp(lp.x + i, 0, int(sdf_shape.x) - 1),
                                                  clamp(lp.y + j, 0, int(sdf_shape.y) - 1),
                                                  clamp(lp.z + k, 0, int(sdf_shape.z) - 1));
                                                  
                    // dTrilinear/dSDF(i, j, k) represents the derivative of the
                    // trilinear interpolation of the SDF at the location ijk
                    // (starts at (-1, -1, -1))
                    float dtrilinear_sdf_ijk = 0.0f;
                    
                    // if we actually have a valid trilinear interpolation of dsdf
                    if (i < 1 && j < 1 && k < 1) {
                        // note that dsdf starts at (0, 0, 0), but ijk starts
                        // at (-1, -1, -1) -- that's why there is an offset
                        dtrilinear_sdf_ijk = index_off(&dsdf, i, j, k, 1);
                    }
                    
                    // dNormalsTrilinear/dSDF(i, j, k) represents
                    // Trilinear(Normals(i, j, k)). It's the trilinear interpolated
                    // normal of the SDF at location ijk (starts at -1, -1, -1)
                    float3 dnormalstrilinear_sdf_ijk = index_off(&dnormals, i, j, k, 1);
                    
                    /*
                      dg_d/ddist -- (dist is a trilinear evaluation of the SDF)
                    
                                                                                                           2
                                                                         -0.5 (-mean + Max(0, dist_t(SDF)))
                                                                         ------------------------------------
                                                                                          2
                                                                                     sigma                                             d
                     -0.707106781186547 (-mean + Max(0, dist_t(SDF)))  e                                      Heaviside(dist_t(SDF)) -----(dist (SDF))
                                                                                                                                     dSDF
                     ------------------------------------------------------------------------------------------------------------------------------
                                                                                    ---------
                                                                               2   /        2
                                                                          sigma  \/ pi sigma
                    
                    */
                    // TODO: Move these somewhere else?
                    float mean = 0.0f;
                    float sigma = 1e-2f;
                    
                    // represents the derivative of g_d
                    float g_d_d = (-(1.0f / sqrtf(2.0f)) * (-1.0f * mean + max(0.0f, dist)) *
                                   exp((-0.5f * SQUARE(-1.0f * mean + max(0.0f, dist))) / (SQUARE(sigma))) *
                                   step_f(dist) * (dtrilinear_sdf_ijk)) /
                                  (SQUARE(sigma) * sqrtf(((float) M_PI) * SQUARE(sigma)));
                                  
                    // represents dScattering/dSDF
                    float scattering_d = g_d_d * u_s;
                    
                    // represents dIntensity/dSDF
                    /*float3 intensity_d = shade_d(pos, ch.origin, ch.normal[tr],
                      dnormalstrilinear_sdf_ijk);*/
                    // set to zero for now -- represents constant intensity (so derivative is 0)
                    float intensity_d = 0.0f;
                                                 
                    /**
                     * dvs_{t + 1}/dSDF
                     *
                     *       2                                   k*opc_t1(SDF)  d                                                          k*opc_t1(SDF)  d                                      k*opc_t1(SDF)  d                                                k*opc_t1(SDF)  d                      d
                     * k*step *intensity(SDF)*scattering_t(SDF)*e             *----(g_d_t(SDF)) + k*step*intensity(SDF)*scattering_t(SDF)*e             *----(opc_t(SDF)) + step*intensity(SDF)*e             *----(scattering_t(SDF)) + step*scattering_t(SDF)*e             *----(intensity(SDF)) + ----(vs_t(SDF))
                     *                                                         dSDF                                                                      dSDF                                                  dSDF                                                            dSDF                   dSDF
                     *
                     * The first term is t1 (dg_d/dSDF), second term is t2 (dopc_t/dSDF), third term is
                     * t3 (dScattering/dSDF), fourth term is t4 (dIntensity/dSDF), fifth term is t5 (dvs_t/dSDF)
                     *
                     * After substituting some of the terms that we computed above, we can rewrite dvs_{t + 1}/dSDf as
                     *
                     * ((t1 * A + t3 * B) * dTrilinear/dSDF + (t4 * C) * dNormalsTrilinear/dSDF) + (t2 * dopc_t/dSDF + t5 * dvs_t/dSDF)
                     *
                     * (we can get A and B and C from the g_d_d, scattering_d, and intensity_d respectively)
                     *
                     * Note that the first term in the expression above can just be added to our accumulator
                     *
                     * This leaves the remaining two terms
                     *
                     * (t2 * dopc_t/dSDF + t5 * dvs_t/dSDF)
                     *
                     * We keep accumulating the value of t2 and t5 through iterations of the reverse tracing loop.
                     * Okay, well t5 is actually kind of boring because it's just 1, but t2 is the interesting bit.
                     *
                     * We can rewrite dvs_{t + 1} as c_1 * dOpc_t1/dSDF + c_2 * dvs_t1/dSDF. c_1 starts out as zero on
                     * the first iteration, but by the second iteration it is equal to t2. (and c_2 is always 1)
                     *
                     **/
                    
                    float3 drops_out_vs = t1 * g_d_d + t3 * scattering_d + t4 * intensity_d;
                    float3 dopc_contribution = index_off(&opc_accumulator, i, j, k,
                                                         1) * (g_d_d * ch.step);
                    float3 dvsdSDF = drops_out_vs + dopc_contribution;
                    //dvsdSDF = dopc_contribution;
                    index_off(&opc_accumulator, i, j, k, 1) = index_off(&opc_accumulator, i, j, k,
                            1) + t2;
                            
                    /**
                     * Now that we have dvs/dSDF, we need to compute dLoss/dSDF
                     *
                     * 1/n * \sum_{i=0}^{n} (y().i - vs(SDF).i)^2
                     *
                     * and the derivative is
                     *
                     * 2/n * \sum{i=0}^{n} (y().i - vs(SDF).i) * (-dvs/dSDF))
                     */
                    
                    // divide by width * height?
                    float dLossdSDF_ijk = (2.0f / (3.0f)) * norm_sq((target_color - ch.volumetric_shaded[ITERATIONS]) * -1.0f *
                                          dvsdSDF);
                                          
                    if (!oob) {
                        atomicAdd(&index(dLossdSDF, sdf_location.x, sdf_location.y, sdf_location.z),
                                  dLossdSDF_ijk);
                    }
                }
            }
        }
    }
}

__global__
void render(float* projection_matrix_,
            float* view_matrix_,
            float* transform_matrix_,
            float* sdf_, size_t sdf_shape[3],
            float3* normals_,
            float* p0_,
            float* p1_,
            float* target_,
            size_t width,
            size_t height,
            
            bool only_forwards,
            
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
    size_t single_dim[1] = {1};
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
                     
    cuda_array<float3, 3> normals;
    assign<float3, 3>(&normals, normals_,
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
                     
    cuda_array<float, 1> loss;
    assign<float, 1>(&loss,
                     loss_,
                     single_dim);
                     
    cuda_array<float, 3> forward;
    assign<float, 3>(&forward,
                     forward_,
                     image_size);
                     
    cuda_array<float, 3> dLossdSDF;
    assign<float, 3>(&dLossdSDF,
                     dLossdSDF_,
                     sdf_shape);
                     
    cuda_array<float, 2> dLossdTransform;
    assign<float, 2>(&dLossdTransform,
                     dLossdTransform_,
                     mat4_size);
                     
    float3 p0_p = make_float3(index(&p0, 0), index(&p0, 1), index(&p0, 2));
    float3 p1_p = make_float3(index(&p1, 0), index(&p1, 1), index(&p1, 2));
    uint3 sdf_shape_p = make_uint3(sdf_shape[0],
                                   sdf_shape[1],
                                   sdf_shape[2]);
                                   
    chk ch;
    create_chk(ch);
    
    float3 c = forward_pass(i, j,
                            &sdf, p0_p, p1_p, sdf_shape_p,
                            &normals,
                            &projection, &view, &transform,
                            width, height, ch);
                            
    if (!only_forwards) {
        backwards_pass(i, j,
                       &sdf, p0_p, p1_p, sdf_shape_p,
                       &normals,
                       &target,
                       &projection, &view, &transform,
                       &loss,
                       &dLossdSDF, &dLossdTransform,
                       width, height, ch);
    }
    
    index(&forward, 0, i, j) = c.x;
    index(&forward, 1, i, j) = c.y;
    index(&forward, 2, i, j) = c.z;
}

void trace() {
    size_t model_n_matrix[3] = {
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
        {0.0f, 0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0}
    };
    
    float p0_matrix[3] = {
        -4.0f, -4.0f, -4.0f
        };
        
    float p1_matrix[3] = {
        4.0f, 4.0f, 4.0f
    };
    
    const int width = 200;
    const int height = 200;
    
    size_t mat4_dims[2] = {4, 4};
    size_t vec3_dims[1] = {3};
    size_t img_dims[3] = {3, width, height};
    size_t single_dim[1] = {1};
    
    size_t* mat4_dims_device;
    size_t* vec3_dims_device;
    size_t* model_n_matrix_device;
    size_t* target_n_matrix_device;
    size_t* img_dims_device;
    size_t* single_dim_device;
    
    size_t target_n_matrix[3];
    
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
                              
    cuda_array<float, 3>* model_sdf_host = create<float, 3>(model_n_matrix);
    gen_sdf<float>(example_sphere,
                   p0_matrix[0], p0_matrix[1], p0_matrix[2],
                   p1_matrix[0], p1_matrix[1], p1_matrix[2],
                   model_sdf_host);
    float* model_sdf_device = to_device<float, 3>(model_sdf_host,
                              &model_n_matrix_device);
                              
    int n_matrix_i[3];
    Buffer<float> target_sdf_buf(read_sdf("bunny.sdf",
                                          p0_matrix[0], p0_matrix[1], p0_matrix[2],
                                          p1_matrix[0], p1_matrix[1], p1_matrix[2],
                                          n_matrix_i[0], n_matrix_i[1], n_matrix_i[2],
                                          true, true, 8.0f));
    cuda_array<float, 3>* target_sdf_host = from_buffer<float, 3>(target_sdf_buf);
    for (int i = 0; i < 3; i++) {
        target_n_matrix[i] = n_matrix_i[i];
    }
    float* target_sdf_device = to_device<float, 3>(target_sdf_host,
                               &target_n_matrix_device);
                               
    cuda_array<float3, 3>* model_normals_host = create<float3, 3>(model_n_matrix);
    float3* model_normals_device = to_device<float3, 3>(model_normals_host,
                                   &model_n_matrix_device);
                                   
    cuda_array<float3, 3>* target_normals_host = create<float3, 3>(target_n_matrix);
    float3* target_normals_device = to_device<float3, 3>(target_normals_host,
                                    &target_n_matrix_device);
                                    
    cuda_array<float, 1>* p0_host = create<float, 1>(vec3_dims);
    assign(p0_host, (float*) p0_matrix, vec3_dims, true);
    float* p0_device = to_device<float, 1>(p0_host, &vec3_dims_device);
    
    cuda_array<float, 1>* p1_host = create<float, 1>(vec3_dims);
    assign(p1_host, (float*) p1_matrix, vec3_dims, true);
    float* p1_device = to_device<float, 1>(p1_host, &vec3_dims_device);
    
    cuda_array<float, 3>* target_host = create<float, 3>(img_dims);
    float* target_device = to_device<float, 3>(target_host, &img_dims_device);
    
    cuda_array<float, 1>* loss_host = create<float, 1>(single_dim);
    index(loss_host, 0) = 0.0f;
    float* loss_device = to_device<float, 1>(loss_host, &single_dim_device);
    
    cuda_array<float, 3>* forward_host = create<float, 3>(img_dims);
    float* forward_device = to_device<float, 3>(forward_host, &img_dims_device);
    
    cuda_array<float, 3>* dloss_dsdf_host = create<float, 3>(model_n_matrix);
    fill(dloss_dsdf_host, 0.0f);
    float* dloss_dsdf_device = to_device<float, 3>(dloss_dsdf_host,
                               &model_n_matrix_device);
                               
    cuda_array<float, 2>* dloss_dtransform_host = create<float, 2>(mat4_dims);
    float* dloss_dtransform_device = to_device<float, 2>(dloss_dtransform_host,
                                     &mat4_dims_device);
                                     
    const size_t sobel_block_size = 4;
    const size_t target_sobel_grid_size_x = (int)(ceil((float) target_n_matrix[0] /
                                            (float) sobel_block_size));
    const size_t target_sobel_grid_size_y = (int)(ceil((float) target_n_matrix[1] /
                                            (float) sobel_block_size));
    const size_t target_sobel_grid_size_z = (int)(ceil((float) target_n_matrix[2] /
                                            (float) sobel_block_size));
                                            
    dim3 target_sobel_blocks(target_sobel_grid_size_x,
                             target_sobel_grid_size_y,
                             target_sobel_grid_size_z);
    dim3 target_sobel_threads(sobel_block_size,
                              sobel_block_size,
                              sobel_block_size);
                              
    const size_t model_sobel_grid_size_x = (int)(ceil((float) model_n_matrix[0] /
                                           (float) sobel_block_size));
    const size_t model_sobel_grid_size_y = (int)(ceil((float) model_n_matrix[1] /
                                           (float) sobel_block_size));
    const size_t model_sobel_grid_size_z = (int)(ceil((float) model_n_matrix[2] /
                                           (float) sobel_block_size));
                                           
    dim3 model_sobel_blocks(model_sobel_grid_size_x,
                            model_sobel_grid_size_y,
                            model_sobel_grid_size_z);
    dim3 model_sobel_threads(sobel_block_size,
                             sobel_block_size,
                             sobel_block_size);
                             
    // TODO: figure out better streaming
    sobel <<< target_sobel_blocks, target_sobel_threads, 0 >>> (target_sdf_device,
            target_n_matrix_device,
            target_normals_device);
    sobel <<< model_sobel_blocks, model_sobel_threads, 0 >>> (model_sdf_device,
            model_n_matrix_device,
            model_normals_device);
            
    //cudaThreadSynchronize();
    
    const size_t block_size = 8;
    const size_t grid_size_x = (int)(ceil((float) width / (float) block_size));
    const size_t grid_size_y = (int)(ceil((float) height / (float) block_size));
    
    dim3 blocks(grid_size_x, grid_size_y);
    dim3 threads(block_size, block_size);
    
    auto start = std::chrono::steady_clock::now();
    
    render <<< blocks, threads, 0 >>> (projection_device,
                                       view_device,
                                       transform_device,
                                       
                                       target_sdf_device, target_n_matrix_device,
                                       target_normals_device,
                                       p0_device, p1_device,
                                       
                                       // dummy input
                                       forward_device,
                                       
                                       width, height,
                                       
                                       // only do forwards pass
                                       true,
                                       
                                       // outputs
                                       loss_device,
                                       target_device,
                                       dloss_dsdf_device,
                                       dloss_dtransform_device
                                      );
                                      
    cudaThreadSynchronize();
    
    auto end = std::chrono::steady_clock::now();
    
    auto diff = end - start;
    
    std::cout << "Rendered target image in "
              << std::chrono::duration <float, std::milli> (diff).count()
              << " ms"
              << std::endl << std::endl;
              
    to_host<float, 3>(target_device, target_host);
    write_img("target_cuda.bmp", target_host);
    
    AdamOptimizer<float, 3> optim(model_sdf_device, dloss_dsdf_device,
                                  model_sdf_host, dloss_dsdf_host,
                                  0.0f);
                                  
    for (int i = 0; i < 10000000; i++) {
        std::cout << "starting epoch " << i << std::endl;
        
        start = std::chrono::steady_clock::now();
        render <<< blocks, threads, 0 >>> (projection_device,
                                           view_device,
                                           transform_device,
                                           
                                           model_sdf_device, model_n_matrix_device,
                                           model_normals_device,
                                           p0_device, p1_device,
                                           
                                           target_device,
                                           
                                           width, height,
                                           
                                           // do both the forwards and the backwards pass
                                           false,
                                           
                                           // outputs
                                           loss_device,
                                           forward_device,
                                           dloss_dsdf_device,
                                           dloss_dtransform_device
                                          );
        cudaThreadSynchronize();
        
        end = std::chrono::steady_clock::now();
        
        diff = end - start;
        
        std::cout << "Rendered model image and performed backwards pass in "
                  << std::chrono::duration <float, std::milli> (diff).count()
                  << " ms"
                  << std::endl << std::endl;
                  
        to_host<float, 1>(loss_device, loss_host);
        to_host<float, 3>(forward_device, forward_host);
        to_host<float, 3>(dloss_dsdf_device, dloss_dsdf_host);
        
        std::cout << "loss " << index(loss_host, 0) << std::endl;
        
        write_img("forward_cuda.bmp", forward_host);
        
        /*for (int i = 0; i < model_n_matrix[0]; i++) {
            for (int j = 0; j < model_n_matrix[0]; j++) {
                for (int k = 0; k < model_n_matrix[0]; k++) {
                    printf("%0.2f\t", index(dloss_dsdf_host, i, j, k));
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            std::cout << "SDF: " << std::endl;
        
            for (int j = 0; j < model_n_matrix[0]; j++) {
                for (int k = 0; k < model_n_matrix[0]; k++) {
                    printf("%0.2f\t", index(model_sdf_host, i, j, k));
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }*/
        
        optim.step();
        
        // zero the loss/gradient/forward
        zero(loss_host, loss_device, 0.0f);
        zero(forward_host, forward_device, 0.0f);
        zero(dloss_dsdf_host, dloss_dsdf_device, 0.0f);
    }
}

int main() {
    trace();
}

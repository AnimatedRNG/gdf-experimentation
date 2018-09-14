#include "stdio.h"
#include "helper_math.h"

#define FLT_MAX 1e10

__device__ float smin(float a, float b, float k) {
    a = pow(a, k);
    b = pow(b, k);
    return pow((a * b) / (a + b), 1.0 / k);
}

__device__ float sign(float a) {
    return (a < 0) ? -1 : ((a > 0) ? 1 : 0);
}

__device__ float dot2(float4 a) {
    return dot(a, a);
}

__device__ float4 cross4(float4 a, float4 b) {
    float3 result = cross(make_float3(a.x, a.y, a.z),
                          make_float3(b.x, b.y, b.z));
    return make_float4(result.x, result.y, result.z, 0.0);
}

__device__ float sdTriangle(float4 p,
                            float4 a,
                            float4 b,
                            float4 c) {
    float4 ba = b - a;
    float4 pa = p - a;
    float4 cb = c - b;
    float4 pb = p - b;
    float4 ac = a - c;
    float4 pc = p - c;
    float4 nor = cross4(ba, ac);
    
    return sign(dot(-nor, pa)) *                // Note, we could use pb or pc here too
           sqrt((sign(dot(cross4(ba, nor), pa)) +
                 sign(dot(cross4(cb, nor), pb)) +
                 sign(dot(cross4(ac, nor), pc)) < 2.0) ?
                min(min(dot2(ba * clamp(dot(ba, pa) / dot2(ba), 0.0, 1.0) - pa),
                        dot2(cb * clamp(dot(cb, pb) / dot2(cb), 0.0, 1.0) - pb)),
                    dot2(ac * clamp(dot(ac, pc) / dot2(ac), 0.0, 1.0) - pc)) :
                dot(nor, pa) * dot(nor, pa) / dot2(nor));
}

extern "C"
__global__ void grid_execution(
    float* sdf_grid, // Output
    int x_size,
    int y_size,
    int z_size,
    float* positions,
    float* vertices_x,
    float* vertices_y,
    float* vertices_z,
    int* indices, int num_faces) {

    const uint i = threadIdx.x + blockIdx.x * blockDim.x;
    const uint j = threadIdx.y + blockIdx.y * blockDim.y;
    const uint k = threadIdx.z + blockIdx.z * blockDim.z;
    int resolution = x_size;

    float best_dist = FLT_MAX;
    float dist_val = 0.0;

    int sdf_index = i * resolution * resolution + j * resolution + k;
    float4 p = make_float4(positions[3 * sdf_index],
                           positions[3 * sdf_index + 1],
                           positions[3 * sdf_index + 2],
                           0.0);

    for (size_t index = 0; index < num_faces; index++) {
        int i1 = indices[3 * index];
        int i2 = indices[3 * index + 1];
        int i3 = indices[3 * index + 2];

        float4 v1 = make_float4(vertices_x[i1], vertices_y[i1], vertices_z[i1], 0.0);
        float4 v2 = make_float4(vertices_x[i2], vertices_y[i2], vertices_z[i2], 0.0);
        float4 v3 = make_float4(vertices_x[i3], vertices_y[i3], vertices_z[i3], 0.0);

        float4 nv = cross4(v1 - v2, v1 - v3);

        float new_dist = sdTriangle(p, v1, v2, v3);
        if (fabsf(new_dist) < best_dist) {
            best_dist = fabsf(new_dist);
            dist_val = new_dist;
        }
    }

    sdf_grid[sdf_index] = dist_val;
}

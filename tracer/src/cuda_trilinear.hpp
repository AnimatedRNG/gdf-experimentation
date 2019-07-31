#pragma once

#include "math.h"
#include "helper_math.h"

// idk why the lerp in helper_math.h isn't right
__device__ float my_lerp(float a, float b, float alpha) {
    return a * (1.0f - alpha) + b * alpha;
}

__device__ float3 my_lerp(float3 a, float3 b, float alpha) {
    return make_float3(
               a.x * (1.0f - alpha) + b.x * alpha,
               a.y * (1.0f - alpha) + b.y * alpha,
               a.z * (1.0f - alpha) + b.z * alpha
           );
}

template <typename T>
__device__ T trilinear(cuda_array < T, 3>* f,
                       float3 p0,
                       float3 p1,
                       int3 sdf_shape,

                       float3 position,
                       T exterior,
                       bool clamp_coords = false
                      ) {
    float3 sdf_shape_f = make_float3(sdf_shape.x, sdf_shape.y, sdf_shape.z);
    float3 grid_space = ((position - p0) / (p1 - p0)) * sdf_shape_f;

    if (!clamp_coords && (grid_space.x < 0.0f || grid_space.x > sdf_shape_f.x
                          || grid_space.y < 0.0f || grid_space.y > sdf_shape_f.y
                          || grid_space.z < 0.0f || grid_space.z > sdf_shape_f.z)) {
        return exterior;
    }

    int3 lp = make_int3(
                  clamp(int(floor(grid_space.x)), 0, sdf_shape.x - 1),
                  clamp(int(floor(grid_space.y)), 0, sdf_shape.y - 1),
                  clamp(int(floor(grid_space.z)), 0, sdf_shape.z - 1)
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

    T c00 = my_lerp(c000, c100, alpha.x);
    T c01 = my_lerp(c001, c101, alpha.x);
    T c10 = my_lerp(c010, c110, alpha.x);
    T c11 = my_lerp(c011, c111, alpha.x);

    T c0 = my_lerp(c00, c10, alpha.y);
    T c1 = my_lerp(c01, c11, alpha.y);

    T c = my_lerp(c0, c1, alpha.z);

    return c;
}

// returns alpha
template <typename T>
__device__ float3 populate_trilinear_pos(
    cuda_array < int3, 3>* sdf_pos, // 4x4x4
    float3 p0,
    float3 p1,
    int3 sdf_shape,

    float3 position,

    float3& grid_space
) {
    float3 sdf_shape_f = make_float3(sdf_shape.x, sdf_shape.y, sdf_shape.z);
    grid_space = ((position - p0) / (p1 - p0)) * sdf_shape_f;

    // at (0, 0, 0)
    int3 lp = make_int3(
                  clamp(int(floor(grid_space.x)), 0, sdf_shape.x - 1),
                  clamp(int(floor(grid_space.y)), 0, sdf_shape.y - 1),
                  clamp(int(floor(grid_space.z)), 0, sdf_shape.z - 1)
              );

    // at (1, 1, 1)
    int3 up = make_int3(
                  clamp(int(ceil(grid_space.x)), 0, sdf_shape.x - 1),
                  clamp(int(ceil(grid_space.y)), 0, sdf_shape.y - 1),
                  clamp(int(ceil(grid_space.z)), 0, sdf_shape.z - 1)
              );

    float3 alpha = grid_space - make_float3(lp);

    // sdf_pos starts at (-1, -1, -1)

    // region 0 - 1
    index_off(sdf_pos, 0, 0, 0, 1) = int3(lp.x, lp.y, lp.z);
    index_off(sdf_pos, 0, 0, 1, 1) = int3(lp.x, lp.y, up.z);
    index_off(sdf_pos, 0, 1, 0, 1) = int3(lp.x, up.y, lp.z);
    index_off(sdf_pos, 0, 1, 1, 1) = int3(lp.x, up.y, up.z);
    index_off(sdf_pos, 1, 0, 0, 1) = int3(up.x, lp.y, lp.z);
    index_off(sdf_pos, 1, 0, 1, 1) = int3(up.x, lp.y, up.z);
    index_off(sdf_pos, 1, 1, 0, 1) = int3(up.x, up.y, lp.z);
    index_off(sdf_pos, 1, 1, 1, 1) = int3(up.x, up.y, up.z);

    // region -1 - 0
    index_off(sdf_pos, 0, 0, 0, 0) = int3(lp.x - 1, lp.y - 1, lp.z - 1);
    index_off(sdf_pos, 0, 0, 1, 0) = int3(lp.x - 1, lp.y - 1, up.z - 1);
    index_off(sdf_pos, 0, 1, 0, 0) = int3(lp.x - 1, up.y - 1, lp.z - 1);
    index_off(sdf_pos, 0, 1, 1, 0) = int3(lp.x - 1, up.y - 1, up.z - 1);
    index_off(sdf_pos, 1, 0, 0, 0) = int3(up.x - 1, lp.y - 1, lp.z - 1);
    index_off(sdf_pos, 1, 0, 1, 0) = int3(up.x - 1, lp.y - 1, up.z - 1);
    index_off(sdf_pos, 1, 1, 0, 0) = int3(up.x - 1, up.y - 1, lp.z - 1);
    index_off(sdf_pos, 1, 1, 1, 0) = int3(up.x - 1, up.y - 1, up.z - 1);

    // region 1 - 2
    index_off(sdf_pos, 0, 0, 0, 2) = int3(lp.x + 1, lp.y + 1, lp.z + 1);
    index_off(sdf_pos, 0, 0, 1, 2) = int3(lp.x + 1, lp.y + 1, up.z + 1);
    index_off(sdf_pos, 0, 1, 0, 2) = int3(lp.x + 1, up.y + 1, lp.z + 1);
    index_off(sdf_pos, 0, 1, 1, 2) = int3(lp.x + 1, up.y + 1, up.z + 1);
    index_off(sdf_pos, 1, 0, 0, 2) = int3(up.x + 1, lp.y + 1, lp.z + 1);
    index_off(sdf_pos, 1, 0, 1, 2) = int3(up.x + 1, lp.y + 1, up.z + 1);
    index_off(sdf_pos, 1, 1, 0, 2) = int3(up.x + 1, up.y + 1, lp.z + 1);
    index_off(sdf_pos, 1, 1, 1, 2) = int3(up.x + 1, up.y + 1, up.z + 1);

    return alpha;
}

template <typename T>
__device__ void dTrilinear_dSDF(cuda_array < int3, 3>* sdf_pos,   // 4x4x4
                                cuda_array < float, 3>* dsdf_vals, // 2x2x2
                                float3 p0,
                                float3 p1,
                                int3 sdf_shape,

                                float3 alpha,
                                float3 grid_space,

                                float3 position,
                                bool clamp_coords = false
                               ) {
    int3 lp = index_off(sdf_pos, 0, 0, 0, 1);
    int3 up = index_off(sdf_pos, 1, 1, 1, 1);

    float3 sdf_shape_f = make_float3(sdf_shape.x, sdf_shape.y, sdf_shape.z);

    if (!clamp_coords && (grid_space.x < 0.0f || grid_space.x > sdf_shape_f.x
                          || grid_space.y < 0.0f || grid_space.y > sdf_shape_f.y
                          || grid_space.z < 0.0f || grid_space.z > sdf_shape_f.z)) {
        // no contribution to derivative
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    index_off(dsdf_vals, i, j, k, 0) = 0.0f;
                }
            }
        }
    } else {
        // NOTE -- offset 0 because dsdf_vals starts at (0, 0, 0)
        index_off(dsdf_vals, 0, 0, 0,
                  0) = (1 - alpha.x) * (1 - alpha.y) * (1 - alpha.z);
        index_off(dsdf_vals, 0, 0, 1, 0) = alpha.z * (1 - alpha.x) * (1 - alpha.y);
        index_off(dsdf_vals, 0, 1, 0, 0) = alpha.y * (1 - alpha.x) * (1 - alpha.z);
        index_off(dsdf_vals, 0, 1, 1, 0) = alpha.y * alpha.z * (1 - alpha.x);
        index_off(dsdf_vals, 1, 0, 0, 0) = alpha.x * (1 - alpha.y) * (1 - alpha.z);
        index_off(dsdf_vals, 1, 0, 1, 0) = alpha.x * alpha.z * (1 - alpha.y);
        index_off(dsdf_vals, 1, 1, 0, 0) = alpha.x * alpha.y * (1 - alpha.z);
        index_off(dsdf_vals, 1, 1, 1, 0) = alpha.x * alpha.y * alpha.z;
    }
}

template <typename T>
__device__ void dTrilinear_dNormals(cuda_array < int3, 3>*
                                    sdf_pos,          // 4x4x4
                                    cuda_array < float, 3>* sdf_vals,        // NxNxN
                                    cuda_array < float3, 3>* dsdf_vals,      // 2x2x2
                                    cuda_array < float3, 3>* dnormals_vals,  // 4x4x4 (x3)
                                    float3 p0,
                                    float3 p1,
                                    int3 sdf_shape,

                                    float3 alpha,
                                    float3 grid_space,

                                    float3 position,
                                    bool clamp_coords = false
                                   ) {
    float3 sdf_shape_f = make_float3(sdf_shape.x, sdf_shape.y, sdf_shape.z);
    int3 lp = index_off(sdf_pos, 0, 0, 0, 1);
    int3 up = index_off(sdf_pos, 1, 1, 1, 1);

    int3 offset = lp;

    if (!clamp_coords && (grid_space.x < 0.0f || grid_space.x > sdf_shape_f.x
                          || grid_space.y < 0.0f || grid_space.y > sdf_shape_f.y
                          || grid_space.z < 0.0f || grid_space.z > sdf_shape_f.z)) {
        // no contribution to derivative
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    index_off(dnormals_vals, i, j, k, 0) = make_float3(0.0f, 0.0f, 0.0f);
                }
            }
        }
    } else {
        index(dnormals_vals, -1, -1, -1).x = 0;
        index(dnormals_vals, -1, -1, 0).x = 0;
        index(dnormals_vals, -1, -1, 1).x = 0;
        index(dnormals_vals, -1, -1, 2).x = 0;
        index(dnormals_vals, -1, 0, -1).x = 0;
        index(dnormals_vals, -1, 0,
              0).x = (1 - alpha.x) * (1 - alpha.y) * (1 - alpha.z) * (index(sdf_vals,
                      offset.x + 0, offset.y - 1, offset.z + 0) + 2 * index(sdf_vals, offset.x + 0,
                              offset.y + 0, offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 1,
                                      offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                              offset.z - 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                      offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 0, offset.z + 1));
        index(dnormals_vals, -1, 0,
              1).x = alpha.z * (1 - alpha.x) * (1 - alpha.y) * (index(sdf_vals, offset.x + 0,
                      offset.y - 1, offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                              offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 1,
                                      offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                              offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                      offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 0, offset.z + 2));
        index(dnormals_vals, -1, 0, 2).x = 0;
        index(dnormals_vals, -1, 1, -1).x = 0;
        index(dnormals_vals, -1, 1,
              0).x = alpha.y * (1 - alpha.x) * (1 - alpha.z) * (index(sdf_vals, offset.x + 0,
                      offset.y + 0, offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                              offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 2,
                                      offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                              offset.z - 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                      offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 1, offset.z + 1));
        index(dnormals_vals, -1, 1,
              1).x = alpha.y * alpha.z * (1 - alpha.x) * (index(sdf_vals, offset.x + 0,
                      offset.y + 0, offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                              offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 2,
                                      offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                              offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                      offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 1, offset.z + 2));
        index(dnormals_vals, -1, 1, 2).x = 0;
        index(dnormals_vals, -1, 2, -1).x = 0;
        index(dnormals_vals, -1, 2, 0).x = 0;
        index(dnormals_vals, -1, 2, 1).x = 0;
        index(dnormals_vals, -1, 2, 2).x = 0;
        index(dnormals_vals, 0, -1, -1).x = 0;
        index(dnormals_vals, 0, -1,
              0).x = (1 - alpha.x) * (1 - alpha.y) * (1 - alpha.z) * (index(sdf_vals,
                      offset.x - 1, offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 1,
                              offset.y + 0, offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                      offset.z - 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                              offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 0, offset.z + 1));
        index(dnormals_vals, 0, -1,
              1).x = alpha.z * (1 - alpha.x) * (1 - alpha.y) * (index(sdf_vals, offset.x - 1,
                      offset.y + 0, offset.z + 1) - index(sdf_vals, offset.x + 1, offset.y + 0,
                              offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                        offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 0, offset.z + 2));
        index(dnormals_vals, 0, -1, 2).x = 0;
        index(dnormals_vals, 0, 0,
              -1).x = (1 - alpha.x) * (1 - alpha.y) * (1 - alpha.z) * (index(sdf_vals,
                      offset.x - 1, offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 1,
                              offset.y + 0, offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y - 1,
                                      offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                              offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 1, offset.z + 0));
        index(dnormals_vals, 0, 0,
              0).x = alpha.z * (1 - alpha.x) * (1 - alpha.y) * (index(sdf_vals, offset.x - 1,
                      offset.y + 0, offset.z + 1) - index(sdf_vals, offset.x + 1, offset.y + 0,
                              offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y - 1,
                                                offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                        offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                offset.z + 1)) + (1 - alpha.z) * (alpha.y * (1 - alpha.x) * (index(sdf_vals,
                                                                        offset.x - 1, offset.y + 1, offset.z + 0) - index(sdf_vals, offset.x + 1,
                                                                                offset.y + 1, offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                        offset.z - 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                        offset.z + 1)) + (1 - alpha.y) * (alpha.x * (index(sdf_vals, offset.x + 1,
                                                                                                                offset.y - 1, offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                        offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                        offset.z - 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                        offset.z + 1)) + 2 * (1 - alpha.x) * (index(sdf_vals, offset.x - 1,
                                                                                                                                                                offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                        offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y - 1,
                                                                                                                                                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                        offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                                offset.z + 0)) + 2 * (1 - alpha.x) * (index(sdf_vals, offset.x - 1,
                                                                                                                                                                                                        offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                                                                offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                        offset.z - 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                                offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 0, offset.z + 1))));
        index(dnormals_vals, 0, 0,
              1).x = alpha.z * (alpha.y * (1 - alpha.x) * (index(sdf_vals, offset.x - 1,
                                offset.y + 1, offset.z + 1) - index(sdf_vals, offset.x + 1, offset.y + 1,
                                        offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                        offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                offset.z + 2)) + (1 - alpha.y) * (alpha.x * (index(sdf_vals, offset.x + 1,
                                                                        offset.y - 1, offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                        offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                        offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                offset.z + 2)) + 2 * (1 - alpha.x) * (index(sdf_vals, offset.x - 1,
                                                                                                                        offset.y + 0, offset.z + 1) - index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y - 1,
                                                                                                                                        offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                        offset.z + 1)) + 2 * (1 - alpha.x) * (index(sdf_vals, offset.x - 1,
                                                                                                                                                                offset.y + 0, offset.z + 1) - index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                        offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                        offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                offset.z + 2)))) + (1 - alpha.x) * (1 - alpha.y) * (1 - alpha.z) * (index(
                                                                                                                                                                                                        sdf_vals, offset.x - 1, offset.y + 0, offset.z + 0) - index(sdf_vals,
                                                                                                                                                                                                                offset.x + 1, offset.y + 0, offset.z + 0)) * (index(sdf_vals, offset.x + 0,
                                                                                                                                                                                                                        offset.y - 1, offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                                offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 1, offset.z + 0));
        index(dnormals_vals, 0, 0,
              2).x = alpha.z * (1 - alpha.x) * (1 - alpha.y) * (index(sdf_vals, offset.x - 1,
                      offset.y + 0, offset.z + 1) - index(sdf_vals, offset.x + 1, offset.y + 0,
                              offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y - 1,
                                                offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                        offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 1, offset.z + 1));
        index(dnormals_vals, 0, 1,
              -1).x = alpha.y * (1 - alpha.x) * (1 - alpha.z) * (index(sdf_vals, offset.x - 1,
                      offset.y + 1, offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 1,
                              offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                        offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 2, offset.z + 0));
        index(dnormals_vals, 0, 1,
              0).x = alpha.y * alpha.z * (1 - alpha.x) * (index(sdf_vals, offset.x - 1,
                      offset.y + 1, offset.z + 1) - index(sdf_vals, offset.x + 1, offset.y + 1,
                              offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                        offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 2,
                                                                offset.z + 1)) + (1 - alpha.z) * (alpha.y * (alpha.x * (index(sdf_vals,
                                                                        offset.x + 1, offset.y + 0, offset.z + 0) + 2 * index(sdf_vals, offset.x + 1,
                                                                                offset.y + 1, offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 2,
                                                                                        offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                offset.z - 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                        offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                offset.z + 1)) + 2 * (1 - alpha.x) * (index(sdf_vals, offset.x - 1,
                                                                                                                        offset.y + 1, offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                        offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 2,
                                                                                                                                                        offset.z + 0)) + 2 * (1 - alpha.x) * (index(sdf_vals, offset.x - 1,
                                                                                                                                                                offset.y + 1, offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                                        offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                offset.z - 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                        offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                                offset.z + 1))) + (1 - alpha.x) * (1 - alpha.y) * (index(sdf_vals, offset.x - 1,
                                                                                                                                                                                                        offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                                                                offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                        offset.z - 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                                offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 0, offset.z + 1)));
        index(dnormals_vals, 0, 1,
              1).x = alpha.y * (1 - alpha.x) * (1 - alpha.z) * (index(sdf_vals, offset.x - 1,
                      offset.y + 1, offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 1,
                              offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                        offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 2,
                                                                offset.z + 0)) + alpha.z * (alpha.y * (alpha.x * (index(sdf_vals, offset.x + 1,
                                                                        offset.y + 0, offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 2,
                                                                                        offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                        offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                offset.z + 2)) + 2 * (1 - alpha.x) * (index(sdf_vals, offset.x - 1,
                                                                                                                        offset.y + 1, offset.z + 1) - index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                        offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 2,
                                                                                                                                                        offset.z + 1)) + 2 * (1 - alpha.x) * (index(sdf_vals, offset.x - 1,
                                                                                                                                                                offset.y + 1, offset.z + 1) - index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                                        offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                        offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                                offset.z + 2))) + (1 - alpha.x) * (1 - alpha.y) * (index(sdf_vals, offset.x - 1,
                                                                                                                                                                                                        offset.y + 0, offset.z + 1) - index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                                                                offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                        offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                                offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 0, offset.z + 2)));
        index(dnormals_vals, 0, 1,
              2).x = alpha.y * alpha.z * (1 - alpha.x) * (index(sdf_vals, offset.x - 1,
                      offset.y + 1, offset.z + 1) - index(sdf_vals, offset.x + 1, offset.y + 1,
                              offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                        offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 2, offset.z + 1));
        index(dnormals_vals, 0, 2, -1).x = 0;
        index(dnormals_vals, 0, 2,
              0).x = alpha.y * (1 - alpha.x) * (1 - alpha.z) * (index(sdf_vals, offset.x - 1,
                      offset.y + 1, offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 1,
                              offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                offset.z - 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                        offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 1, offset.z + 1));
        index(dnormals_vals, 0, 2,
              1).x = alpha.y * alpha.z * (1 - alpha.x) * (index(sdf_vals, offset.x - 1,
                      offset.y + 1, offset.z + 1) - index(sdf_vals, offset.x + 1, offset.y + 1,
                              offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                        offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 1, offset.z + 2));
        index(dnormals_vals, 0, 2, 2).x = 0;
        index(dnormals_vals, 1, -1, -1).x = 0;
        index(dnormals_vals, 1, -1,
              0).x = alpha.x * (1 - alpha.y) * (1 - alpha.z) * (index(sdf_vals, offset.x + 0,
                      offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 2, offset.y + 0,
                              offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                offset.z - 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                        offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 0, offset.z + 1));
        index(dnormals_vals, 1, -1,
              1).x = alpha.x * alpha.z * (1 - alpha.y) * (index(sdf_vals, offset.x + 0,
                      offset.y + 0, offset.z + 1) - index(sdf_vals, offset.x + 2, offset.y + 0,
                              offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                        offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 0, offset.z + 2));
        index(dnormals_vals, 1, -1, 2).x = 0;
        index(dnormals_vals, 1, 0,
              -1).x = alpha.x * (1 - alpha.y) * (1 - alpha.z) * (index(sdf_vals, offset.x + 0,
                      offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 2, offset.y + 0,
                              offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y - 1,
                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                        offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1, offset.z + 0));
        index(dnormals_vals, 1, 0,
              0).x = alpha.x * alpha.z * (1 - alpha.y) * (index(sdf_vals, offset.x + 0,
                      offset.y + 0, offset.z + 1) - index(sdf_vals, offset.x + 2, offset.y + 0,
                              offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y - 1,
                                                offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                        offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                offset.z + 1)) + (1 - alpha.z) * (alpha.x * alpha.y * (index(sdf_vals,
                                                                        offset.x + 0, offset.y + 1, offset.z + 0) - index(sdf_vals, offset.x + 2,
                                                                                offset.y + 1, offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                        offset.z - 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                        offset.z + 1)) + (1 - alpha.y) * (2 * alpha.x * (index(sdf_vals, offset.x + 0,
                                                                                                                offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 2, offset.y + 0,
                                                                                                                        offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y - 1,
                                                                                                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                        offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                offset.z + 0)) + 2 * alpha.x * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                        offset.z + 0) - index(sdf_vals, offset.x + 2, offset.y + 0,
                                                                                                                                                                offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                        offset.z - 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                                offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                                        offset.z + 1)) + (alpha.x - 1) * (index(sdf_vals, offset.x + 0, offset.y - 1,
                                                                                                                                                                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                        offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                                                offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                        offset.z - 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                                offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 0, offset.z + 1))));
        index(dnormals_vals, 1, 0,
              1).x = alpha.x * (1 - alpha.y) * (1 - alpha.z) * (index(sdf_vals, offset.x + 0,
                      offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 2, offset.y + 0,
                              offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y - 1,
                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                        offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                offset.z + 0)) + alpha.z * (alpha.x * alpha.y * (index(sdf_vals, offset.x + 0,
                                                                        offset.y + 1, offset.z + 1) - index(sdf_vals, offset.x + 2, offset.y + 1,
                                                                                offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                        offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                        offset.z + 2)) + (1 - alpha.y) * (2 * alpha.x * (index(sdf_vals, offset.x + 0,
                                                                                                                offset.y + 0, offset.z + 1) - index(sdf_vals, offset.x + 2, offset.y + 0,
                                                                                                                        offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y - 1,
                                                                                                                                offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                        offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                offset.z + 1)) + 2 * alpha.x * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                        offset.z + 1) - index(sdf_vals, offset.x + 2, offset.y + 0,
                                                                                                                                                                offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                        offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                                offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                                        offset.z + 2)) + (alpha.x - 1) * (index(sdf_vals, offset.x + 0, offset.y - 1,
                                                                                                                                                                                                offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                        offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                                                offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                        offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                                offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 0, offset.z + 2))));
        index(dnormals_vals, 1, 0,
              2).x = alpha.x * alpha.z * (1 - alpha.y) * (index(sdf_vals, offset.x + 0,
                      offset.y + 0, offset.z + 1) - index(sdf_vals, offset.x + 2, offset.y + 0,
                              offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y - 1,
                                                offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                        offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1, offset.z + 1));
        index(dnormals_vals, 1, 1,
              -1).x = alpha.x * alpha.y * (1 - alpha.z) * (index(sdf_vals, offset.x + 0,
                      offset.y + 1, offset.z + 0) - index(sdf_vals, offset.x + 2, offset.y + 1,
                              offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                        offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 2, offset.z + 0));
        index(dnormals_vals, 1, 1, 0).x = alpha.x * alpha.y * alpha.z * (index(sdf_vals,
                                          offset.x + 0, offset.y + 1, offset.z + 1) - index(sdf_vals, offset.x + 2,
                                                  offset.y + 1, offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                          offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                  offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 2,
                                                                          offset.z + 1)) + (1 - alpha.z) * (alpha.x * (1 - alpha.y) * (index(sdf_vals,
                                                                                  offset.x + 0, offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 2,
                                                                                          offset.y + 0, offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                  offset.z - 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                          offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                  offset.z + 1)) + alpha.y * (2 * alpha.x * (index(sdf_vals, offset.x + 0,
                                                                                                                          offset.y + 1, offset.z + 0) - index(sdf_vals, offset.x + 2, offset.y + 1,
                                                                                                                                  offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                          offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                  offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 2,
                                                                                                                                                          offset.z + 0)) + 2 * alpha.x * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                  offset.z + 0) - index(sdf_vals, offset.x + 2, offset.y + 1,
                                                                                                                                                                          offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                                                  offset.z - 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                                                          offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                                                                  offset.z + 1)) + (alpha.x - 1) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                          offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                                                  offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 2,
                                                                                                                                                                                                                          offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                                                                  offset.z - 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                                                                          offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 1, offset.z + 1))));
        index(dnormals_vals, 1, 1,
              1).x = alpha.x * alpha.y * (1 - alpha.z) * (index(sdf_vals, offset.x + 0,
                      offset.y + 1, offset.z + 0) - index(sdf_vals, offset.x + 2, offset.y + 1,
                              offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                        offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 2,
                                                                offset.z + 0)) + alpha.z * (alpha.x * (1 - alpha.y) * (index(sdf_vals,
                                                                        offset.x + 0, offset.y + 0, offset.z + 1) - index(sdf_vals, offset.x + 2,
                                                                                offset.y + 0, offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                        offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                        offset.z + 2)) + alpha.y * (2 * alpha.x * (index(sdf_vals, offset.x + 0,
                                                                                                                offset.y + 1, offset.z + 1) - index(sdf_vals, offset.x + 2, offset.y + 1,
                                                                                                                        offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                        offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 2,
                                                                                                                                                offset.z + 1)) + 2 * alpha.x * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                        offset.z + 1) - index(sdf_vals, offset.x + 2, offset.y + 1,
                                                                                                                                                                offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                                        offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                                                offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                                                        offset.z + 2)) + (alpha.x - 1) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                                        offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 2,
                                                                                                                                                                                                                offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                                                        offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                                                                offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 1, offset.z + 2))));
        index(dnormals_vals, 1, 1, 2).x = alpha.x * alpha.y * alpha.z * (index(sdf_vals,
                                          offset.x + 0, offset.y + 1, offset.z + 1) - index(sdf_vals, offset.x + 2,
                                                  offset.y + 1, offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                          offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                  offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 2, offset.z + 1));
        index(dnormals_vals, 1, 2, -1).x = 0;
        index(dnormals_vals, 1, 2,
              0).x = alpha.x * alpha.y * (1 - alpha.z) * (index(sdf_vals, offset.x + 0,
                      offset.y + 1, offset.z + 0) - index(sdf_vals, offset.x + 2, offset.y + 1,
                              offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y + 1,
                                                offset.z - 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                        offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1, offset.z + 1));
        index(dnormals_vals, 1, 2, 1).x = alpha.x * alpha.y * alpha.z * (index(sdf_vals,
                                          offset.x + 0, offset.y + 1, offset.z + 1) - index(sdf_vals, offset.x + 2,
                                                  offset.y + 1, offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 1,
                                                          offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                  offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1, offset.z + 2));
        index(dnormals_vals, 1, 2, 2).x = 0;
        index(dnormals_vals, 2, -1, -1).x = 0;
        index(dnormals_vals, 2, -1, 0).x = 0;
        index(dnormals_vals, 2, -1, 1).x = 0;
        index(dnormals_vals, 2, -1, 2).x = 0;
        index(dnormals_vals, 2, 0, -1).x = 0;
        index(dnormals_vals, 2, 0,
              0).x = -alpha.x * (1 - alpha.y) * (1 - alpha.z) * (index(sdf_vals, offset.x + 1,
                      offset.y - 1, offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                              offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                      offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                              offset.z - 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                      offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 0, offset.z + 1));
        index(dnormals_vals, 2, 0,
              1).x = -alpha.x * alpha.z * (1 - alpha.y) * (index(sdf_vals, offset.x + 1,
                      offset.y - 1, offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                              offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                      offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                              offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                      offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 0, offset.z + 2));
        index(dnormals_vals, 2, 0, 2).x = 0;
        index(dnormals_vals, 2, 1, -1).x = 0;
        index(dnormals_vals, 2, 1,
              0).x = -alpha.x * alpha.y * (1 - alpha.z) * (index(sdf_vals, offset.x + 1,
                      offset.y + 0, offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                              offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 2,
                                      offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y + 1,
                                              offset.z - 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                      offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1, offset.z + 1));
        index(dnormals_vals, 2, 1,
              1).x = -alpha.x * alpha.y * alpha.z * (index(sdf_vals, offset.x + 1,
                      offset.y + 0, offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                              offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 2,
                                      offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 1,
                                              offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                      offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1, offset.z + 2));
        index(dnormals_vals, 2, 1, 2).x = 0;
        index(dnormals_vals, 2, 2, -1).x = 0;
        index(dnormals_vals, 2, 2, 0).x = 0;
        index(dnormals_vals, 2, 2, 1).x = 0;
        index(dnormals_vals, 2, 2, 2).x = 0;
        index(dnormals_vals, -1, -1, -1).y = 0;
        index(dnormals_vals, -1, -1, 0).y = 0;
        index(dnormals_vals, -1, -1, 1).y = 0;
        index(dnormals_vals, -1, -1, 2).y = 0;
        index(dnormals_vals, -1, 0, -1).y = 0;
        index(dnormals_vals, -1, 0,
              0).y = (1 - alpha.x) * (1 - alpha.y) * (1 - alpha.z) * (index(sdf_vals,
                      offset.x + 0, offset.y - 1, offset.z + 0) - index(sdf_vals, offset.x + 0,
                              offset.y + 1, offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                      offset.z - 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                              offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 0, offset.z + 1));
        index(dnormals_vals, -1, 0,
              1).y = alpha.z * (1 - alpha.x) * (1 - alpha.y) * (index(sdf_vals, offset.x + 0,
                      offset.y - 1, offset.z + 1) - index(sdf_vals, offset.x + 0, offset.y + 1,
                              offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                        offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 0, offset.z + 2));
        index(dnormals_vals, -1, 0, 2).y = 0;
        index(dnormals_vals, -1, 1, -1).y = 0;
        index(dnormals_vals, -1, 1,
              0).y = alpha.y * (1 - alpha.x) * (1 - alpha.z) * (index(sdf_vals, offset.x + 0,
                      offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 0, offset.y + 2,
                              offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                offset.z - 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                        offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 1, offset.z + 1));
        index(dnormals_vals, -1, 1,
              1).y = alpha.y * alpha.z * (1 - alpha.x) * (index(sdf_vals, offset.x + 0,
                      offset.y + 0, offset.z + 1) - index(sdf_vals, offset.x + 0, offset.y + 2,
                              offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                        offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 1, offset.z + 2));
        index(dnormals_vals, -1, 1, 2).y = 0;
        index(dnormals_vals, -1, 2, -1).y = 0;
        index(dnormals_vals, -1, 2, 0).y = 0;
        index(dnormals_vals, -1, 2, 1).y = 0;
        index(dnormals_vals, -1, 2, 2).y = 0;
        index(dnormals_vals, 0, -1, -1).y = 0;
        index(dnormals_vals, 0, -1,
              0).y = (1 - alpha.x) * (1 - alpha.y) * (1 - alpha.z) * (index(sdf_vals,
                      offset.x - 1, offset.y + 0, offset.z + 0) + 2 * index(sdf_vals, offset.x + 0,
                              offset.y + 0, offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 0,
                                      offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                              offset.z - 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                      offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 0, offset.z + 1));
        index(dnormals_vals, 0, -1,
              1).y = alpha.z * (1 - alpha.x) * (1 - alpha.y) * (index(sdf_vals, offset.x - 1,
                      offset.y + 0, offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                              offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 0,
                                      offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                              offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                      offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 0, offset.z + 2));
        index(dnormals_vals, 0, -1, 2).y = 0;
        index(dnormals_vals, 0, 0,
              -1).y = (1 - alpha.x) * (1 - alpha.y) * (1 - alpha.z) * (index(sdf_vals,
                      offset.x + 0, offset.y - 1, offset.z + 0) - index(sdf_vals, offset.x + 0,
                              offset.y + 1, offset.z + 0)) * (index(sdf_vals, offset.x - 1, offset.y + 0,
                                      offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                              offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 0, offset.z + 0));
        index(dnormals_vals, 0, 0,
              0).y = alpha.z * (1 - alpha.x) * (1 - alpha.y) * (index(sdf_vals, offset.x + 0,
                      offset.y - 1, offset.z + 1) - index(sdf_vals, offset.x + 0, offset.y + 1,
                              offset.z + 1)) * (index(sdf_vals, offset.x - 1, offset.y + 0,
                                                offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                        offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                offset.z + 1)) + (1 - alpha.z) * (alpha.y * (1 - alpha.x) * (index(sdf_vals,
                                                                        offset.x - 1, offset.y + 1, offset.z + 0) + 2 * index(sdf_vals, offset.x + 0,
                                                                                offset.y + 1, offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                        offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                offset.z - 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                        offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                offset.z + 1)) + (1 - alpha.y) * (alpha.x * (index(sdf_vals, offset.x + 1,
                                                                                                                        offset.y - 1, offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                        offset.z - 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                        offset.z + 1)) + 2 * (1 - alpha.x) * (index(sdf_vals, offset.x + 0,
                                                                                                                                                                offset.y - 1, offset.z + 0) - index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                        offset.z + 0)) * (index(sdf_vals, offset.x - 1, offset.y + 0,
                                                                                                                                                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                        offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                                                offset.z + 0)) + 2 * (1 - alpha.x) * (index(sdf_vals, offset.x + 0,
                                                                                                                                                                                                        offset.y - 1, offset.z + 0) - index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                                                offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                        offset.z - 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                                offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 0, offset.z + 1))));
        index(dnormals_vals, 0, 0,
              1).y = alpha.z * (alpha.y * (1 - alpha.x) * (index(sdf_vals, offset.x - 1,
                                offset.y + 1, offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                        offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                        offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                        offset.z + 2)) + (1 - alpha.y) * (alpha.x * (index(sdf_vals, offset.x + 1,
                                                                                offset.y - 1, offset.z + 1) - index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                        offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                        offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                offset.z + 2)) + 2 * (1 - alpha.x) * (index(sdf_vals, offset.x + 0,
                                                                                                                        offset.y - 1, offset.z + 1) - index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                offset.z + 1)) * (index(sdf_vals, offset.x - 1, offset.y + 0,
                                                                                                                                        offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                        offset.z + 1)) + 2 * (1 - alpha.x) * (index(sdf_vals, offset.x + 0,
                                                                                                                                                                offset.y - 1, offset.z + 1) - index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                        offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                        offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                offset.z + 2)))) + (1 - alpha.x) * (1 - alpha.y) * (1 - alpha.z) * (index(
                                                                                                                                                                                                        sdf_vals, offset.x + 0, offset.y - 1, offset.z + 0) - index(sdf_vals,
                                                                                                                                                                                                                offset.x + 0, offset.y + 1, offset.z + 0)) * (index(sdf_vals, offset.x - 1,
                                                                                                                                                                                                                        offset.y + 0, offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                                offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 0, offset.z + 0));
        index(dnormals_vals, 0, 0,
              2).y = alpha.z * (1 - alpha.x) * (1 - alpha.y) * (index(sdf_vals, offset.x + 0,
                      offset.y - 1, offset.z + 1) - index(sdf_vals, offset.x + 0, offset.y + 1,
                              offset.z + 1)) * (index(sdf_vals, offset.x - 1, offset.y + 0,
                                                offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                        offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 0, offset.z + 1));
        index(dnormals_vals, 0, 1,
              -1).y = alpha.y * (1 - alpha.x) * (1 - alpha.z) * (index(sdf_vals, offset.x + 0,
                      offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 0, offset.y + 2,
                              offset.z + 0)) * (index(sdf_vals, offset.x - 1, offset.y + 1,
                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                        offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1, offset.z + 0));
        index(dnormals_vals, 0, 1,
              0).y = alpha.y * alpha.z * (1 - alpha.x) * (index(sdf_vals, offset.x + 0,
                      offset.y + 0, offset.z + 1) - index(sdf_vals, offset.x + 0, offset.y + 2,
                              offset.z + 1)) * (index(sdf_vals, offset.x - 1, offset.y + 1,
                                                offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                        offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                offset.z + 1)) + (1 - alpha.z) * (alpha.y * (alpha.x * (index(sdf_vals,
                                                                        offset.x + 1, offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 1,
                                                                                offset.y + 2, offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                        offset.z - 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                        offset.z + 1)) + 2 * (1 - alpha.x) * (index(sdf_vals, offset.x + 0,
                                                                                                                offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 0, offset.y + 2,
                                                                                                                        offset.z + 0)) * (index(sdf_vals, offset.x - 1, offset.y + 1,
                                                                                                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                        offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                offset.z + 0)) + 2 * (1 - alpha.x) * (index(sdf_vals, offset.x + 0,
                                                                                                                                                        offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 0, offset.y + 2,
                                                                                                                                                                offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                        offset.z - 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                        offset.z + 1))) + (1 - alpha.y) * (alpha.x - 1) * (index(sdf_vals, offset.x - 1,
                                                                                                                                                                                                offset.y + 0, offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                        offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                                                                offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                        offset.z - 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                                offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 0, offset.z + 1)));
        index(dnormals_vals, 0, 1,
              1).y = alpha.y * (1 - alpha.x) * (1 - alpha.z) * (index(sdf_vals, offset.x + 0,
                      offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 0, offset.y + 2,
                              offset.z + 0)) * (index(sdf_vals, offset.x - 1, offset.y + 1,
                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                        offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                offset.z + 0)) + alpha.z * (alpha.y * (alpha.x * (index(sdf_vals, offset.x + 1,
                                                                        offset.y + 0, offset.z + 1) - index(sdf_vals, offset.x + 1, offset.y + 2,
                                                                                offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                        offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                        offset.z + 2)) + 2 * (1 - alpha.x) * (index(sdf_vals, offset.x + 0,
                                                                                                                offset.y + 0, offset.z + 1) - index(sdf_vals, offset.x + 0, offset.y + 2,
                                                                                                                        offset.z + 1)) * (index(sdf_vals, offset.x - 1, offset.y + 1,
                                                                                                                                offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                        offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                offset.z + 1)) + 2 * (1 - alpha.x) * (index(sdf_vals, offset.x + 0,
                                                                                                                                                        offset.y + 0, offset.z + 1) - index(sdf_vals, offset.x + 0, offset.y + 2,
                                                                                                                                                                offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                        offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                        offset.z + 2))) + (1 - alpha.y) * (alpha.x - 1) * (index(sdf_vals, offset.x - 1,
                                                                                                                                                                                                offset.y + 0, offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                        offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                                                                offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                        offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                                offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 0, offset.z + 2)));
        index(dnormals_vals, 0, 1,
              2).y = alpha.y * alpha.z * (1 - alpha.x) * (index(sdf_vals, offset.x + 0,
                      offset.y + 0, offset.z + 1) - index(sdf_vals, offset.x + 0, offset.y + 2,
                              offset.z + 1)) * (index(sdf_vals, offset.x - 1, offset.y + 1,
                                                offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                        offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1, offset.z + 1));
        index(dnormals_vals, 0, 2, -1).y = 0;
        index(dnormals_vals, 0, 2,
              0).y = alpha.y * (1 - alpha.z) * (alpha.x - 1) * (index(sdf_vals, offset.x - 1,
                      offset.y + 1, offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                              offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                      offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                              offset.z - 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                      offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 1, offset.z + 1));
        index(dnormals_vals, 0, 2,
              1).y = alpha.y * alpha.z * (alpha.x - 1) * (index(sdf_vals, offset.x - 1,
                      offset.y + 1, offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                              offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                      offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                              offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                      offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 1, offset.z + 2));
        index(dnormals_vals, 0, 2, 2).y = 0;
        index(dnormals_vals, 1, -1, -1).y = 0;
        index(dnormals_vals, 1, -1,
              0).y = alpha.x * (1 - alpha.y) * (1 - alpha.z) * (index(sdf_vals, offset.x + 0,
                      offset.y + 0, offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                              offset.z + 0) + index(sdf_vals, offset.x + 2, offset.y + 0,
                                      offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                              offset.z - 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                      offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 0, offset.z + 1));
        index(dnormals_vals, 1, -1,
              1).y = alpha.x * alpha.z * (1 - alpha.y) * (index(sdf_vals, offset.x + 0,
                      offset.y + 0, offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                              offset.z + 1) + index(sdf_vals, offset.x + 2, offset.y + 0,
                                      offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                              offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                      offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 0, offset.z + 2));
        index(dnormals_vals, 1, -1, 2).y = 0;
        index(dnormals_vals, 1, 0,
              -1).y = alpha.x * (1 - alpha.y) * (1 - alpha.z) * (index(sdf_vals, offset.x + 1,
                      offset.y - 1, offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 1,
                              offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                        offset.z + 0) + index(sdf_vals, offset.x + 2, offset.y + 0, offset.z + 0));
        index(dnormals_vals, 1, 0,
              0).y = alpha.x * alpha.z * (1 - alpha.y) * (index(sdf_vals, offset.x + 1,
                      offset.y - 1, offset.z + 1) - index(sdf_vals, offset.x + 1, offset.y + 1,
                              offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                        offset.z + 1) + index(sdf_vals, offset.x + 2, offset.y + 0,
                                                                offset.z + 1)) + (1 - alpha.z) * (alpha.x * alpha.y * (index(sdf_vals,
                                                                        offset.x + 0, offset.y + 1, offset.z + 0) + 2 * index(sdf_vals, offset.x + 1,
                                                                                offset.y + 1, offset.z + 0) + index(sdf_vals, offset.x + 2, offset.y + 1,
                                                                                        offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                offset.z - 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                        offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                offset.z + 1)) + (1 - alpha.y) * (2 * alpha.x * (index(sdf_vals, offset.x + 1,
                                                                                                                        offset.y - 1, offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                        offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                offset.z + 0) + index(sdf_vals, offset.x + 2, offset.y + 0,
                                                                                                                                                        offset.z + 0)) + 2 * alpha.x * (index(sdf_vals, offset.x + 1, offset.y - 1,
                                                                                                                                                                offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                                        offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                                offset.z - 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                                        offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                                                offset.z + 1)) + (1 - alpha.x) * (index(sdf_vals, offset.x + 0, offset.y - 1,
                                                                                                                                                                                                        offset.z + 0) - index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                                                offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                        offset.z - 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                                offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 0, offset.z + 1))));
        index(dnormals_vals, 1, 0,
              1).y = alpha.x * (1 - alpha.y) * (1 - alpha.z) * (index(sdf_vals, offset.x + 1,
                      offset.y - 1, offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 1,
                              offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                        offset.z + 0) + index(sdf_vals, offset.x + 2, offset.y + 0,
                                                                offset.z + 0)) + alpha.z * (alpha.x * alpha.y * (index(sdf_vals, offset.x + 0,
                                                                        offset.y + 1, offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                offset.z + 1) + index(sdf_vals, offset.x + 2, offset.y + 1,
                                                                                        offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                        offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                offset.z + 2)) + (1 - alpha.y) * (2 * alpha.x * (index(sdf_vals, offset.x + 1,
                                                                                                                        offset.y - 1, offset.z + 1) - index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                        offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                offset.z + 1) + index(sdf_vals, offset.x + 2, offset.y + 0,
                                                                                                                                                        offset.z + 1)) + 2 * alpha.x * (index(sdf_vals, offset.x + 1, offset.y - 1,
                                                                                                                                                                offset.z + 1) - index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                                        offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                                        offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                                                offset.z + 2)) + (1 - alpha.x) * (index(sdf_vals, offset.x + 0, offset.y - 1,
                                                                                                                                                                                                        offset.z + 1) - index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                                                offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                        offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                                offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 0, offset.z + 2))));
        index(dnormals_vals, 1, 0,
              2).y = alpha.x * alpha.z * (1 - alpha.y) * (index(sdf_vals, offset.x + 1,
                      offset.y - 1, offset.z + 1) - index(sdf_vals, offset.x + 1, offset.y + 1,
                              offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                        offset.z + 1) + index(sdf_vals, offset.x + 2, offset.y + 0, offset.z + 1));
        index(dnormals_vals, 1, 1,
              -1).y = alpha.x * alpha.y * (1 - alpha.z) * (index(sdf_vals, offset.x + 1,
                      offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 2,
                              offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                        offset.z + 0) + index(sdf_vals, offset.x + 2, offset.y + 1, offset.z + 0));
        index(dnormals_vals, 1, 1, 0).y = alpha.x * alpha.y * alpha.z * (index(sdf_vals,
                                          offset.x + 1, offset.y + 0, offset.z + 1) - index(sdf_vals, offset.x + 1,
                                                  offset.y + 2, offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                          offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                  offset.z + 1) + index(sdf_vals, offset.x + 2, offset.y + 1,
                                                                          offset.z + 1)) + (1 - alpha.z) * (-alpha.x * (1 - alpha.y) * (index(sdf_vals,
                                                                                  offset.x + 0, offset.y + 0, offset.z + 0) + 2 * index(sdf_vals, offset.x + 1,
                                                                                          offset.y + 0, offset.z + 0) + index(sdf_vals, offset.x + 2, offset.y + 0,
                                                                                                  offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                          offset.z - 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                  offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                          offset.z + 1)) + alpha.y * (2 * alpha.x * (index(sdf_vals, offset.x + 1,
                                                                                                                                  offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 2,
                                                                                                                                          offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                  offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                          offset.z + 0) + index(sdf_vals, offset.x + 2, offset.y + 1,
                                                                                                                                                                  offset.z + 0)) + 2 * alpha.x * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                          offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 2,
                                                                                                                                                                                  offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                                                          offset.z - 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                                                                  offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                                                                          offset.z + 1)) + (1 - alpha.x) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                  offset.z + 0) - index(sdf_vals, offset.x + 0, offset.y + 2,
                                                                                                                                                                                                                          offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                                                                  offset.z - 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                                                                          offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 1, offset.z + 1))));
        index(dnormals_vals, 1, 1,
              1).y = alpha.x * alpha.y * (1 - alpha.z) * (index(sdf_vals, offset.x + 1,
                      offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 2,
                              offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                        offset.z + 0) + index(sdf_vals, offset.x + 2, offset.y + 1,
                                                                offset.z + 0)) + alpha.z * (-alpha.x * (1 - alpha.y) * (index(sdf_vals,
                                                                        offset.x + 0, offset.y + 0, offset.z + 1) + 2 * index(sdf_vals, offset.x + 1,
                                                                                offset.y + 0, offset.z + 1) + index(sdf_vals, offset.x + 2, offset.y + 0,
                                                                                        offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                        offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                offset.z + 2)) + alpha.y * (2 * alpha.x * (index(sdf_vals, offset.x + 1,
                                                                                                                        offset.y + 0, offset.z + 1) - index(sdf_vals, offset.x + 1, offset.y + 2,
                                                                                                                                offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                        offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                offset.z + 1) + index(sdf_vals, offset.x + 2, offset.y + 1,
                                                                                                                                                        offset.z + 1)) + 2 * alpha.x * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                offset.z + 1) - index(sdf_vals, offset.x + 1, offset.y + 2,
                                                                                                                                                                        offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                                                        offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                                                                offset.z + 2)) + (1 - alpha.x) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                        offset.z + 1) - index(sdf_vals, offset.x + 0, offset.y + 2,
                                                                                                                                                                                                                offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                                                        offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                                                                offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 1, offset.z + 2))));
        index(dnormals_vals, 1, 1, 2).y = alpha.x * alpha.y * alpha.z * (index(sdf_vals,
                                          offset.x + 1, offset.y + 0, offset.z + 1) - index(sdf_vals, offset.x + 1,
                                                  offset.y + 2, offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                          offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                  offset.z + 1) + index(sdf_vals, offset.x + 2, offset.y + 1, offset.z + 1));
        index(dnormals_vals, 1, 2, -1).y = 0;
        index(dnormals_vals, 1, 2,
              0).y = -alpha.x * alpha.y * (1 - alpha.z) * (index(sdf_vals, offset.x + 0,
                      offset.y + 1, offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                              offset.z + 0) + index(sdf_vals, offset.x + 2, offset.y + 1,
                                      offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y + 1,
                                              offset.z - 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                      offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1, offset.z + 1));
        index(dnormals_vals, 1, 2,
              1).y = -alpha.x * alpha.y * alpha.z * (index(sdf_vals, offset.x + 0,
                      offset.y + 1, offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                              offset.z + 1) + index(sdf_vals, offset.x + 2, offset.y + 1,
                                      offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 1,
                                              offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                      offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1, offset.z + 2));
        index(dnormals_vals, 1, 2, 2).y = 0;
        index(dnormals_vals, 2, -1, -1).y = 0;
        index(dnormals_vals, 2, -1, 0).y = 0;
        index(dnormals_vals, 2, -1, 1).y = 0;
        index(dnormals_vals, 2, -1, 2).y = 0;
        index(dnormals_vals, 2, 0, -1).y = 0;
        index(dnormals_vals, 2, 0,
              0).y = alpha.x * (1 - alpha.y) * (1 - alpha.z) * (index(sdf_vals, offset.x + 1,
                      offset.y - 1, offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 1,
                              offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                offset.z - 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                        offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 0, offset.z + 1));
        index(dnormals_vals, 2, 0,
              1).y = alpha.x * alpha.z * (1 - alpha.y) * (index(sdf_vals, offset.x + 1,
                      offset.y - 1, offset.z + 1) - index(sdf_vals, offset.x + 1, offset.y + 1,
                              offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                        offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 0, offset.z + 2));
        index(dnormals_vals, 2, 0, 2).y = 0;
        index(dnormals_vals, 2, 1, -1).y = 0;
        index(dnormals_vals, 2, 1,
              0).y = alpha.x * alpha.y * (1 - alpha.z) * (index(sdf_vals, offset.x + 1,
                      offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 2,
                              offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y + 1,
                                                offset.z - 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                        offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1, offset.z + 1));
        index(dnormals_vals, 2, 1, 1).y = alpha.x * alpha.y * alpha.z * (index(sdf_vals,
                                          offset.x + 1, offset.y + 0, offset.z + 1) - index(sdf_vals, offset.x + 1,
                                                  offset.y + 2, offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 1,
                                                          offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                  offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1, offset.z + 2));
        index(dnormals_vals, 2, 1, 2).y = 0;
        index(dnormals_vals, 2, 2, -1).y = 0;
        index(dnormals_vals, 2, 2, 0).y = 0;
        index(dnormals_vals, 2, 2, 1).y = 0;
        index(dnormals_vals, 2, 2, 2).y = 0;
        index(dnormals_vals, -1, -1, -1).z = 0;
        index(dnormals_vals, -1, -1, 0).z = 0;
        index(dnormals_vals, -1, -1, 1).z = 0;
        index(dnormals_vals, -1, -1, 2).z = 0;
        index(dnormals_vals, -1, 0, -1).z = 0;
        index(dnormals_vals, -1, 0,
              0).z = (1 - alpha.x) * (1 - alpha.y) * (1 - alpha.z) * (index(sdf_vals,
                      offset.x + 0, offset.y + 0, offset.z - 1) - index(sdf_vals, offset.x + 0,
                              offset.y + 0, offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y - 1,
                                      offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                              offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 1, offset.z + 0));
        index(dnormals_vals, -1, 0,
              1).z = alpha.z * (1 - alpha.x) * (1 - alpha.y) * (index(sdf_vals, offset.x + 0,
                      offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 0, offset.y + 0,
                              offset.z + 2)) * (index(sdf_vals, offset.x + 0, offset.y - 1,
                                                offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                        offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 1, offset.z + 1));
        index(dnormals_vals, -1, 0, 2).z = 0;
        index(dnormals_vals, -1, 1, -1).z = 0;
        index(dnormals_vals, -1, 1,
              0).z = alpha.y * (1 - alpha.x) * (1 - alpha.z) * (index(sdf_vals, offset.x + 0,
                      offset.y + 1, offset.z - 1) - index(sdf_vals, offset.x + 0, offset.y + 1,
                              offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                        offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 2, offset.z + 0));
        index(dnormals_vals, -1, 1,
              1).z = alpha.y * alpha.z * (1 - alpha.x) * (index(sdf_vals, offset.x + 0,
                      offset.y + 1, offset.z + 0) - index(sdf_vals, offset.x + 0, offset.y + 1,
                              offset.z + 2)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                        offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 2, offset.z + 1));
        index(dnormals_vals, -1, 1, 2).z = 0;
        index(dnormals_vals, -1, 2, -1).z = 0;
        index(dnormals_vals, -1, 2, 0).z = 0;
        index(dnormals_vals, -1, 2, 1).z = 0;
        index(dnormals_vals, -1, 2, 2).z = 0;
        index(dnormals_vals, 0, -1, -1).z = 0;
        index(dnormals_vals, 0, -1,
              0).z = (1 - alpha.x) * (1 - alpha.y) * (1 - alpha.z) * (index(sdf_vals,
                      offset.x + 0, offset.y + 0, offset.z - 1) - index(sdf_vals, offset.x + 0,
                              offset.y + 0, offset.z + 1)) * (index(sdf_vals, offset.x - 1, offset.y + 0,
                                      offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                              offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 0, offset.z + 0));
        index(dnormals_vals, 0, -1,
              1).z = alpha.z * (1 - alpha.x) * (1 - alpha.y) * (index(sdf_vals, offset.x + 0,
                      offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 0, offset.y + 0,
                              offset.z + 2)) * (index(sdf_vals, offset.x - 1, offset.y + 0,
                                                offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                        offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 0, offset.z + 1));
        index(dnormals_vals, 0, -1, 2).z = 0;
        index(dnormals_vals, 0, 0,
              -1).z = (1 - alpha.x) * (1 - alpha.y) * (1 - alpha.z) * (index(sdf_vals,
                      offset.x - 1, offset.y + 0, offset.z + 0) + 2 * index(sdf_vals, offset.x + 0,
                              offset.y + 0, offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 0,
                                      offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y - 1,
                                              offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                      offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 1, offset.z + 0));
        index(dnormals_vals, 0, 0,
              0).z = alpha.z * (1 - alpha.x) * (1 - alpha.y) * (index(sdf_vals, offset.x - 1,
                      offset.y + 0, offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                              offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 0,
                                      offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y - 1,
                                              offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                      offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 1,
                                                              offset.z + 1)) + (1 - alpha.z) * (alpha.y * (1 - alpha.x) * (index(sdf_vals,
                                                                      offset.x + 0, offset.y + 1, offset.z - 1) - index(sdf_vals, offset.x + 0,
                                                                              offset.y + 1, offset.z + 1)) * (index(sdf_vals, offset.x - 1, offset.y + 1,
                                                                                      offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                              offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                      offset.z + 0)) + (1 - alpha.y) * (alpha.x * (index(sdf_vals, offset.x + 1,
                                                                                                              offset.y + 0, offset.z - 1) - index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                      offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y - 1,
                                                                                                                              offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                      offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                              offset.z + 0)) + 2 * (1 - alpha.x) * (index(sdf_vals, offset.x + 0,
                                                                                                                                                      offset.y + 0, offset.z - 1) - index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                              offset.z + 1)) * (index(sdf_vals, offset.x - 1, offset.y + 0,
                                                                                                                                                                      offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                              offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                                      offset.z + 0)) + 2 * (1 - alpha.x) * (index(sdf_vals, offset.x + 0,
                                                                                                                                                                                              offset.y + 0, offset.z - 1) - index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                      offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y - 1,
                                                                                                                                                                                                              offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                      offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 1, offset.z + 0))));
        index(dnormals_vals, 0, 0,
              1).z = alpha.z * (alpha.y * (1 - alpha.x) * (index(sdf_vals, offset.x + 0,
                                offset.y + 1, offset.z + 0) - index(sdf_vals, offset.x + 0, offset.y + 1,
                                        offset.z + 2)) * (index(sdf_vals, offset.x - 1, offset.y + 1,
                                                offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                        offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                offset.z + 1)) + (1 - alpha.y) * (alpha.x * (index(sdf_vals, offset.x + 1,
                                                                        offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                offset.z + 2)) * (index(sdf_vals, offset.x + 1, offset.y - 1,
                                                                                        offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                        offset.z + 1)) + 2 * (1 - alpha.x) * (index(sdf_vals, offset.x + 0,
                                                                                                                offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                        offset.z + 2)) * (index(sdf_vals, offset.x - 1, offset.y + 0,
                                                                                                                                offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                        offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                offset.z + 1)) + 2 * (1 - alpha.x) * (index(sdf_vals, offset.x + 0,
                                                                                                                                                        offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                offset.z + 2)) * (index(sdf_vals, offset.x + 0, offset.y - 1,
                                                                                                                                                                        offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                        offset.z + 1)))) + (1 - alpha.y) * (1 - alpha.z) * (alpha.x - 1) * (index(
                                                                                                                                                                                                sdf_vals, offset.x - 1, offset.y + 0, offset.z + 0) + 2 * index(sdf_vals,
                                                                                                                                                                                                        offset.x + 0, offset.y + 0, offset.z + 0) + index(sdf_vals, offset.x + 1,
                                                                                                                                                                                                                offset.y + 0, offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y - 1,
                                                                                                                                                                                                                        offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                                offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 1, offset.z + 0));
        index(dnormals_vals, 0, 0,
              2).z = alpha.z * (1 - alpha.y) * (alpha.x - 1) * (index(sdf_vals, offset.x - 1,
                      offset.y + 0, offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                              offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 0,
                                      offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y - 1,
                                              offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                      offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 1, offset.z + 1));
        index(dnormals_vals, 0, 1,
              -1).z = alpha.y * (1 - alpha.x) * (1 - alpha.z) * (index(sdf_vals, offset.x - 1,
                      offset.y + 1, offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                              offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                      offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                              offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                      offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 2, offset.z + 0));
        index(dnormals_vals, 0, 1,
              0).z = alpha.y * alpha.z * (1 - alpha.x) * (index(sdf_vals, offset.x - 1,
                      offset.y + 1, offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                              offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                      offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                              offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                      offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 2,
                                                              offset.z + 1)) + (1 - alpha.z) * (alpha.y * (alpha.x * (index(sdf_vals,
                                                                      offset.x + 1, offset.y + 1, offset.z - 1) - index(sdf_vals, offset.x + 1,
                                                                              offset.y + 1, offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                      offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                              offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 2,
                                                                                                      offset.z + 0)) + 2 * (1 - alpha.x) * (index(sdf_vals, offset.x + 0,
                                                                                                              offset.y + 1, offset.z - 1) - index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                      offset.z + 1)) * (index(sdf_vals, offset.x - 1, offset.y + 1,
                                                                                                                              offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                      offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                              offset.z + 0)) + 2 * (1 - alpha.x) * (index(sdf_vals, offset.x + 0,
                                                                                                                                                      offset.y + 1, offset.z - 1) - index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                              offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                      offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                              offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 2,
                                                                                                                                                                                      offset.z + 0))) + (1 - alpha.x) * (1 - alpha.y) * (index(sdf_vals, offset.x + 0,
                                                                                                                                                                                              offset.y + 0, offset.z - 1) - index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                      offset.z + 1)) * (index(sdf_vals, offset.x - 1, offset.y + 0,
                                                                                                                                                                                                              offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                      offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 0, offset.z + 0)));
        index(dnormals_vals, 0, 1,
              1).z = alpha.y * (1 - alpha.z) * (alpha.x - 1) * (index(sdf_vals, offset.x - 1,
                      offset.y + 1, offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                              offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                      offset.z + 0)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                              offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                      offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 2,
                                                              offset.z + 0)) + alpha.z * (alpha.y * (alpha.x * (index(sdf_vals, offset.x + 1,
                                                                      offset.y + 1, offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                              offset.z + 2)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                      offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                              offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 2,
                                                                                                      offset.z + 1)) + 2 * (1 - alpha.x) * (index(sdf_vals, offset.x + 0,
                                                                                                              offset.y + 1, offset.z + 0) - index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                      offset.z + 2)) * (index(sdf_vals, offset.x - 1, offset.y + 1,
                                                                                                                              offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                      offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                              offset.z + 1)) + 2 * (1 - alpha.x) * (index(sdf_vals, offset.x + 0,
                                                                                                                                                      offset.y + 1, offset.z + 0) - index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                              offset.z + 2)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                      offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                              offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 2,
                                                                                                                                                                                      offset.z + 1))) + (1 - alpha.x) * (1 - alpha.y) * (index(sdf_vals, offset.x + 0,
                                                                                                                                                                                              offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                      offset.z + 2)) * (index(sdf_vals, offset.x - 1, offset.y + 0,
                                                                                                                                                                                                              offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                      offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 0, offset.z + 1)));
        index(dnormals_vals, 0, 1,
              2).z = alpha.y * alpha.z * (alpha.x - 1) * (index(sdf_vals, offset.x - 1,
                      offset.y + 1, offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                              offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                      offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                              offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                      offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 2, offset.z + 1));
        index(dnormals_vals, 0, 2, -1).z = 0;
        index(dnormals_vals, 0, 2,
              0).z = alpha.y * (1 - alpha.x) * (1 - alpha.z) * (index(sdf_vals, offset.x + 0,
                      offset.y + 1, offset.z - 1) - index(sdf_vals, offset.x + 0, offset.y + 1,
                              offset.z + 1)) * (index(sdf_vals, offset.x - 1, offset.y + 1,
                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                        offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1, offset.z + 0));
        index(dnormals_vals, 0, 2,
              1).z = alpha.y * alpha.z * (1 - alpha.x) * (index(sdf_vals, offset.x + 0,
                      offset.y + 1, offset.z + 0) - index(sdf_vals, offset.x + 0, offset.y + 1,
                              offset.z + 2)) * (index(sdf_vals, offset.x - 1, offset.y + 1,
                                                offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                        offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1, offset.z + 1));
        index(dnormals_vals, 0, 2, 2).z = 0;
        index(dnormals_vals, 1, -1, -1).z = 0;
        index(dnormals_vals, 1, -1,
              0).z = alpha.x * (1 - alpha.y) * (1 - alpha.z) * (index(sdf_vals, offset.x + 1,
                      offset.y + 0, offset.z - 1) - index(sdf_vals, offset.x + 1, offset.y + 0,
                              offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                        offset.z + 0) + index(sdf_vals, offset.x + 2, offset.y + 0, offset.z + 0));
        index(dnormals_vals, 1, -1,
              1).z = alpha.x * alpha.z * (1 - alpha.y) * (index(sdf_vals, offset.x + 1,
                      offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 0,
                              offset.z + 2)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                        offset.z + 1) + index(sdf_vals, offset.x + 2, offset.y + 0, offset.z + 1));
        index(dnormals_vals, 1, -1, 2).z = 0;
        index(dnormals_vals, 1, 0,
              -1).z = alpha.x * (1 - alpha.y) * (1 - alpha.z) * (index(sdf_vals, offset.x + 0,
                      offset.y + 0, offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                              offset.z + 0) + index(sdf_vals, offset.x + 2, offset.y + 0,
                                      offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y - 1,
                                              offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                      offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1, offset.z + 0));
        index(dnormals_vals, 1, 0,
              0).z = alpha.x * alpha.z * (1 - alpha.y) * (index(sdf_vals, offset.x + 0,
                      offset.y + 0, offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                              offset.z + 1) + index(sdf_vals, offset.x + 2, offset.y + 0,
                                      offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y - 1,
                                              offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                      offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                              offset.z + 1)) + (1 - alpha.z) * (alpha.x * alpha.y * (index(sdf_vals,
                                                                      offset.x + 1, offset.y + 1, offset.z - 1) - index(sdf_vals, offset.x + 1,
                                                                              offset.y + 1, offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                      offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                              offset.z + 0) + index(sdf_vals, offset.x + 2, offset.y + 1,
                                                                                                      offset.z + 0)) + (1 - alpha.y) * (2 * alpha.x * (index(sdf_vals, offset.x + 1,
                                                                                                              offset.y + 0, offset.z - 1) - index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                      offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                              offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                      offset.z + 0) + index(sdf_vals, offset.x + 2, offset.y + 0,
                                                                                                                                              offset.z + 0)) + 2 * alpha.x * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                      offset.z - 1) - index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                              offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y - 1,
                                                                                                                                                                      offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                              offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                                                      offset.z + 0)) + (1 - alpha.x) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                              offset.z - 1) - index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                      offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y - 1,
                                                                                                                                                                                                              offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                      offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 1, offset.z + 0))));
        index(dnormals_vals, 1, 0,
              1).z = -alpha.x * (1 - alpha.y) * (1 - alpha.z) * (index(sdf_vals, offset.x + 0,
                      offset.y + 0, offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                              offset.z + 0) + index(sdf_vals, offset.x + 2, offset.y + 0,
                                      offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y - 1,
                                              offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                      offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                              offset.z + 0)) + alpha.z * (alpha.x * alpha.y * (index(sdf_vals, offset.x + 1,
                                                                      offset.y + 1, offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                              offset.z + 2)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                      offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                              offset.z + 1) + index(sdf_vals, offset.x + 2, offset.y + 1,
                                                                                                      offset.z + 1)) + (1 - alpha.y) * (2 * alpha.x * (index(sdf_vals, offset.x + 1,
                                                                                                              offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                      offset.z + 2)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                              offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                      offset.z + 1) + index(sdf_vals, offset.x + 2, offset.y + 0,
                                                                                                                                              offset.z + 1)) + 2 * alpha.x * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                      offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                              offset.z + 2)) * (index(sdf_vals, offset.x + 1, offset.y - 1,
                                                                                                                                                                      offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                              offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                                                      offset.z + 1)) + (1 - alpha.x) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                              offset.z + 0) - index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                      offset.z + 2)) * (index(sdf_vals, offset.x + 0, offset.y - 1,
                                                                                                                                                                                                              offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                      offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 1, offset.z + 1))));
        index(dnormals_vals, 1, 0,
              2).z = -alpha.x * alpha.z * (1 - alpha.y) * (index(sdf_vals, offset.x + 0,
                      offset.y + 0, offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                              offset.z + 1) + index(sdf_vals, offset.x + 2, offset.y + 0,
                                      offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y - 1,
                                              offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                      offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1, offset.z + 1));
        index(dnormals_vals, 1, 1,
              -1).z = alpha.x * alpha.y * (1 - alpha.z) * (index(sdf_vals, offset.x + 0,
                      offset.y + 1, offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                              offset.z + 0) + index(sdf_vals, offset.x + 2, offset.y + 1,
                                      offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                              offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                      offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 2, offset.z + 0));
        index(dnormals_vals, 1, 1, 0).z = alpha.x * alpha.y * alpha.z * (index(sdf_vals,
                                          offset.x + 0, offset.y + 1, offset.z + 1) + 2 * index(sdf_vals, offset.x + 1,
                                                  offset.y + 1, offset.z + 1) + index(sdf_vals, offset.x + 2, offset.y + 1,
                                                          offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                  offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                          offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 2,
                                                                                  offset.z + 1)) + (1 - alpha.z) * (alpha.x * (1 - alpha.y) * (index(sdf_vals,
                                                                                          offset.x + 1, offset.y + 0, offset.z - 1) - index(sdf_vals, offset.x + 1,
                                                                                                  offset.y + 0, offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                          offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                  offset.z + 0) + index(sdf_vals, offset.x + 2, offset.y + 0,
                                                                                                                          offset.z + 0)) + alpha.y * (2 * alpha.x * (index(sdf_vals, offset.x + 1,
                                                                                                                                  offset.y + 1, offset.z - 1) - index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                          offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                  offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                          offset.z + 0) + index(sdf_vals, offset.x + 2, offset.y + 1,
                                                                                                                                                                  offset.z + 0)) + 2 * alpha.x * (index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                                          offset.z - 1) - index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                                                  offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                                          offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                                                                  offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 2,
                                                                                                                                                                                                          offset.z + 0)) + (1 - alpha.x) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                                                  offset.z - 1) - index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                                                          offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                                                  offset.z + 0) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                                                                          offset.z + 0) + index(sdf_vals, offset.x + 0, offset.y + 2, offset.z + 0))));
        index(dnormals_vals, 1, 1,
              1).z = -alpha.x * alpha.y * (1 - alpha.z) * (index(sdf_vals, offset.x + 0,
                      offset.y + 1, offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                              offset.z + 0) + index(sdf_vals, offset.x + 2, offset.y + 1,
                                      offset.z + 0)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                              offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                      offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 2,
                                                              offset.z + 0)) + alpha.z * (alpha.x * (1 - alpha.y) * (index(sdf_vals,
                                                                      offset.x + 1, offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 1,
                                                                              offset.y + 0, offset.z + 2)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                      offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                              offset.z + 1) + index(sdf_vals, offset.x + 2, offset.y + 0,
                                                                                                      offset.z + 1)) + alpha.y * (2 * alpha.x * (index(sdf_vals, offset.x + 1,
                                                                                                              offset.y + 1, offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                      offset.z + 2)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                              offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                      offset.z + 1) + index(sdf_vals, offset.x + 2, offset.y + 1,
                                                                                                                                              offset.z + 1)) + 2 * alpha.x * (index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                      offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                              offset.z + 2)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                                                                                                                                      offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                                                                                                                              offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 2,
                                                                                                                                                                                      offset.z + 1)) + (1 - alpha.x) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                              offset.z + 0) - index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                                      offset.z + 2)) * (index(sdf_vals, offset.x + 0, offset.y + 0,
                                                                                                                                                                                                              offset.z + 1) + 2 * index(sdf_vals, offset.x + 0, offset.y + 1,
                                                                                                                                                                                                                      offset.z + 1) + index(sdf_vals, offset.x + 0, offset.y + 2, offset.z + 1))));
        index(dnormals_vals, 1, 1,
              2).z = -alpha.x * alpha.y * alpha.z * (index(sdf_vals, offset.x + 0,
                      offset.y + 1, offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                              offset.z + 1) + index(sdf_vals, offset.x + 2, offset.y + 1,
                                      offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                              offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                      offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 2, offset.z + 1));
        index(dnormals_vals, 1, 2, -1).z = 0;
        index(dnormals_vals, 1, 2,
              0).z = alpha.x * alpha.y * (1 - alpha.z) * (index(sdf_vals, offset.x + 1,
                      offset.y + 1, offset.z - 1) - index(sdf_vals, offset.x + 1, offset.y + 1,
                              offset.z + 1)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                        offset.z + 0) + index(sdf_vals, offset.x + 2, offset.y + 1, offset.z + 0));
        index(dnormals_vals, 1, 2, 1).z = alpha.x * alpha.y * alpha.z * (index(sdf_vals,
                                          offset.x + 1, offset.y + 1, offset.z + 0) - index(sdf_vals, offset.x + 1,
                                                  offset.y + 1, offset.z + 2)) * (index(sdf_vals, offset.x + 0, offset.y + 1,
                                                          offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                  offset.z + 1) + index(sdf_vals, offset.x + 2, offset.y + 1, offset.z + 1));
        index(dnormals_vals, 1, 2, 2).z = 0;
        index(dnormals_vals, 2, -1, -1).z = 0;
        index(dnormals_vals, 2, -1, 0).z = 0;
        index(dnormals_vals, 2, -1, 1).z = 0;
        index(dnormals_vals, 2, -1, 2).z = 0;
        index(dnormals_vals, 2, 0, -1).z = 0;
        index(dnormals_vals, 2, 0,
              0).z = alpha.x * (1 - alpha.y) * (1 - alpha.z) * (index(sdf_vals, offset.x + 1,
                      offset.y + 0, offset.z - 1) - index(sdf_vals, offset.x + 1, offset.y + 0,
                              offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y - 1,
                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                        offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 1, offset.z + 0));
        index(dnormals_vals, 2, 0,
              1).z = alpha.x * alpha.z * (1 - alpha.y) * (index(sdf_vals, offset.x + 1,
                      offset.y + 0, offset.z + 0) - index(sdf_vals, offset.x + 1, offset.y + 0,
                              offset.z + 2)) * (index(sdf_vals, offset.x + 1, offset.y - 1,
                                                offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 0,
                                                        offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 1, offset.z + 1));
        index(dnormals_vals, 2, 0, 2).z = 0;
        index(dnormals_vals, 2, 1, -1).z = 0;
        index(dnormals_vals, 2, 1,
              0).z = alpha.x * alpha.y * (1 - alpha.z) * (index(sdf_vals, offset.x + 1,
                      offset.y + 1, offset.z - 1) - index(sdf_vals, offset.x + 1, offset.y + 1,
                              offset.z + 1)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                offset.z + 0) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                        offset.z + 0) + index(sdf_vals, offset.x + 1, offset.y + 2, offset.z + 0));
        index(dnormals_vals, 2, 1, 1).z = alpha.x * alpha.y * alpha.z * (index(sdf_vals,
                                          offset.x + 1, offset.y + 1, offset.z + 0) - index(sdf_vals, offset.x + 1,
                                                  offset.y + 1, offset.z + 2)) * (index(sdf_vals, offset.x + 1, offset.y + 0,
                                                          offset.z + 1) + 2 * index(sdf_vals, offset.x + 1, offset.y + 1,
                                                                  offset.z + 1) + index(sdf_vals, offset.x + 1, offset.y + 2, offset.z + 1));
        index(dnormals_vals, 2, 1, 2).z = 0;
        index(dnormals_vals, 2, 2, -1).z = 0;
        index(dnormals_vals, 2, 2, 0).z = 0;
        index(dnormals_vals, 2, 2, 1).z = 0;
        index(dnormals_vals, 2, 2, 2).z = 0;
    }
}

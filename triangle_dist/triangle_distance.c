#include "stdio.h"
#include "math.h"
#include <float.h>
#include "smmintrin.h"

typedef float v4sf __attribute__((vector_size(16)));

union vec4 {
    v4sf v;
    float e[4];
};

inline float min(const float x, const float y) {
    return x < y ? x : y;
}

// power smooth min (k = 8);
// http://www.iquilezles.org/www/articles/smin/smin.htm
inline float smin(float a, float b, float k) {
    a = pow(a, k);
    b = pow(b, k);
    return pow((a * b) / (a + b), 1.0 / k);
}

inline float max(const float x, const float y) {
    return x > y ? x : y;
}

inline union vec4 make_vec(const float x, const float y, const float z,
                               const float w) {
    union vec4 new_vec;
    new_vec.e[0] = x;
    new_vec.e[1] = y;
    new_vec.e[2] = z;
    new_vec.e[3] = w;
    return new_vec;
}

inline union vec4 tv4(const v4sf l) {
    union vec4 vec;
    vec.v = l;
    return vec;
}

inline float dot(const v4sf v1, const v4sf v2) {
    __m128 dotted = __builtin_ia32_dpps(v1, v2, 0xFF);

    float result;
    _mm_store_ss(&result, dotted);
    return result;
}

inline float dot2(const v4sf v1) {
    return dot(v1, v1);
}

inline float sign(const float val) {
    return (signbit(val)) ? -1.0 : ((val > 0.0) ? 1.0 : 0.0);
}

// https://stackoverflow.com/questions/427477/fastest-way-to-clamp-a-real-fixed-floating-point-value
inline float clamp(float d, float min, float max) {
    const float t = d < min ? min : d;
    return t > max ? max : t;
}

// https://fastcpp.blogspot.com/2011/04/vector-cross-product-using-sse-code.html
inline v4sf cross(const v4sf a, const v4sf b) {
    return _mm_sub_ps(
               _mm_mul_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1)),
                          _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 0, 2))),
               _mm_mul_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2)),
                          _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1)))
           );
}

void print_vec(const union vec4 vec) {
    printf("(%f, %f, %f, %f)\n", vec.e[0], vec.e[1], vec.e[2], vec.e[3]);
}

// http://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
float udTriangle(const v4sf p, const v4sf a, const v4sf b, const v4sf c) {
    v4sf ba = b - a;
    v4sf pa = p - a;
    v4sf cb = c - b;
    v4sf pb = p - b;
    v4sf ac = a - c;
    v4sf pc = p - c;

    v4sf nor = cross(ba, ac);

    return
        sqrt((sign(dot(cross(ba, nor), pa)) +
              sign(dot(cross(cb, nor), pb)) +
              sign(dot(cross(ac, nor), pc)) < 2.0) ?
             min(min(dot2(ba * clamp(dot(ba, pa) / dot2(ba), 0.0, 1.0) - pa),
                     dot2(cb * clamp(dot(cb, pb) / dot2(cb), 0.0, 1.0) - pb)),
                 dot2(ac * clamp(dot(ac, pc) / dot2(ac), 0.0, 1.0) - pc)) :
             dot(nor, pa) * dot(nor, pa) / dot2(nor));
}

float sdTriangle(const v4sf p, const v4sf a, const v4sf b, const v4sf c,
                 const v4sf n) {
    v4sf ba = b - a;
    v4sf pa = p - a;
    v4sf cb = c - b;
    v4sf pb = p - b;
    v4sf ac = a - c;
    v4sf pc = p - c;

    v4sf nor = cross(ba, ac);

    return sign(dot(n, pa)) *                // Note, we could use pb or pc here too
           sqrt((sign(dot(cross(ba, nor), pa)) +
                 sign(dot(cross(cb, nor), pb)) +
                 sign(dot(cross(ac, nor), pc)) < 2.0) ?
                min(min(dot2(ba * clamp(dot(ba, pa) / dot2(ba), 0.0, 1.0) - pa),
                        dot2(cb * clamp(dot(cb, pb) / dot2(cb), 0.0, 1.0) - pb)),
                    dot2(ac * clamp(dot(ac, pc) / dot2(ac), 0.0, 1.0) - pc)) :
                dot(nor, pa) * dot(nor, pa) / dot2(nor));
}

v4sf vecTriangle(const v4sf p, const v4sf a, const v4sf b, const v4sf c) {
    v4sf ba = b - a;
    v4sf pa = p - a;
    v4sf cb = c - b;
    v4sf pb = p - b;
    v4sf ac = a - c;
    v4sf pc = p - c;

    v4sf nor = cross(ba, ac);

    if (sign(dot(cross(ba, nor), pa)) +
            sign(dot(cross(cb, nor), pb)) +
            sign(dot(cross(ac, nor), pc)) < 2.0) {
        v4sf eab = clamp(dot(pa, ba) / dot2(ba), 0.0, 1.0) * ba - pa;
        v4sf eac = clamp(dot(pa, ac) / dot2(ac), 0.0, 1.0) * ac - pc;
        v4sf ecb = clamp(dot(pc, cb) / dot2(cb), 0.0, 1.0) * cb - pb;

        if (dot2(eab) < dot2(eac)) {
            if (dot2(eab) < dot2(ecb)) {
                return eab;
            } else {
                return ecb;
            }
        } else {
            if (dot2(eac) < dot2(ecb)) {
                return eac;
            } else {
                return ecb;
            }
        }
    } else {
        return dot(pa, nor) * nor;
    }
}

float closest_intersection(const float* position,
                           const float* vertices_x,
                           const float* vertices_y,
                           const float* vertices_z,
                           const int* indices, const size_t num_faces) {
    union vec4 p = make_vec(position[0], position[1], position[2], 0.0);
    float best_dist = FLT_MAX;
    float dist_val = 0.0;
    for (size_t index = 0; index < num_faces; index++) {
        int i1 = indices[3 * index];
        int i2 = indices[3 * index + 1];
        int i3 = indices[3 * index + 2];

        union vec4 v1 = make_vec(vertices_x[i1], vertices_y[i1], vertices_z[i1], 0.0);
        union vec4 v2 = make_vec(vertices_x[i2], vertices_y[i2], vertices_z[i2], 0.0);
        union vec4 v3 = make_vec(vertices_x[i3], vertices_y[i3], vertices_z[i3], 0.0);
        v4sf nv = cross(v1.v - v2.v, v1.v - v3.v);

        //best_dist = min(best_dist, sdTriangle(p.v, v1.v, v2.v, v3.v, nv));
        float new_dist = sdTriangle(p.v, v1.v, v2.v, v3.v, nv);
        if (fabsf(new_dist) < best_dist) {
            best_dist = fabsf(new_dist);
            //if (sign(dist_val) != sign(new_dist)) {
            dist_val = new_dist;
            //} else {
            //dist_val = sign(new_dist) *
            //smin(fabsf(dist_val), fabsf(new_dist), 10);
            //}
        }
    }
    return dist_val;
}

int main() {
    union vec4 v1 = make_vec(1.0, 0.0, 0.0, 0.0);
    union vec4 v2 = make_vec(0.0, 1.0, 0.0, 0.0);
    union vec4 v3 = make_vec(0.0, 0.0, 0.0, 0.0);
    union vec4 p = make_vec(-1.0, -1.0, 0.0, 0.0);
    union vec4 n = make_vec(0.0, 0.0, -1.0, 0.0);

    //print_vec(tv4(cross(v1.v, v2.v)));
    //printf("Dist: %f", udTriangle(p.v, v1.v, v2.v, v3.v));
    //printf("Dist: %f", sdTriangle(p.v, v1.v, v2.v, v3.v, n.v));
    print_vec(tv4(vecTriangle(p.v, v1.v, v2.v, v3.v)));
}

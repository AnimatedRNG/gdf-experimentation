#include <iostream>
#include <tuple>

#include "Halide.h"

#include "matmul.hpp"

#include <stdio.h>

using namespace Halide;

Var x("x"), y("y"), c("c"), t("t");
Var dx("dx"), dy("dy"), dz("dz");

typedef struct {
    Halide::Buffer<float> buffer;
    float x0, y0, z0;
    float x1, y1, z1;
    int nx, ny, nz;
} GridSDF;

// For debugging analytical functions
GridSDF to_grid_sdf(std::function<Expr(Tuple)> sdf, float x0, float y0,
                    float z0, float x1, float y1,
                    float z1, int nx, int ny, int nz) {
    Func field_func("field_func");
    field_func(dx, dy, dz) = sdf({
        (dx / (float) nx) * (x1 - x0) + x0,
        (dy / (float) ny) * (y1 - y0) + y0,
        (dz / (float) nz) * (z1 - z0) + z0
    });
    Halide::Buffer<float> buffer = field_func.realize(nx, ny, nz);
    return GridSDF {
        buffer, x0, y0, z0, x1, y1, z1, nx, ny, nz
    };
}

// effectively converts a GridSDF into a regular SDF
Expr trilinear(const GridSDF& sdf, Tuple position) {
    Tuple grid_space = {
        ((position[0] - sdf.x0) / (sdf.x1 - sdf.x0))* ((float) sdf.nx),
        ((position[1] - sdf.y0) / (sdf.y1 - sdf.y0))* ((float) sdf.ny),
        ((position[2] - sdf.z0) / (sdf.z1 - sdf.z0))* ((float) sdf.nz)
    };

    // floor and ceil slow?
    Tuple lp = {
        clamp(cast<int32_t>(Halide::floor(grid_space[0])), 0, sdf.nx - 1),
        clamp(cast<int32_t>(Halide::floor(grid_space[1])), 0, sdf.ny - 1),
        clamp(cast<int32_t>(Halide::floor(grid_space[2])), 0, sdf.nz - 1),
    };

    Tuple up = {
        clamp(cast<int32_t>(Halide::ceil(grid_space[0])), 0, sdf.nx - 1),
        clamp(cast<int32_t>(Halide::ceil(grid_space[1])), 0, sdf.ny - 1),
        clamp(cast<int32_t>(Halide::ceil(grid_space[2])), 0, sdf.nz - 1),
    };

    Tuple alpha = {
        grid_space[0] - lp[0],
        grid_space[1] - lp[1],
        grid_space[2] - lp[2],
    };

    Expr c000 = sdf.buffer(lp[0], lp[1], lp[2]);
    Expr c001 = sdf.buffer(lp[0], lp[1], up[2]);
    Expr c010 = sdf.buffer(lp[0], up[1], lp[2]);
    Expr c011 = sdf.buffer(lp[0], up[1], up[2]);
    Expr c100 = sdf.buffer(up[0], lp[1], lp[2]);
    Expr c101 = sdf.buffer(up[0], lp[1], up[2]);
    Expr c110 = sdf.buffer(up[0], up[1], lp[2]);
    Expr c111 = sdf.buffer(up[0], up[1], up[2]);

    // interpolate on x
    Expr c00 = Halide::lerp(c000, c100, alpha[0]);
    Expr c01 = Halide::lerp(c001, c101, alpha[0]);
    Expr c10 = Halide::lerp(c010, c110, alpha[0]);
    Expr c11 = Halide::lerp(c011, c111, alpha[0]);

    // interpolate on y
    Expr c0 = Halide::lerp(c00, c10, alpha[1]);
    Expr c1 = Halide::lerp(c01, c11, alpha[1]);

    // interpolate on z
    Expr c = Halide::lerp(c0, c1, alpha[2]);

    return c;
}

void apply_auto_schedule(Func F) {
    std::map<std::string, Internal::Function> flist =
        Internal::find_transitive_calls(
            F.function());
    flist.insert(std::make_pair(F.name(), F.function()));
    std::map<std::string, Internal::Function>::iterator fit;
    for (fit = flist.begin(); fit != flist.end(); fit++) {
        Func f(fit->second);
        f.compute_root();
        std::cout << "Warning: applying default schedule to " << f.name() << std::endl;
    }
    std::cout << std::endl;
}

Expr example_sphere(Tuple position) {
    return Halide::sqrt(
               position[0] * position[0] +
               position[1] * position[1] +
               position[2] * position[2]) - 3.0f;
}

class TracerGenerator : public Halide::Generator<TracerGenerator> {
  public:

    Input<Buffer<float>> projection_{"projection", 2};
    Input<Buffer<float>> view_{"view", 2};
    Input<int32_t> width{"width"};
    Input<int32_t> height{"height"};
    Output<Buffer<float>> out_{"out", 3};
    Output<Buffer<float>> debug_{"debug", 2};

    std::tuple<std::tuple<Func, Func>, Func> projection(Func proj_matrix,
            Func view_matrix,
            float near = 0.1f) {
        // TODO: Get rid of scheduling calls
        Func rays("rays");

        Func ss_norm;
        ss_norm(x, y) = {
            cast<float>(x) / (cast<float>(width)),
            cast<float>(y) / (cast<float>(height))
        };

        Func clip_space("clip_space");
        clip_space(x, y, c) = 1.0f;
        clip_space(x, y, 0) = ss_norm(x, y)[0] * 2.0f - 1.0f;
        clip_space(x, y, 1) = ss_norm(x, y)[1] * 2.0f - 1.0f;
        clip_space.bound(c, 0, 4).unroll(c);

        Func viewproj_inv("viewproj_inv");
        viewproj_inv =
            matmul::inverse(matmul::product(proj_matrix, view_matrix));

        Func view_inv("view_inv");
        view_inv =
            matmul::inverse(view_matrix);
        debug_(x, y) = 0.0f;

        RDom k(0, 4);
        Func homogeneous("homogeneous");
        homogeneous(x, y, c) = sum(viewproj_inv(c, k) *
                                   clip_space(x, y, k));
        homogeneous.bound(c, 0, 4).unroll(c);
        viewproj_inv.compute_at(homogeneous, x);

        Func origin("origin");
        origin(c) = view_inv(c, 3);
        origin.bound(c, 0, 4).unroll(c);

        Func projected("projected");
        projected(x, y, c) = (homogeneous(x, y, c) / homogeneous(x, y, 3))
                             - origin(c);
        projected.bound(c, 0, 4).unroll(c);
        homogeneous.compute_at(projected, x);
        origin.compute_at(projected, x);

        Func ray_vec("ray_vec");
        RDom norm_k(0, 3);
        // could use fast inverse sqrt, but not worth accuracy loss
        Expr ray_vec_norm = Halide::sqrt(sum(
                                             projected(x, y, norm_k) *
                                             projected(x, y, norm_k)));
        ray_vec(x, y) = Tuple(projected(x, y, 0) / ray_vec_norm,
                              projected(x, y, 1) / ray_vec_norm,
                              projected(x, y, 2) / ray_vec_norm);
        projected.compute_at(ray_vec, x);

        Func ray_pos("ray_pos");
        //ray_pos(x, y, c) = origin(c) + ray_vec(x, y, c) * near;
        ray_pos(x, y) = Tuple(origin(0) + ray_vec(x, y)[0] * near,
                              origin(1) + ray_vec(x, y)[1] * near,
                              origin(2) + ray_vec(x, y)[2] * near);

        //Func projection_result("projection_result");
        //projection_result(x, y) = {ray_pos(x, y), ray_vec(x, y)};
        //ray_vec.compute_at(projection_result, x);
        //ray_pos.compute_at(projection_result, x);

        //apply_auto_schedule(projection_result);

        std::tuple<Func, Func> projection_result(ray_pos, ray_vec);
        apply_auto_schedule(ray_pos);
        apply_auto_schedule(ray_vec);

        return std::make_tuple(projection_result, origin);
    }

    Expr normal_pdf(Expr x, float sigma = 1e-7f, float mean = 0.0f) {
        return (1.0f / Halide::sqrt(2.0f * (float) M_PI * sigma * sigma)) *
               Halide::exp((x - mean) * (x - mean) / (-2.0f * sigma * sigma));
    }

    Expr relu(Expr a) {
        return Halide::max(a, 0);
    }

    Expr normal_pdf_rectified(Expr x, float sigma = 1e-2f, float mean = 0.0f) {
        return normal_pdf(relu(x), sigma, mean);
    }

    Func sphere_trace(std::function<Expr(Tuple)> sdf, int iterations = 300,
                      float EPS = 1e-6) {
        Func original_ray_pos("original_ray_pos");
        Func ray_vec("ray_vec");
        Func origin("origin");

        std::forward_as_tuple(std::tie(original_ray_pos,
                                       ray_vec), origin) =
                                           projection(projection_, view_);
        RDom tr(0, iterations);
        Func pos("pos");
        Expr d("d");
        Func depth("depth");

        // debug
        GridSDF grid_sdf = to_grid_sdf(example_sphere,
                                       -4.0, -4.0, -4.0, 4.0, 4.0, 4.0, 32, 32, 32);

        // Remember how update definitions work
        pos(x, y, t) = Tuple(0.0f, 0.0f, 0.0f);
        pos(x, y, 0) = original_ray_pos(x, y);
        //d = sdf(pos(x, y, tr));
        d = trilinear(grid_sdf, pos(x, y, tr));
        pos(x, y, tr + 1) = Tuple(
                                pos(x, y, tr)[0] + d * ray_vec(x, y)[0],
                                pos(x, y, tr)[1] + d * ray_vec(x, y)[1],
                                pos(x, y, tr)[2] + d * ray_vec(x, y)[2]);
        //pos.trace_stores();
        Var xi, xo, yi, yo;
        depth(x, y, t) = 0.0f;
        depth(x, y, tr + 1) = d;

        Func endpoint("endpoint");
        endpoint(x, y) = pos(x, y, iterations);
        /*endpoint(x, y) = {depth(x, y, iterations),
                          depth(x, y, iterations),
                          depth(x, y, iterations)
                          };*/
        endpoint.compute_root();
        pos.unroll(t);
        pos.compute_at(endpoint, x);
        pos.store_at(endpoint, x);

        return endpoint;
    }

    void generate() {
        Func original_ray_pos("original_ray_pos");
        Func original_ray_vec("original_ray_vec");
        Func origin("origin");

        std::forward_as_tuple(std::tie(original_ray_pos,
                                       original_ray_vec), origin) =
                                           projection(projection_, view_);
        original_ray_pos.compute_root();
        original_ray_vec.compute_root();

        Func end("end");
        end(x, y) = sphere_trace(example_sphere)(x, y);
        GridSDF grid_sdf = to_grid_sdf(example_sphere,
                                       -4.0, -4.0, -4.0, 4.0, 4.0, 4.0, 32, 32, 32);

        /*Func val("val");
        val(dx) = trilinear(grid_sdf, {-3.5f, -3.5f, -3.5f});
        std::cout << "at (-3.5, -3.5, -3.5) " << Buffer<float>(val.realize(1))(0) <<
        std::endl;*/

        //out_(x, y, c) =
        //    matmul::product(matmul::product(Func(b), 3.0f), Func(b))(x, y);
        //out_(x, y, c) = matmul::inverse(Func(a))(x, y);
        //out_(x, y, c) = clamp(rays(x, y, 2 - c)[1], 0.0f, 1.0f);
        //out_(x, y, c) = rays(x, y, c)[1];

        out_(x, y, c) = 0.0f;
        out_(x, y, 0) = clamp(end(x, y)[0], 0.0f, 1.0f);
        out_(x, y, 1) = clamp(end(x, y)[1], 0.0f, 1.0f);
        out_(x, y, 2) = clamp(end(x, y)[2], 0.0f, 1.0f);
    }
};

HALIDE_REGISTER_GENERATOR(TracerGenerator, tracer_render)

#include <iostream>
#include <tuple>

#include "Halide.h"

#include "matmul.hpp"

#include <stdio.h>

using namespace Halide;

constexpr static int iterations = 300;

Var x("x"), y("y"), c("c"), t("t");
Var dx("dx"), dy("dy"), dz("dz");
RDom tr;

typedef struct {
    Halide::Buffer<float> buffer;
    TupleVec<3> p0;
    TupleVec<3> p1;
    TupleVec<3> n;
} GridSDF;

// For debugging analytical functions
GridSDF to_grid_sdf(std::function<Expr(TupleVec<3>)> sdf,
                    TupleVec<3> p0,
                    TupleVec<3> p1,
                    int nx, int ny, int nz) {
    Func field_func("field_func");
    field_func(dx, dy, dz) = sdf(TupleVec<3>({
        (dx / cast<float>(nx)) * (p1[0] - p0[0]) + p0[0],
        (dy / cast<float>(ny)) * (p1[1] - p0[1]) + p0[1],
        (dz / cast<float>(nz)) * (p1[2] - p0[2]) + p0[2]
    }));
    Halide::Buffer<float> buffer = field_func.realize(nx, ny, nz);

    return GridSDF {
        buffer, p0, p1, TupleVec<3>({nx, ny, nz})
    };
}

// effectively converts a GridSDF into a regular SDF
Expr trilinear(const GridSDF& sdf, TupleVec<3> position) {
    TupleVec<3> grid_space = ((position - sdf.p0) / (sdf.p1 - sdf.p0)) *
                             (cast<float>(sdf.n));

    // floor and ceil slow?
    TupleVec<3> lp = build<3>([grid_space, sdf](unsigned int i) {
        return clamp(cast<int32_t>(grid_space[i]), 0, sdf.n[i] - 1);
    });

    TupleVec<3> up = {
        clamp(cast<int32_t>(Halide::ceil(grid_space[0])), 0, sdf.n[0] - 1),
        clamp(cast<int32_t>(Halide::ceil(grid_space[1])), 0, sdf.n[1] - 1),
        clamp(cast<int32_t>(Halide::ceil(grid_space[2])), 0, sdf.n[2] - 1),
    };

    // why won't this work?
    /*Tuple up = {
        clamp(lp[0] + 1, 0, sdf.nx - 1),
        clamp(lp[1] + 1, 0, sdf.ny - 1),
        clamp(lp[2] + 1, 0, sdf.nz - 1)
        };*/

    /*TupleVec<3> alpha = {
        grid_space[0] - lp[0],
        grid_space[1] - lp[1],
        grid_space[2] - lp[2],
        };*/
    TupleVec<3> alpha = grid_space - lp;

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

/*Expr example_sphere(Tuple position) {
    return Halide::sqrt(
               position[0] * position[0] +
               position[1] * position[1] +
               position[2] * position[2]) - 3.0f;
               }*/

Expr example_sphere(TupleVec<3> position) {
    return norm(position) - 3.0f;
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
        Expr ray_vec_norm = Halide::sqrt(
                                sum(
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

    Func light_source(Tuple light_color,
                      Func positions,
                      Tuple light_position,
                      Func normals,
                      float kd = 0.7f,
                      float ks = 0.3f,
                      float ka = 100.0f) {
        Func light_vec("light_vec");
        light_vec(x, y, tr) = {
            light_position[0] - positions(x, y, tr)[0],
            light_position[1] - positions(x, y, tr)[1],
            light_position[2] - positions(x, y, tr)[2]
        };

        Func light_vec_norm("light_vec_norm");
        light_vec_norm(x, y, tr) =
            Halide::sqrt(light_vec(x, y, tr)[0] +
                         light_vec(x, y, tr)[1] +
                         light_vec(x, y, tr)[2]);

        Func light_vec_normalized("light_vec_normalized");
        light_vec_normalized(x, y, tr) = {
            light_vec(x, y, tr)[0] / light_vec_norm(x, y, tr),
            light_vec(x, y, tr)[1] / light_vec_norm(x, y, tr),
            light_vec(x, y, tr)[2] / light_vec_norm(x, y, tr),
        };

        Func ray_position("ray_position");
        ray_position(x, y, tr) = {
            positions(x, y, tr)[0] + light_vec(x, y, tr)[0],
            positions(x, y, tr)[1] + light_vec(x, y, tr)[1],
            positions(x, y, tr)[2] + light_vec(x, y, tr)[2],
        };

        Func diffuse("diffuse");

    }

    Func sphere_trace(std::function<Expr(Tuple)> sdf,
                      float EPS = 1e-6) {
        Func original_ray_pos("original_ray_pos");
        Func ray_vec("ray_vec");
        Func origin("origin");

        std::forward_as_tuple(std::tie(original_ray_pos,
                                       ray_vec), origin) =
                                           projection(projection_, view_);
        Func pos("pos");
        Expr d("d");
        Func depth("depth");

        // debug
        GridSDF grid_sdf = to_grid_sdf(example_sphere,
                                       TupleVec<3>({-4.0f, -4.0f, -4.0f}),
                                       TupleVec<3>({4.0f, 4.0f, 4.0f}),
                                       32, 32, 32);

        // Remember how update definitions work
        pos(x, y, t) = Tuple(0.0f, 0.0f, 0.0f);
        pos(x, y, 0) = original_ray_pos(x, y);
        //d = sdf(pos(x, y, tr));
        d = trilinear(grid_sdf, TupleVec<3>(Tuple(pos(x, y, tr))));
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
        tr = RDom(0, iterations);

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
        {-4.0f, -4.0f, -4.0f},
        {4.0f, 4.0f, 4.0f}, 32, 32, 32);

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

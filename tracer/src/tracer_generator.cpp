#include <iostream>
#include <tuple>

#include "Halide.h"

#include "matmul.hpp"

#include <stdio.h>

using namespace Halide;

Var x("x"), y("y"), c("c"), t("t");

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

    Func sphere_trace(/*Func sdf, */size_t iterations = 300, float EPS = 1e-6) {
        Func original_ray_pos("original_ray_pos");
        Func ray_vec("ray_vec");
        Func origin("origin");

        std::forward_as_tuple(std::tie(original_ray_pos,
                                       ray_vec), origin) =
                                           projection(projection_, view_);
        RDom tr(0, 300);
        Func pos("pos");
        Expr d("d");
        Func depth("depth");

        // Remember how update definitions work
        pos(x, y, t) = Tuple(0.0f, 0.0f, 0.0f);
        pos(x, y, 0) = original_ray_pos(x, y);
        d = Halide::sqrt(
                pos(x, y, tr)[0] * pos(x, y, tr)[0] +
                pos(x, y, tr)[1] * pos(x, y, tr)[1] +
                pos(x, y, tr)[2] * pos(x, y, tr)[2]) - 3.0f;
        pos(x, y, tr + 1) = Tuple(
                             pos(x, y, tr)[0] + d * ray_vec(x, y)[0],
                             pos(x, y, tr)[1] + d * ray_vec(x, y)[1],
                             pos(x, y, tr)[2] + d * ray_vec(x, y)[2]);
        Var xi, xo, yi, yo;
        //depth(x, y, t) = 0.0f;
        //depth(x, y, tr + 1) = d;
        //original_ray_pos.compute_at(pos, x);

        Func endpoint("endpoint");
        endpoint(x, y) = pos(x, y, (int) iterations);
        /*endpoint(x, y) = {depth(x, y, 300),
                          depth(x, y, 300),
                          depth(x, y, 300)};*/
        endpoint.compute_root();
        pos.compute_at(endpoint, x);

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

        Func sphere_sdf("sphere_sdf");
        float radius = 3.0f;

        /*Expr xf("xf"), yf("yf"), zf("zf");
        sphere_sdf(xf, yf, zf) =
            Halide::sqrt(
            xf * xf + yf * yf + zf * zf) - radius;*/

        Func end("end");
        end(x, y) = sphere_trace()(x, y);

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

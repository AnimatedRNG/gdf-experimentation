#include <iostream>

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

    std::tuple<Func, Func> projection(Func proj_matrix,
                                      Func view_matrix,
                                      float near = 0.1f) {
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
        ray_vec(x, y, c) = projected(x, y, c) / Halide::sqrt(sum(
                               projected(x, y, norm_k) *
                               projected(x, y, norm_k)));
        ray_vec.bound(c, 0, 4).unroll(c);
        projected.compute_at(ray_vec, x);

        Func ray_pos("ray_pos");
        ray_pos(x, y, c) = origin(c) + ray_vec(x, y, c) * near;

        Func projection_result("projection_result");
        projection_result(x, y, c) = {ray_pos(x, y, c), ray_vec(x, y, c)};
        ray_vec.compute_at(projection_result, x);
        ray_pos.compute_at(projection_result, x);

        apply_auto_schedule(projection_result);

        return std::make_tuple(projection_result, origin);
    }

    Func sphere_trace(/*Func sdf, */size_t iterations = 300, float EPS = 1e-6) {
        Func rays("rays");
        Func origin("origin");

        std::tie(rays, origin) = projection(projection_, view_);

        RDom tr(0, 300);
        Func pos("pos");
        Func d("d");
        pos(x, y, c, t) = 0.0f;
        d(x, y, c, t) = 0.0f;

        pos(x, y, c, tr) = pos(x, y, c, tr - 1) +
                           d(x, y, c, tr - 1) * rays(x, y, c)[1];
        /*d(x, y, c, tr) = sdf(pos(x, y, 0, tr),
                             pos(x, y, 1, tr),
                             pos(x, y, 2, tr));*/
        d(x, y, c, tr) = Halide::sqrt(
                             pos(x, y, 0, tr) * pos(x, y, 0, tr) +
                             pos(x, y, 1, tr) * pos(x, y, 1, tr) +
                             pos(x, y, 2, tr) * pos(x, y, 2, tr)
                         ) - 3.0f;
        pos(x, y, c, 0) = rays(x, y, c)[0];

        Func endpoint("endpoint");
        endpoint(x, y, c) = d(x, y, c, 300);

        return endpoint;
    }

    void generate() {
        Func rays("rays");
        Func origin("origin");
        std::tie(rays, origin) = projection(projection_, view_);
        rays.compute_root();

        Func sphere_sdf("sphere_sdf");
        float radius = 3.0f;

        /*Expr xf("xf"), yf("yf"), zf("zf");
        sphere_sdf(xf, yf, zf) =
            Halide::sqrt(
            xf * xf + yf * yf + zf * zf) - radius;*/

        Func end("end");
        end(x, y, c) = sphere_trace()(x, y, c);

        //out_(x, y, c) =
        //    matmul::product(matmul::product(Func(b), 3.0f), Func(b))(x, y);
        //out_(x, y, c) = matmul::inverse(Func(a))(x, y);
        //out_(x, y, c) = clamp(rays(x, y, 2 - c)[1], 0.0f, 1.0f);
        //out_(x, y, c) = rays(x, y, c)[1];
        out_(x, y, c) = end(x, y, c);
    }
};

HALIDE_REGISTER_GENERATOR(TracerGenerator, tracer_render)

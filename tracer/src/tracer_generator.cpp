#include <iostream>

#include "Halide.h"

#include "matmul.hpp"

#include <stdio.h>

using namespace Halide;

Var x("x"), y("y"), c("c");

void apply_auto_schedule(Func F) {
    std::map<std::string, Internal::Function> flist = Internal::find_transitive_calls(
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

    Input<Buffer<float>> projection_{"projection" , 2};
    Input<Buffer<float>> view_{"view" , 2};
    Output<Buffer<float>> out_{"out", 3};

    Func projection(Func proj_matrix,
                    Func view_matrix,
                    size_t width,
                    size_t height,
                    float near = 0.1f) {
        Func rays("rays");

        Func ss_norm;
        ss_norm(x, y) = {
            cast<float>(x) / ((float) width),
            cast<float>(y) / ((float) height)
        };

        Func clip_space("clip_space");
        clip_space(x, y, c) = 1.0f;
        clip_space(x, y, 0) = ss_norm(x, y)[0] * 2.0f - 1.0f;
        clip_space(x, y, 1) = ss_norm(x, y)[1] * 2.0f - 1.0f;
        clip_space.bound(x, 0, 4).unroll(x);
        clip_space.bound(y, 0, 4).unroll(y);

        Func viewproj_inv("viewproj_inv");
        viewproj_inv =
            matmul::inverse(matmul::product(proj_matrix, view_matrix));

        Func view_inv("view_inv");
        view_inv =
            matmul::inverse(view_matrix);

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
        ray_vec.bound(c, 0, 3).unroll(c);
        projected.compute_at(ray_vec, x);

        Func ray_pos("ray_pos");
        ray_pos(x, y, c) = origin(c) + ray_vec(x, y, c) * near;

        Func projection_result("projection_result");
        projection_result(x, y, c) = {ray_pos(x, y, c), ray_vec(x, y, c)};
        ray_vec.compute_at(projection_result, x);
        ray_pos.compute_at(projection_result, x);

        apply_auto_schedule(projection_result);

        return projection_result;
    }

    void generate() {

        Buffer<float> a(4, 4);
        Buffer<float> b(4, 4);

        auto identity = [](Buffer<float>& a) {
            return [&a](int x, int y) {
                if (x == y) {
                    a(x, y) = 1.0f;
                } else {
                    a(x, y) = 0.0;
                }
            };
        };

        a.for_each_element(identity(a));

        b.for_each_element(identity(b));

        Func rays("rays");
        rays = projection(Func(a), Func(b), 100, 100);
        rays.compute_root();

        //out_(x, y, c) =
        //    matmul::product(matmul::product(Func(b), 3.0f), Func(b))(x, y);
        //out_(x, y, c) = matmul::inverse(Func(a))(x, y);
        out_(x, y, c) = rays(x, y, c)[0];
    }
};

HALIDE_REGISTER_GENERATOR(TracerGenerator, tracer_render)

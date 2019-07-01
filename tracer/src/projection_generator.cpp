#include <iostream>
#include <tuple>

#include "Halide.h"

#include "grid_sdf.hpp"
#include "matmul.hpp"

#include <stdio.h>

using namespace Halide;

namespace {

    class ProjectionGenerator : public Halide::Generator<ProjectionGenerator> {
        Input<Buffer<float>> projection_{"projection", 2};
        Input<Buffer<float>> view_{"view", 2};
        Input<int32_t> width{"width"};
        Input<int32_t> height{"height"};

        Func view_matrix{"view_matrix"};
        Func projection_matrix{"projection_matrix"};
        Func rays{"rays"};
        Func ss_norm{"ss_norm"};
        Func clip_space{"clip_space"};
        Func viewproj_inv{"viewproj_inv"};
        Func view_inv{"view_inv"};
        Func homogeneous{"homogeneous"};
        Func projected{"projected"};

        Output<Func> ray_pos{"ray_pos", {Float(32), Float(32), Float(32)}, 2};
        Output<Func> ray_vec{"ray_vec", {Float(32), Float(32), Float(32)}, 2};
        Output<Func> origin{"origin", 1};

        constexpr static float near = 0.1f;

      public:

        void generate() {
            view_matrix(x, y) = view_(x, y);
            projection_matrix(x, y) = projection_(x, y);

            ss_norm(x, y) = {
                cast<float>(x) / (cast<float>(width)),
                cast<float>(y) / (cast<float>(height))
            };

            clip_space(x, y, c) = 1.0f;
            clip_space(x, y, 0) = ss_norm(x, y)[0] * 2.0f - 1.0f;
            clip_space(x, y, 1) = ss_norm(x, y)[1] * 2.0f - 1.0f;
            clip_space.bound(c, 0, 4).unroll(c);

            viewproj_inv =
                matmul::inverse(matmul::product(projection_matrix, view_matrix));

            view_inv =
                matmul::inverse(view_matrix);

            homogeneous(x, y, c) = sum(viewproj_inv(c, k) *
                                       clip_space(x, y, k));
            homogeneous.bound(c, 0, 4).unroll(c);

            origin(c) = view_inv(c, 3);
            origin.bound(c, 0, 4).unroll(c);

            projected(x, y, c) = (homogeneous(x, y, c) / homogeneous(x, y, 3))
                                 - origin(c);
            projected.bound(c, 0, 4).unroll(c);

            // could use fast inverse sqrt, but not worth accuracy loss
            Expr ray_vec_norm = Halide::sqrt(
                                    sum(
                                        projected(x, y, norm_k) *
                                        projected(x, y, norm_k)));
            ray_vec(x, y) = Tuple(projected(x, y, 0) / ray_vec_norm,
                                  projected(x, y, 1) / ray_vec_norm,
                                  projected(x, y, 2) / ray_vec_norm);

            //ray_pos(x, y, c) = origin(c) + ray_vec(x, y, c) * near;
            ray_pos(x, y) = Tuple(origin(0) + ray_vec(x, y)[0] * near,
                                  origin(1) + ray_vec(x, y)[1] * near,
                                  origin(2) + ray_vec(x, y)[2] * near);

            if (auto_schedule) {
                std::cout << "auto schedule projection" << std::endl;
                projection_.dim(0).set_bounds_estimate(0, 4)
                           .dim(1).set_bounds_estimate(0, 4);
                view_.dim(0).set_bounds_estimate(0, 4)
                     .dim(1).set_bounds_estimate(0, 4);
            } else {
                std::cout << "not auto schedule projection" << std::endl;

                /*viewproj_inv.compute_at(homogeneous, x);

                homogeneous.compute_at(projected, x);
                origin.compute_at(projected, x);

                projected.compute_at(ray_vec, x);*/
                apply_auto_schedule(ray_pos);
                apply_auto_schedule(ray_vec);
                apply_auto_schedule(origin);
            }
        }

      private:
        Var x{"x"}, y{"y"}, c{"c"};
        RDom k{0, 4};
        RDom norm_k{0, 3};
    };
} // namespace

HALIDE_REGISTER_GENERATOR(ProjectionGenerator, projection)

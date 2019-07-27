#include <iostream>
#include <string>
#include <stdio.h>

#include "Halide.h"

#include "matmul.hpp"
#include "grid_sdf.hpp"

using namespace Halide;

#define SQRT_2 1.41421356f
#define SQRT_3 1.73205080f

class FMMGenerator : public Halide::Generator<FMMGenerator> {
  public:
    Input<Func> sdf_{"sdf", Float(32), 3};
    Input<int32_t> d0{"d0"};
    Input<int32_t> d1{"d1"};
    Input<int32_t> d2{"d2"};

    Output<Func> sdf_output_{"sdf_output", 3};

    Expr sign(Expr a) {
        return select(a < 0.0f, -1.0f, 1.0f);
    }

    void generate() {
        Func clamped = BoundaryConditions::repeat_edge(sdf_, {{0, d0}, {0, d1}, {0, d2}});

        Func sdf_iterative("sdf_iterative");
        sdf_iterative(x, y, z, t) = 1e10f;

        // neighborhood around each voxel
        RDom r(-1, 2, -1, 2, -1, 2, "r");

        //Expr all_iters = max(max(d0, d1), d2);
        Expr all_iters = cast<int32_t>(Halide::sqrt(
                                           cast<float>(d0 * d0 + d1 * d1 + d2 * d2)));
        RDom it(0, all_iters);

        Func level_set_min("level_set_min");
        level_set_min(x, y, z) = minimum(clamped(x + r.x, y + r.y, z + r.z));

        Func level_set_max("level_set_max");
        level_set_max(x, y, z) = maximum(clamped(x + r.x, y + r.y, z + r.z));

        Func level_set("level_set");
        // level set involves a zero crossing
        level_set(x, y, z) = select(level_set_min(x, y, z) <= 0.0f &&
                                    level_set_max(x, y, z) >= 0.0f,
                                    1, 0);

        // each iteration, we consider our neighbors
        Func considered("considered");
        RDom rd(0, d0, 0, d1, 0, d2, 0, all_iters, "rd");

        considered(x, y, z, t) = level_set(x, y, z);

        considered(rd.x, rd.y, rd.z, rd.w + 1) =
            cast<int32_t>(max(
                              considered(rd.x - 1, rd.y, rd.z, rd.w),
                              considered(rd.x - 1, rd.y - 1, rd.z, rd.w),
                              considered(rd.x - 1, rd.y + 1, rd.z, rd.w),
                              considered(rd.x - 1, rd.y, rd.z - 1, rd.w),
                              considered(rd.x - 1, rd.y, rd.z + 1, rd.w),
                              considered(rd.x - 1, rd.y - 1, rd.z - 1, rd.w),
                              considered(rd.x - 1, rd.y - 1, rd.z + 1, rd.w),
                              considered(rd.x - 1, rd.y + 1, rd.z - 1, rd.w),
                              considered(rd.x - 1, rd.y + 1, rd.z + 1, rd.w),

                              considered(rd.x, rd.y - 1, rd.z, rd.w),
                              considered(rd.x, rd.y + 1, rd.z, rd.w),
                              considered(rd.x, rd.y, rd.z - 1, rd.w),
                              considered(rd.x, rd.y, rd.z + 1, rd.w),
                              considered(rd.x, rd.y - 1, rd.z - 1, rd.w),
                              considered(rd.x, rd.y - 1, rd.z + 1, rd.w),
                              considered(rd.x, rd.y + 1, rd.z - 1, rd.w),
                              considered(rd.x, rd.y + 1, rd.z + 1, rd.w),

                              considered(rd.x + 1, rd.y, rd.z, rd.w),
                              considered(rd.x + 1, rd.y - 1, rd.z, rd.w),
                              considered(rd.x + 1, rd.y + 1, rd.z, rd.w),
                              considered(rd.x + 1, rd.y, rd.z - 1, rd.w),
                              considered(rd.x + 1, rd.y, rd.z + 1, rd.w),
                              considered(rd.x + 1, rd.y - 1, rd.z - 1, rd.w),
                              considered(rd.x + 1, rd.y - 1, rd.z + 1, rd.w),
                              considered(rd.x + 1, rd.y + 1, rd.z - 1, rd.w),
                              considered(rd.x + 1, rd.y + 1, rd.z + 1, rd.w)
                          ));

        // select the cells which will be occupied next iteration
        Func next_iter("next_iter");
        next_iter(x, y, z, t) = 0;
        next_iter(x, y, z, it) = select(considered(x, y, z, it + 1) !=
                                        considered(x, y, z, it), 1, 0);

        sdf_iterative(x, y, z, 0) = cast<float>(level_set(x, y, z)) * clamped(x, y, z);
        sdf_iterative(rd.x, rd.y, rd.z, rd.w + 1) =
            select(next_iter(rd.x, rd.y, rd.z, rd.w) == 1,
                   min(
                       sdf_iterative(rd.x - 1, rd.y, rd.z, rd.w)
                       + 1 * sign(sdf_iterative(rd.x - 1, rd.y, rd.z, rd.w)),
                       sdf_iterative(rd.x - 1, rd.y - 1, rd.z, rd.w)
                       + SQRT_2 * sign(sdf_iterative(rd.x - 1, rd.y - 1, rd.z, rd.w)),
                       sdf_iterative(rd.x - 1, rd.y + 1, rd.z, rd.w)
                       + SQRT_2 * sign(sdf_iterative(rd.x - 1, rd.y + 1, rd.z, rd.w)),
                       sdf_iterative(rd.x - 1, rd.y, rd.z - 1, rd.w)
                       + SQRT_2 * sign(sdf_iterative(rd.x - 1, rd.y, rd.z - 1, rd.w)),
                       sdf_iterative(rd.x - 1, rd.y, rd.z + 1, rd.w)
                       + SQRT_2 * sign(sdf_iterative(rd.x - 1, rd.y, rd.z + 1, rd.w)),
                       sdf_iterative(rd.x - 1, rd.y - 1, rd.z - 1, rd.w)
                       + SQRT_3 * sign(sdf_iterative(rd.x - 1, rd.y - 1, rd.z - 1, rd.w)),
                       sdf_iterative(rd.x - 1, rd.y - 1, rd.z + 1, rd.w)
                       + SQRT_3 * sign(sdf_iterative(rd.x - 1, rd.y - 1, rd.z + 1, rd.w)),
                       sdf_iterative(rd.x - 1, rd.y + 1, rd.z - 1, rd.w)
                       + SQRT_3 * sign(sdf_iterative(rd.x - 1, rd.y + 1, rd.z - 1, rd.w)),
                       sdf_iterative(rd.x - 1, rd.y + 1, rd.z + 1, rd.w)
                       + SQRT_3 * sign(sdf_iterative(rd.x - 1, rd.y + 1, rd.z + 1, rd.w)),

                       sdf_iterative(rd.x, rd.y - 1, rd.z, rd.w)
                       + 1 * sign(sdf_iterative(rd.x, rd.y - 1, rd.z, rd.w)),
                       sdf_iterative(rd.x, rd.y + 1, rd.z, rd.w)
                       + 1 * sign(sdf_iterative(rd.x, rd.y + 1, rd.z, rd.w)),
                       sdf_iterative(rd.x, rd.y, rd.z - 1, rd.w)
                       + 1 * sign(sdf_iterative(rd.x, rd.y, rd.z - 1, rd.w)),
                       sdf_iterative(rd.x, rd.y, rd.z + 1, rd.w)
                       + 1 * sign(sdf_iterative(rd.x, rd.y, rd.z + 1, rd.w)),
                       sdf_iterative(rd.x, rd.y - 1, rd.z - 1, rd.w)
                       + SQRT_2 * sign(sdf_iterative(rd.x, rd.y - 1, rd.z - 1, rd.w)),
                       sdf_iterative(rd.x, rd.y - 1, rd.z + 1, rd.w)
                       + SQRT_2 * sign(sdf_iterative(rd.x, rd.y - 1, rd.z + 1, rd.w)),
                       sdf_iterative(rd.x, rd.y + 1, rd.z - 1, rd.w)
                       + SQRT_2 * sign(sdf_iterative(rd.x, rd.y + 1, rd.z - 1, rd.w)),
                       sdf_iterative(rd.x, rd.y + 1, rd.z + 1, rd.w)
                       + SQRT_2 * sign(sdf_iterative(rd.x, rd.y + 1, rd.z + 1, rd.w)),

                       sdf_iterative(rd.x + 1, rd.y, rd.z, rd.w)
                       + 1 * sign(sdf_iterative(rd.x + 1, rd.y, rd.z, rd.w)),
                       sdf_iterative(rd.x + 1, rd.y - 1, rd.z, rd.w)
                       + SQRT_2 * sign(sdf_iterative(rd.x + 1, rd.y - 1, rd.z, rd.w)),
                       sdf_iterative(rd.x + 1, rd.y + 1, rd.z, rd.w)
                       + SQRT_2 * sign(sdf_iterative(rd.x + 1, rd.y + 1, rd.z, rd.w)),
                       sdf_iterative(rd.x + 1, rd.y, rd.z - 1, rd.w)
                       + SQRT_2 * sign(sdf_iterative(rd.x + 1, rd.y, rd.z - 1, rd.w)),
                       sdf_iterative(rd.x + 1, rd.y, rd.z + 1, rd.w)
                       + SQRT_2 * sign(sdf_iterative(rd.x + 1, rd.y, rd.z + 1, rd.w)),
                       sdf_iterative(rd.x + 1, rd.y - 1, rd.z - 1, rd.w)
                       + SQRT_3 * sign(sdf_iterative(rd.x + 1, rd.y - 1, rd.z - 1, rd.w)),
                       sdf_iterative(rd.x + 1, rd.y - 1, rd.z + 1, rd.w)
                       + SQRT_3 * sign(sdf_iterative(rd.x + 1, rd.y - 1, rd.z + 1, rd.w)),
                       sdf_iterative(rd.x + 1, rd.y + 1, rd.z - 1, rd.w)
                       + SQRT_3 * sign(sdf_iterative(rd.x + 1, rd.y + 1, rd.z - 1, rd.w)),
                       sdf_iterative(rd.x + 1, rd.y + 1, rd.z + 1, rd.w)
                       + SQRT_3 * sign(sdf_iterative(rd.x + 1, rd.y + 1, rd.z + 1, rd.w))
                   ),
                   sdf_iterative(rd.x, rd.y, rd.z, rd.w)
                  );

        // we start with just the level set
        /*sdf_iterative(x, y, z, 0) = cast<float>(level_set(x, y, z)) * sdf_(x, y, z);

        // a tuple with the locations within the RDom which have the minimum value
        Tuple sdf_neighbors = argmin(sdf_iterative(x + r.x, y + r.y, z + r.z, it));
        // the distance to the nearest neighbor
        Expr neighbor_dist = norm(sdf_neighbors);

        // we only update when we are considering a node
        sdf_iterative(x, y, z, it + 1) =
            select(next_iter(x, y, z, it) == 1,
                   neighbor_dist,
                   sdf_iterative(x, y, z, it));

                   sdf_output_(x, y, z) = sdf_iterative(x, y, z, all_iters);*/
        //sdf_output_(x, y, z) = cast<float>(considered(x, y, z, 2));
        //sdf_output_(x, y, z) = cast<float>(next_iter(x, y, z, 2));
        //sdf_output_(x, y, z) = cast<float>(sdf_iterative(x, y, z, all_iters));
        sdf_output_(x, y, z) = abs(clamped(x, y, z));
        apply_auto_schedule(sdf_output_);

        /*Halide::SimpleAutoscheduleOptions options;
        options.gpu = get_target().has_gpu_feature();
        options.gpu_tile_channel = 1;
        options.unroll_rvar_size = 128;

        std::vector<Func> output_func({sdf_output_});

        Halide::simple_autoschedule(
        output_func, {
            {"sdf_.min.0", 0},
            {"sdf_.extent.0", 128},
            {"sdf_.min.1", 0},
            {"sdf_.extent.1", 128},
            {"sdf_.min.2", 0},
            {"sdf_.extent.2", 128},
        }, {
            {
                {0, 128},
                {0, 128},
                {0, 128}
            }
        },
        options);*/
    }

  private:
    Var x{"x"}, y{"y"}, z{"c"}, t{"t"}, n{"n"};
};

HALIDE_REGISTER_GENERATOR(FMMGenerator, fmm_gen)

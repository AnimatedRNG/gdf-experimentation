#include <iostream>
#include <string>
#include <stdio.h>

#include "Halide.h"

#include "matmul.hpp"
#include "grid_sdf.hpp"

using namespace Halide;

class FMMGenerator : public Halide::Generator<FMMGenerator> {
  public:
    Input<Func> sdf_{"sdf", Float(32), 3};
    Input<int32_t> d0{"d0"};
    Input<int32_t> d1{"d1"};
    Input<int32_t> d2{"d2"};

    Output<Func> sdf_output_{"sdf_output", 3};

    void generate() {
        Func clamped = BoundaryConditions::repeat_edge(sdf_, {{0, d0}, {0, d1}, {0, d2}});

        Func sdf_iterative("sdf_iterative");
        sdf_iterative(x, y, z, t) = 1e10f;

        // neighborhood around each voxel
        RDom r(-1, 2, -1, 2, -1, 2, "r");

        Expr all_iters = max(max(d0, d1), d2);
        RDom it(0, all_iters);

        Func level_set_min("level_set_min");
        level_set_min(x, y, z) = minimum(sdf_(x + r.x, y + r.y, z + r.z));

        Func level_set_max("level_set_max");
        level_set_max(x, y, z) = maximum(sdf_(x + r.x, y + r.y, z + r.z));

        Func level_set("level_set");
        // level set involves a zero crossing
        level_set(x, y, z) = select(level_set_min(x, y, z) <= 0.0f &&
                                    level_set_max(x, y, z) >= 0.0f,
                                    1, 0);

        Func considered("considered");
        considered(x, y, z, t) = level_set(x, y, z);

        // each iteration, we consider our neighbors
        considered(x, y, z, it + 1) =
            cast<int32_t>(
                maximum(r, considered(x + r.x, y + r.y, z + r.z, it)));


        // select the cells which will be occupied next iteration
        Func next_iter("next_iter");
        next_iter(x, y, z, t) = 0;
        next_iter(x, y, z, it) = select(considered(x, y, z, it + 1) !=
                                        considered(x, y, z, it), 1, 0);

        // we start with just the level set
        sdf_iterative(x, y, z, 0) = cast<float>(level_set(x, y, z)) * sdf_(x, y, z);

        // a tuple with the locations within the RDom which have the minimum value
        Tuple sdf_neighbors = argmin(sdf_iterative(x + r.x, y + r.y, z + r.z, it));
        // the distance to the nearest neighbor
        Expr neighbor_dist = norm(sdf_neighbors);

        // we only update when we are considering a node
        sdf_iterative(x, y, z, it + 1) =
            select(next_iter(x, y, z, it) == 1,
                   neighbor_dist,
                   sdf_iterative(x, y, z, it));

        sdf_output_(x, y, z) = sdf_iterative(x, y, z, all_iters);

        Halide::SimpleAutoscheduleOptions options;
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
        options);
    }

  private:
    Var x{"x"}, y{"y"}, z{"c"}, t{"t"}, n{"n"};
};

HALIDE_REGISTER_GENERATOR(FMMGenerator, fmm_gen)

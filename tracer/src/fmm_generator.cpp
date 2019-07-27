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
    Input<Func> p0_{"p0", Float(32), 1};
    Input<Func> p1_{"p1", Float(32), 1};
    Input<int32_t> d0{"d0"};
    Input<int32_t> d1{"d1"};
    Input<int32_t> d2{"d2"};

    Output<Func> sdf_output_{"sdf_output", 3};

    Expr sign(Expr a) {
        return select(a < 0.0f, -1.0f, 1.0f);
    }

    Expr abs_min(Tuple a) {
        // abs min value, min value
        Tuple tmp = {1e20f, 1e20f};
        for (const Expr& e : a.as_vector()) {
            tmp[0] = select(abs(e) <= tmp[0], abs(e), tmp[0]);
            tmp[1] = select(abs(e) <= tmp[0], e, tmp[1]);
        }
        return tmp[1];
    }

    void generate() {
        Func clamped = BoundaryConditions::repeat_edge(sdf_, {{0, d0}, {0, d1}, {0, d2}});

        Func sdf_iterative("sdf_iterative");
        sdf_iterative(x, y, z, t) = 1e10f;

        TupleVec<3> p0({p0_(0), p0_(1), p0_(2)});
        TupleVec<3> p1({p1_(0), p1_(1), p1_(2)});

        TupleVec<3> pd = abs(p1 - p0);
        TupleVec<3> us = pd / TupleVec<3>({
            cast<float>(d0),
            cast<float>(d1),
            cast<float>(d2)
        });

        Var x1, y1, z1;
        Func ds("ds");
        // maps from the corner of the voxel to the other corners
        ds(x1, y1, z1) = 0.0f;

        for (int i1 = 0; i1 <= 1; i1++) {
            for (int j1 = 0; j1 <= 1; j1++) {
                for (int k1 = 0; k1 <= 1; k1++) {
                    TupleVec<3> p({us[0] * i1, us[1] * j1, us[2] * k1});
                    ds(i1, j1, k1) = norm(p);
                }
            }
        }

        // neighborhood around each voxel
        RDom r(0, 3, 0, 3, 0, 3, "r");

        //Expr all_iters = max(max(d0, d1), d2);
        Expr all_iters = cast<int32_t>(Halide::sqrt(
                                           cast<float>(d0 * d0 + d1 * d1 + d2 * d2)));
        RDom it(0, all_iters);

        Func level_set_min("level_set_min");
        level_set_min(x, y, z) = minimum(clamped(x + r.x - 1, y + r.y - 1, z + r.z - 1));

        Func level_set_max("level_set_max");
        level_set_max(x, y, z) = maximum(clamped(x + r.x - 1, y + r.y - 1, z + r.z - 1));

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

        //sdf_iterative(x, y, z, 0) = cast<float>(level_set(x, y, z)) * clamped(x, y, z);
        sdf_iterative(x, y, z, 0) = select(level_set(x, y, z) == 1,
                                           clamped(x, y, z),
                                           1e10f);
        sdf_iterative(rd.x, rd.y, rd.z, rd.w + 1) =
            select(next_iter(rd.x, rd.y, rd.z, rd.w) == 1,
                   abs_min(
                       Tuple(
                           sdf_iterative(rd.x - 1, rd.y, rd.z, rd.w)
                           + ds(1, 0, 0) * sign(sdf_iterative(rd.x - 1, rd.y, rd.z, rd.w)),
                           sdf_iterative(rd.x - 1, rd.y - 1, rd.z, rd.w)
                           + ds(1, 1, 0) * sign(sdf_iterative(rd.x - 1, rd.y - 1, rd.z, rd.w)),
                           sdf_iterative(rd.x - 1, rd.y + 1, rd.z, rd.w)
                           + ds(1, 1, 0) * sign(sdf_iterative(rd.x - 1, rd.y + 1, rd.z, rd.w)),
                           sdf_iterative(rd.x - 1, rd.y, rd.z - 1, rd.w)
                           + ds(1, 0, 1) * sign(sdf_iterative(rd.x - 1, rd.y, rd.z - 1, rd.w)),
                           sdf_iterative(rd.x - 1, rd.y, rd.z + 1, rd.w)
                           + ds(1, 0, 1) * sign(sdf_iterative(rd.x - 1, rd.y, rd.z + 1, rd.w)),
                           sdf_iterative(rd.x - 1, rd.y - 1, rd.z - 1, rd.w)
                           + ds(1, 1, 1) * sign(sdf_iterative(rd.x - 1, rd.y - 1, rd.z - 1, rd.w)),
                           sdf_iterative(rd.x - 1, rd.y - 1, rd.z + 1, rd.w)
                           + ds(1, 1, 1) * sign(sdf_iterative(rd.x - 1, rd.y - 1, rd.z + 1, rd.w)),
                           sdf_iterative(rd.x - 1, rd.y + 1, rd.z - 1, rd.w)
                           + ds(1, 1, 1) * sign(sdf_iterative(rd.x - 1, rd.y + 1, rd.z - 1, rd.w)),
                           sdf_iterative(rd.x - 1, rd.y + 1, rd.z + 1, rd.w)
                           + ds(1, 1, 1) * sign(sdf_iterative(rd.x - 1, rd.y + 1, rd.z + 1, rd.w)),

                           sdf_iterative(rd.x, rd.y - 1, rd.z, rd.w)
                           + ds(0, 1, 0) * sign(sdf_iterative(rd.x, rd.y - 1, rd.z, rd.w)),
                           sdf_iterative(rd.x, rd.y + 1, rd.z, rd.w)
                           + ds(0, 1, 0) * sign(sdf_iterative(rd.x, rd.y + 1, rd.z, rd.w)),
                           sdf_iterative(rd.x, rd.y, rd.z - 1, rd.w)
                           + ds(0, 0, 1) * sign(sdf_iterative(rd.x, rd.y, rd.z - 1, rd.w)),
                           sdf_iterative(rd.x, rd.y, rd.z + 1, rd.w)
                           + ds(0, 0, 1) * sign(sdf_iterative(rd.x, rd.y, rd.z + 1, rd.w)),
                           sdf_iterative(rd.x, rd.y - 1, rd.z - 1, rd.w)
                           + ds(0, 1, 1) * sign(sdf_iterative(rd.x, rd.y - 1, rd.z - 1, rd.w)),
                           sdf_iterative(rd.x, rd.y - 1, rd.z + 1, rd.w)
                           + ds(0, 1, 1) * sign(sdf_iterative(rd.x, rd.y - 1, rd.z + 1, rd.w)),
                           sdf_iterative(rd.x, rd.y + 1, rd.z - 1, rd.w)
                           + ds(0, 1, 1) * sign(sdf_iterative(rd.x, rd.y + 1, rd.z - 1, rd.w)),
                           sdf_iterative(rd.x, rd.y + 1, rd.z + 1, rd.w)
                           + ds(0, 1, 1) * sign(sdf_iterative(rd.x, rd.y + 1, rd.z + 1, rd.w)),

                           sdf_iterative(rd.x + 1, rd.y, rd.z, rd.w)
                           + ds(1, 0, 0) * sign(sdf_iterative(rd.x + 1, rd.y, rd.z, rd.w)),
                           sdf_iterative(rd.x + 1, rd.y - 1, rd.z, rd.w)
                           + ds(1, 1, 0) * sign(sdf_iterative(rd.x + 1, rd.y - 1, rd.z, rd.w)),
                           sdf_iterative(rd.x + 1, rd.y + 1, rd.z, rd.w)
                           + ds(1, 1, 0) * sign(sdf_iterative(rd.x + 1, rd.y + 1, rd.z, rd.w)),
                           sdf_iterative(rd.x + 1, rd.y, rd.z - 1, rd.w)
                           + ds(1, 0, 1) * sign(sdf_iterative(rd.x + 1, rd.y, rd.z - 1, rd.w)),
                           sdf_iterative(rd.x + 1, rd.y, rd.z + 1, rd.w)
                           + ds(1, 0, 1) * sign(sdf_iterative(rd.x + 1, rd.y, rd.z + 1, rd.w)),
                           sdf_iterative(rd.x + 1, rd.y - 1, rd.z - 1, rd.w)
                           + ds(1, 1, 1) * sign(sdf_iterative(rd.x + 1, rd.y - 1, rd.z - 1, rd.w)),
                           sdf_iterative(rd.x + 1, rd.y - 1, rd.z + 1, rd.w)
                           + ds(1, 1, 1) * sign(sdf_iterative(rd.x + 1, rd.y - 1, rd.z + 1, rd.w)),
                           sdf_iterative(rd.x + 1, rd.y + 1, rd.z - 1, rd.w)
                           + ds(1, 1, 1) * sign(sdf_iterative(rd.x + 1, rd.y + 1, rd.z - 1, rd.w)),
                           sdf_iterative(rd.x + 1, rd.y + 1, rd.z + 1, rd.w)
                           + ds(1, 1, 1) * sign(sdf_iterative(rd.x + 1, rd.y + 1, rd.z + 1, rd.w))
                       )
                   ),
                   sdf_iterative(rd.x, rd.y, rd.z, rd.w)
                  );

        //sdf_output_(x, y, z) = cast<float>(considered(x, y, z, 2));
        //sdf_output_(x, y, z) = cast<float>(next_iter(x, y, z, 2));
        sdf_output_(x, y, z) = sdf_iterative(x, y, z, all_iters);
        //sdf_output_(x, y, z) = max(0.0f, sdf_iterative(x, y, z, 4));
        //sdf_output_(x, y, z) = cast<float>(level_set(x, y, z));
        //sdf_output_(x, y, z) = clamped(x, y, z);
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

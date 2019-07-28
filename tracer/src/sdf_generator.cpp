#include "Halide.h"

#include "grid_sdf.hpp"

using namespace Halide;

namespace {
    Expr example_sphere(TupleVec<3> position) {
        return norm(position) - 3.0f;
    }

    Expr vmax(TupleVec<3> v) {
        return Halide::max(Halide::max(v[0], v[1]), v[2]);
    }

    Expr example_box(TupleVec<3> position) {
        TupleVec<3> b = {3.0f, 3.0f, 3.0f};
        TupleVec<3> d = abs(position + Tuple(-0.22f, 0.19f, 0.34f)) - b;

        return norm(max(d, Expr(0.0f))) + vmax(min(d, Expr(0.0f)));
    }

    class SDFGenerator : public Halide::Generator<SDFGenerator> {
      public:
        Input<int32_t> res_x{"res_x"};
        Input<int32_t> res_y{"res_y"};
        Input<int32_t> res_z{"res_z"};

        Output<Func> sdf_{"sdf_", Float(32), 3};
        Output<Func> p0{"p0", Float(32), 1};
        Output<Func> p1{"p1", Float(32), 1};

        void generate() {
            GridSDF grid_sdf = to_grid_sdf(example_box,
            {-4.0f, -4.0f, -4.0f},
            {4.0f, 4.0f, 4.0f}, 64, 64, 64);

            sdf_(x, y, z) = grid_sdf.buffer(x, y, z);

            p0(x) = 0.0f;
            p0(0) = grid_sdf.p0.get()[0];
            p0(1) = grid_sdf.p0.get()[1];
            p0(2) = grid_sdf.p0.get()[2];

            p1(x) = 0.0f;
            p1(0) = grid_sdf.p1.get()[0];
            p1(1) = grid_sdf.p1.get()[1];
            p1(2) = grid_sdf.p1.get()[2];

            if (/*auto_schedule*/ true) {
                sdf_.estimate(x, 0, 128)
                .estimate(y, 0, 128)
                .estimate(z, 0, 128);
                p0.estimate(x, 0, 3);
                p1.estimate(x, 0, 3);

                std::vector<Func> output_func({sdf_, p0, p1});

                Halide::SimpleAutoscheduleOptions options;
                options.gpu = get_target().has_gpu_feature();
                options.gpu_tile_channel = 1;
                options.unroll_rvar_size = 128;

                Halide::simple_autoschedule(
                output_func, {
                }, {
                    {
                        {0, 128},
                        {0, 128},
                        {0, 128}
                    },
                    {{0, 4}},
                    {{0, 4}}
                },
                options);
            } else {
                // TODO
            }
        }
      private:
        Var x{"x"}, y{"y"}, z{"z"};
    };

} // namespace

HALIDE_REGISTER_GENERATOR(SDFGenerator, sdf_gen)

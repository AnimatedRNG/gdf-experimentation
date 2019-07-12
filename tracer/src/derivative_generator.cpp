#include <iostream>

#include "Halide.h"

#include "tracer_render.stub.h"

#include "recorder.hpp"

namespace {

    class DerivativeGenerator : public Halide::Generator<DerivativeGenerator> {
      public:
        Input<Buffer<float>> projection_{"projection", 2};
        Input<Buffer<float>> view_{"view", 2};

        Input<Buffer<float>> sdf_{"sdf_", 3};
        Input<Buffer<float>> p0_{"p0", 1};
        Input<Buffer<float>> p1_{"p1", 1};

        Input<int32_t> width{"width"};
        Input<int32_t> height{"height"};
        Input<int32_t> initial_debug{"initial_debug"};
        Output<Buffer<float>> out_{"out", 3};
        Output<Func> debug_{"debug", UInt(8), 5};
        Output<Func> num_debug{"num_debug", Int(32), 1};

        Func forward_pass{"forward_pass"};

        int current_debug = 0;

        void record(Func f) {
            _record(f, debug_, num_debug, initial_debug, current_debug);
        }

        void generate() {
            Func p0("p0");

            auto outputs = tracer_render::generate(Halide::GeneratorContext(
            this->get_target(), auto_schedule), {
                projection_,
                view_,
                sdf_,
                p0_,
                p1_,
                width,
                height,
                initial_debug
            });

            forward_pass = Func(outputs.out);
            out_(x, y, c) = forward_pass(x, y, c);

            debug_ = outputs.debug;

            num_debug(x) = Func(Expr(current_debug) + outputs.num_debug(
                                    0) + initial_debug)();

            if (auto_schedule) {
                projection_.dim(0).set_bounds_estimate(0, 4)
                .dim(1).set_bounds_estimate(0, 4);
                view_.dim(0).set_bounds_estimate(0, 4)
                .dim(1).set_bounds_estimate(0, 4);

                out_.dim(0).set_bounds_estimate(0, 1920)
                .dim(1).set_bounds_estimate(0, 1920)
                .dim(2).set_bounds_estimate(0, 3);

                debug_
                .estimate(i, 0, current_debug)
                .estimate(t, 0, 300)
                .estimate(x, 0, 1920)
                .estimate(y, 0, 1080)
                .estimate(c, 0, 3);
                num_debug.estimate(x, 0, 1);
            }
        }

      private:
        Var x{"x"}, y{"y"}, c{"c"}, t{"t"}, i{"i"};
    };
} // namespace

HALIDE_REGISTER_GENERATOR(DerivativeGenerator, derivative_render)

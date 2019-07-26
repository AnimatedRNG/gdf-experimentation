#include <iostream>
#include <string>
#include <stdio.h>

#include "Halide.h"

#include "matmul.hpp"
#include "grid_sdf.hpp"

using namespace Halide;

class OptimizerGenerator : public Halide::Generator<OptimizerGenerator> {
  public:
    Input<Buffer<float>> f_{"sdf", 1};
    Input<Buffer<float>> gradient_{"gradient", 1};

    Input<float> lr{"lr"};
    Input<float> beta_1{"beta_1"};
    Input<float> beta_2{"beta_2"};
    Input<float> weight_decay{"weight_decay"};
    Input<float> eps{"eps"};
    Input<float> iteration{"iteration"};

    Input<Buffer<float>> exp_avg{"exp_avg", 1};
    Input<Buffer<float>> exp_avg_sq{"exp_avg_sq", 1};

    Output<Func> exp_avg_out_{"exp_avg_out", 1};
    Output<Func> exp_avg_sq_out_{"exp_avg_sq_out", 1};

    Output<Func> f_output_{"f_output", 1};

    void generate() {
        Func grad("grad");
        grad(x) = gradient_(x) + weight_decay * f_(x);

        Func exp_avg_out("exp_avg_out");
        exp_avg_out(x) = exp_avg(x) * beta_1 + (1.0f - beta_1) * grad(x);

        Func exp_avg_sq_out("exp_avg_out_sq");
        exp_avg_sq_out(x) = exp_avg_sq(x) * beta_2 + (1.0f - beta_2) * grad(x) * grad(x);

        Func denom("denom");
        denom(x) = Halide::sqrt(exp_avg_sq_out(x)) + eps;

        Func bias_correction1("bias_correction1");
        Func bias_correction2("bias_correction2");
        bias_correction1() = 1.0f / (1.0f - Halide::pow(beta_1, iteration));
        bias_correction2() = 1.0f / (1.0f - Halide::pow(beta_2, iteration));

        Func adapted_learning_rate("adapted_learning_rate");
        adapted_learning_rate() = lr * bias_correction1() / Halide::sqrt(bias_correction2());

        f_output_(x) = f_(x) - adapted_learning_rate() * exp_avg_out(x) / denom(x);
        //f_output_(x) = f_(x) - lr * gradient_(x);
        //f_output_(x) = f_(x) - 0.01f;
        //f_output_(x) = gradient_(x);

        exp_avg_out_(x) = exp_avg_out(x);
        exp_avg_sq_out_(x) = exp_avg_sq_out(x);

        std::vector<Func> output_func({f_output_, exp_avg_out_, exp_avg_sq_out_});

        f_.dim(0).set_bounds_estimate(0, 100);
        gradient_.dim(0).set_bounds_estimate(0, 100);
        exp_avg.dim(0).set_bounds_estimate(0, 100);
        exp_avg_sq.dim(0).set_bounds_estimate(0, 100);

        f_output_.estimate(x, 0, 100);
        exp_avg_out_.estimate(x, 0, 100);
        exp_avg_sq_out_.estimate(x, 0, 100);

        Pipeline p(output_func);
        p.auto_schedule(this->get_target());

        /*Halide::SimpleAutoscheduleOptions options;
        options.gpu = get_target().has_gpu_feature();
        options.gpu_tile_channel = 1;
        options.unroll_rvar_size = 128;

        Halide::simple_autoschedule(
        output_func, {
            {"f_.min.0", 0},
            {"f_.extent.0", 128 * 128 * 128},
            {"exp_avg_.min.0", 0},
            {"exp_avg_.extent.0", 128 * 128 * 128},
            {"exp_avg_sq_.min.0", 0},
            {"exp_avg_sq_.extent.0", 128 * 128 * 128},
        }, {
            {
                {0, 128 * 128 * 128},
            },
            {
                {0, 128 * 128 * 128},
            },
            {
                {0, 128 * 128 * 128},
            }
        },
        options);*/
    }

  private:
    Var x{"x"};
};

HALIDE_REGISTER_GENERATOR(OptimizerGenerator, optimizer_gen)

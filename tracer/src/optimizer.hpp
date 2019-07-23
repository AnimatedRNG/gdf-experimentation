#pragma once

#include <iostream>
#include <memory>

#include "HalideBuffer.h"
#include "halide_image_io.h"

#include "optimizer_gen.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

class ADAM {
  public:

    explicit ADAM(Buffer<float>& params,
         Buffer<float>& gradient,
         const float& lr = 1e-3f,
         const float& beta_1 = 0.9f,
         const float& beta_2 = 0.99f,
         const float& weight_decay = 0.0f,
         const float& eps = 1e-8f) :
        params_(params), gradient_(gradient),
        lr_(lr), beta_1_(beta_1), beta_2_(beta_2),
        weight_decay_(weight_decay), eps_(eps), iterations(0),
        num_elems(params.number_of_elements()), sizes(),
        exp_avg_(params.number_of_elements()), exp_avg_sq_(params.number_of_elements()) {

        exp_avg_.fill(0);
        exp_avg_sq_.fill(0);

        sizes.clear();
        for (int i = 0; i < params.dimensions(); i++) {
            sizes.push_back(params.dim(i).extent());
        }
    }

    inline void step() {
        Buffer<float> params_flattened(params_.data(), {num_elems});
        Buffer<float> gradient_flattened(gradient_.data(), {num_elems});

        Buffer<float> exp_avg_out(num_elems);
        Buffer<float> exp_avg_sq_out(num_elems);

        // TODO: remove copies at some point!
        params_flattened.set_host_dirty();
        params_flattened.copy_to_device(halide_cuda_device_interface());

        gradient_flattened.set_host_dirty();
        gradient_flattened.copy_to_device(halide_cuda_device_interface());

        Buffer<float> output_params_flattened(num_elems);

        optimizer_gen(params_flattened, gradient_flattened,
                      lr_, beta_1_, beta_2_, weight_decay_, eps_,
                      ++iterations,
                      exp_avg_, exp_avg_sq_,
                      exp_avg_out, exp_avg_sq_out,
                      output_params_flattened
                     );

        // TODO: remove copies at some point!
        output_params_flattened.copy_to_host();

        Buffer<float> output_params(output_params_flattened.data(), sizes);
        params_ = output_params;

        exp_avg_ = Buffer<float>(exp_avg_out.data(), sizes);
        exp_avg_sq_ = Buffer<float>(exp_avg_sq_out.data(), sizes);
    }

  private:
    Buffer<float>& params_;
    Buffer<float>& gradient_;

    float lr_;
    float beta_1_;
    float beta_2_;
    float weight_decay_;
    float eps_;
    int iterations;

    int num_elems;
    std::vector<int> sizes;

    Buffer<float> exp_avg_;
    Buffer<float> exp_avg_sq_;
};

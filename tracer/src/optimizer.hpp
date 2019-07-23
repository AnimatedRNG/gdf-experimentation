#pragma once

#include <iostream>
#include <memory>

#include "HalideBuffer.h"
#include "halide_image_io.h"

#include "optimizer_gen.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

template<int D>
class ADAM {
  public:

    ADAM(const float& lr,
         const float& beta_1,
         const float& beta_2,
         const float& weight_decay,
         const float& eps) : lr_(lr), beta_1_(beta_1), beta_2_(beta_2), iterations(0),
        weight_decay_(weight_decay), eps_(eps),
        exp_avg_(nullptr), exp_avg_sq_(nullptr) {
    }

    Buffer<float> update(Buffer<float, D> params,
                         Buffer<float, D> gradient) {
        int num_elems = params.number_of_elements();
        std::vector<int> sizes;
        sizes.clear();
        for (int i = 0; i < D; i++) {
            sizes.push_back(params.dim(i).extent);
        }

        if (!exp_avg_ || !exp_avg_sq_) {
            this->exp_avg_ = std::shared_ptr<Buffer<float>>(new Buffer<float>(num_elems));
            this->exp_avg_sq_ = std::shared_ptr<Buffer<float>>(new Buffer<float>
                                (num_elems));

            this->exp_avg_->fill(0);
            this->exp_avg_sq_->fill(0);
        }

        Buffer<float> params_flattened(params.data(), {num_elems});
        Buffer<float> gradient_flattened(gradient.data(), {num_elems});

        Buffer<float> exp_avg_out(num_elems);
        Buffer<float> exp_avg_sq_out(num_elems);

        Buffer<float> output_params_flattened(num_elems);

        optimizer_gen(params_flattened, gradient_flattened,
                      lr_, beta_1_, beta_2_, weight_decay_, eps_,
                      ++iterations,
                      *exp_avg_, *exp_avg_sq_,
                      exp_avg_out, exp_avg_sq_out,
                      output_params_flattened
                     );

        Buffer<float> output_params(output_params_flattened.data(), sizes);
        params = std::move(output_params);

        exp_avg_.reset(new Buffer<float>(exp_avg_out.data(), {sizes}));
        exp_avg_sq_.reset(new Buffer<float>(exp_avg_sq_out.data(), {sizes}));
    }

  private:
    float lr_;
    float beta_1_;
    float beta_2_;
    float weight_decay_;
    float eps_;
    int iterations;

    std::shared_ptr<Buffer<float>> exp_avg_;
    std::shared_ptr<Buffer<float>> exp_avg_sq_;
};

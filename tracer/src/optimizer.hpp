#pragma once

#include <iostream>
#include <memory>

#include "HalideBuffer.h"
#include "halide_image_io.h"

#include "optimizer_gen.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

void test_print(void* ctx, const char* data) {
    char* casted = (char*) data;
    std::cout << casted;
}

class ADAM {
  public:

    explicit ADAM(Buffer<float>& params,
                  Buffer<float>& gradient,
                  const float& lr = 1e-3f,
                  const float& beta_1 = 0.9f,
                  const float& beta_2 = 0.99f,
                  const float& weight_decay = 0.0f,
                  const float& eps = 1e-8f) :
        interface(halide_cuda_device_interface()),
        params_(params), gradient_(gradient),
        lr_(lr), beta_1_(beta_1), beta_2_(beta_2),
        weight_decay_(weight_decay), eps_(eps), iterations(0),
        num_elems(params.number_of_elements()), sizes(),
        exp_avg_(params.number_of_elements()),
        exp_avg_sq_(params.number_of_elements()) {
        exp_avg_.fill(0);
        exp_avg_sq_.fill(0);

        exp_avg_.set_host_dirty();
        exp_avg_.copy_to_device(interface);

        exp_avg_.set_host_dirty();
        exp_avg_sq_.copy_to_device(interface);

        sizes.clear();
        for (int i = 0; i < params.dimensions(); i++) {
            sizes.push_back(params.dim(i).extent());
        }
    }

    inline void step() {
        params_.flatten();
        gradient_.flatten();

        Buffer<float> exp_avg_out(num_elems);
        Buffer<float> exp_avg_sq_out(num_elems);

        exp_avg_out.set_host_dirty(true);
        exp_avg_sq_out.set_host_dirty(true);

        exp_avg_out.device_malloc(interface);
        exp_avg_sq_out.device_malloc(interface);

        Buffer<float> output_params_flattened(num_elems);
        output_params_flattened.set_host_dirty(true);
        output_params_flattened.device_malloc(interface);

        optimizer_gen(params_, gradient_,
                      lr_, beta_1_, beta_2_, weight_decay_, eps_,
                      (float)(++iterations),
                      exp_avg_, exp_avg_sq_,
                      exp_avg_out, exp_avg_sq_out,
                      output_params_flattened
                     );

        output_params_flattened.reshape(sizes);
        params_.reshape(sizes);
        gradient_.reshape(sizes);

        output_params_flattened.reshape(sizes);
        params_ = std::move(output_params_flattened);

        interface->buffer_copy(nullptr, exp_avg_, interface, exp_avg_out);
        interface->buffer_copy(nullptr, exp_avg_sq_, interface, exp_avg_sq_out);
    }

  private:
    const halide_device_interface_t* interface;

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

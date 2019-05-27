#include <iostream>

#include "Halide.h"

#include "matmul.hpp"

#include <stdio.h>

using namespace Halide;

Var x("x"), y("y"), c("c");

class TracerGenerator : public Halide::Generator<TracerGenerator> {
  public:

    //Input<Buffer<float>> in_{"in" , 2};
    Output<Buffer<float>> out_{"out", 3};

    Func projection(Func proj_matrix,
                    Func view_matrix) {

    }

    void generate() {

        Buffer<float> a(4, 4);
        Buffer<float> b(4, 4);

        auto identity = [](Buffer<float>& a) {
            return [&a](int x, int y) {
                if (x == y) {
                    a(x, y) = 1.0f;
                } else {
                    a(x, y) = 0.0;
                }
            };
        };

        a.for_each_element(identity(a));

        a(0, 1) = 2.0f;
        a(0, 2) = 3.0f;
        a(0, 3) = 4.0f;

        a(1, 2) = 2.0f;
        a(1, 3) = 3.0f;
        a(1, 0) = 4.0f;

        a(2, 3) = 2.0f;
        a(2, 0) = 3.0f;
        a(2, 1) = 4.0f;

        a(3, 0) = 2.0f;
        a(3, 1) = 3.0f;
        a(3, 2) = 4.0f;

        b.for_each_element(identity(b));

        //out_(x, y, c) =
        //    matmul::product(Func(a), Func(b))(x, y);
        //out_(x, y, c) = matmul::inverse(Func(a))(x, y);
    }
};

HALIDE_REGISTER_GENERATOR(TracerGenerator, tracer_render)

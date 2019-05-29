#include <iostream>

#include "HalideBuffer.h"
#include "halide_image_io.h"
#include "matplotlibcpp.h"

#include "tracer_render.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

namespace plt = matplotlibcpp;

int main() {
    float projection_matrix[4][4] = {
        {0.75, 0.0, 0.0,  0.0},
        {0.0,  1.0, 0.0,  0.0},
        {0.0,  0.0, 1.0,  0.0},
        {0.0,  -0.2, 1.0,  0.0}
    };

    float view_matrix[4][4] = {
        {-0.9825, 0.1422, 0.1206, 0.0},
        {0.0, 0.6469, -0.7626, 0.0},
        {0.1865, 0.7492, 0.6356, 0.0},
        {-0.3166, 1.1503, 8.8977, 1.0}
    };

    Buffer<float> projection(projection_matrix);
    Buffer<float> view(view_matrix);

    Buffer<float> output(4, 4, 1);

    tracer_render(projection, view, output);

    //plt::show();
}

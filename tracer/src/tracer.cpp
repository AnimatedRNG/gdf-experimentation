#include <iostream>

#include "HalideBuffer.h"
#include "halide_image_io.h"
#include "matplotlibcpp.h"

#include "tracer_render.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

namespace plt = matplotlibcpp;

int main() {
    const float projection_matrix[4][4] = {
        {0.75, 0.0, 0.0,  0.0},
        {0.0,  1.0, 0.0,  0.0},
        {0.0,  0.0, 1.0002,  1.0},
        {0.0,  0.0, -0.2,  0.0}
    };

    const float view_matrix[4][4] = {
        {-0.9825, 0.1422, 0.1206, 0.0},
        {0.0, 0.6469, -0.7626, 0.0},
        {0.1865, 0.7492, 0.6356, 0.0},
        {-0.3166, 1.1503, 8.8977, 1.0}
    };

    int width = 100;
    int height = 100;

    Buffer<float> projection(projection_matrix);
    Buffer<float> view(view_matrix);
    Buffer<float> output(width, height, 3);

    tracer_render(projection, view, width, height, output);

    convert_and_save_image(output, "test.png");

    //plt::show();
}

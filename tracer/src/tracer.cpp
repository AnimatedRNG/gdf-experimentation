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

    int width = 200;
    int height = 200;

    Buffer<float> projection(projection_matrix);
    Buffer<float> view(view_matrix);
    Buffer<float> output(width, height, 3);
    Buffer<float> debug(4, 4);

    tracer_render(projection, view, width, height, output, debug);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << debug(i, j) << " ";
        }
        std::cout << std::endl;
    }

    /*for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout
                    << "("
                    << output(i, j, 0) << ", "
                    << output(i, j, 1) << ", "
                    << output(i, j, 2) << ", ";
                //<< output(i, j, 3) << ") ";
        }
        std::cout << std::endl;
        }*/

    convert_and_save_image(output, "test.png");

    //plt::show();
}

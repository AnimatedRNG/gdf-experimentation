#include <iostream>

#include "HalideBuffer.h"
#include "halide_image_io.h"

#include "tracer_render.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

int main() {
    Buffer<float> output(4, 4, 1);

    tracer_render(output);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << output(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

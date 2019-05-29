#include <iostream>

#include "HalideBuffer.h"
#include "halide_image_io.h"
#include "matplotlibcpp.h"

#include "tracer_render.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

namespace plt = matplotlibcpp;

bool gluInvertMatrix(const float mat[4][4], float invOut[4][4]) {
    float inv[4][4], det;
    int i;

    inv[0][0] = mat[1][1]  * mat[2][2] * mat[3][3] -
                mat[1][1]  * mat[3][2] * mat[2][3] -
                mat[1][2]  * mat[2][1]  * mat[3][3] +
                mat[1][2]  * mat[3][1]  * mat[2][3] +
                mat[1][3] * mat[2][1]  * mat[3][2] -
                mat[1][3] * mat[3][1]  * mat[2][2];

    inv[0][1] = -mat[0][1]  * mat[2][2] * mat[3][3] +
                mat[0][1]  * mat[3][2] * mat[2][3] +
                mat[0][2]  * mat[2][1]  * mat[3][3] -
                mat[0][2]  * mat[3][1]  * mat[2][3] -
                mat[0][3] * mat[2][1]  * mat[3][2] +
                mat[0][3] * mat[3][1]  * mat[2][2];

    inv[0][2] = mat[0][1]  * mat[1][2] * mat[3][3] -
                mat[0][1]  * mat[3][2] * mat[1][3] -
                mat[0][2]  * mat[1][1] * mat[3][3] +
                mat[0][2]  * mat[3][1] * mat[1][3] +
                mat[0][3] * mat[1][1] * mat[3][2] -
                mat[0][3] * mat[3][1] * mat[1][2];

    inv[0][3] = -mat[0][1]  * mat[1][2] * mat[2][3] +
                mat[0][1]  * mat[2][2] * mat[1][3] +
                mat[0][2]  * mat[1][1] * mat[2][3] -
                mat[0][2]  * mat[2][1] * mat[1][3] -
                mat[0][3] * mat[1][1] * mat[2][2] +
                mat[0][3] * mat[2][1] * mat[1][2];

    inv[1][0] = -mat[1][0]  * mat[2][2] * mat[3][3] +
                mat[1][0]  * mat[3][2] * mat[2][3] +
                mat[1][2]  * mat[2][0] * mat[3][3] -
                mat[1][2]  * mat[3][0] * mat[2][3] -
                mat[1][3] * mat[2][0] * mat[3][2] +
                mat[1][3] * mat[3][0] * mat[2][2];

    inv[1][1] = mat[0][0]  * mat[2][2] * mat[3][3] -
                mat[0][0]  * mat[3][2] * mat[2][3] -
                mat[0][2]  * mat[2][0] * mat[3][3] +
                mat[0][2]  * mat[3][0] * mat[2][3] +
                mat[0][3] * mat[2][0] * mat[3][2] -
                mat[0][3] * mat[3][0] * mat[2][2];

    inv[1][2] = -mat[0][0]  * mat[1][2] * mat[3][3] +
                mat[0][0]  * mat[3][2] * mat[1][3] +
                mat[0][2]  * mat[1][0] * mat[3][3] -
                mat[0][2]  * mat[3][0] * mat[1][3] -
                mat[0][3] * mat[1][0] * mat[3][2] +
                mat[0][3] * mat[3][0] * mat[1][2];

    inv[1][3] = mat[0][0]  * mat[1][2] * mat[2][3] -
                mat[0][0]  * mat[2][2] * mat[1][3] -
                mat[0][2]  * mat[1][0] * mat[2][3] +
                mat[0][2]  * mat[2][0] * mat[1][3] +
                mat[0][3] * mat[1][0] * mat[2][2] -
                mat[0][3] * mat[2][0] * mat[1][2];

    inv[2][0] = mat[1][0]  * mat[2][1] * mat[3][3] -
                mat[1][0]  * mat[3][1] * mat[2][3] -
                mat[1][1]  * mat[2][0] * mat[3][3] +
                mat[1][1]  * mat[3][0] * mat[2][3] +
                mat[1][3] * mat[2][0] * mat[3][1] -
                mat[1][3] * mat[3][0] * mat[2][1];

    inv[2][1] = -mat[0][0]  * mat[2][1] * mat[3][3] +
                mat[0][0]  * mat[3][1] * mat[2][3] +
                mat[0][1]  * mat[2][0] * mat[3][3] -
                mat[0][1]  * mat[3][0] * mat[2][3] -
                mat[0][3] * mat[2][0] * mat[3][1] +
                mat[0][3] * mat[3][0] * mat[2][1];

    inv[2][2] = mat[0][0]  * mat[1][1] * mat[3][3] -
                mat[0][0]  * mat[3][1] * mat[1][3] -
                mat[0][1]  * mat[1][0] * mat[3][3] +
                mat[0][1]  * mat[3][0] * mat[1][3] +
                mat[0][3] * mat[1][0] * mat[3][1] -
                mat[0][3] * mat[3][0] * mat[1][1];

    inv[2][3] = -mat[0][0]  * mat[1][1] * mat[2][3] +
                mat[0][0]  * mat[2][1] * mat[1][3] +
                mat[0][1]  * mat[1][0] * mat[2][3] -
                mat[0][1]  * mat[2][0] * mat[1][3] -
                mat[0][3] * mat[1][0] * mat[2][1] +
                mat[0][3] * mat[2][0] * mat[1][1];

    inv[3][0] = -mat[1][0] * mat[2][1] * mat[3][2] +
                mat[1][0] * mat[3][1] * mat[2][2] +
                mat[1][1] * mat[2][0] * mat[3][2] -
                mat[1][1] * mat[3][0] * mat[2][2] -
                mat[1][2] * mat[2][0] * mat[3][1] +
                mat[1][2] * mat[3][0] * mat[2][1];

    inv[3][1] = mat[0][0] * mat[2][1] * mat[3][2] -
                mat[0][0] * mat[3][1] * mat[2][2] -
                mat[0][1] * mat[2][0] * mat[3][2] +
                mat[0][1] * mat[3][0] * mat[2][2] +
                mat[0][2] * mat[2][0] * mat[3][1] -
                mat[0][2] * mat[3][0] * mat[2][1];

    inv[3][2] = -mat[0][0] * mat[1][1] * mat[3][2] +
                mat[0][0] * mat[3][1] * mat[1][2] +
                mat[0][1] * mat[1][0] * mat[3][2] -
                mat[0][1] * mat[3][0] * mat[1][2] -
                mat[0][2] * mat[1][0] * mat[3][1] +
                mat[0][2] * mat[3][0] * mat[1][1];

    inv[3][3] = mat[0][0] * mat[1][1] * mat[2][2] -
                mat[0][0] * mat[2][1] * mat[1][2] -
                mat[0][1] * mat[1][0] * mat[2][2] +
                mat[0][1] * mat[2][0] * mat[1][2] +
                mat[0][2] * mat[1][0] * mat[2][1] -
                mat[0][2] * mat[2][0] * mat[1][1];

    det = 1.0f / (mat[0][0] * inv[0][0] +
                  mat[1][0] * inv[0][1] +
                  mat[2][0] * inv[0][2] +
                  mat[3][0] * inv[0][3]);

    for (i = 0; i < 16; i++)
        invOut[i % 4][i / 4] = inv[i % 4][i / 4] * det;

    return true;
}

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

    Buffer<float> projection(projection_matrix);
    Buffer<float> view(view_matrix);
    Buffer<float> output(200, 200, 4);
    Buffer<float> debug(4, 4);

    tracer_render(projection, view, 200, 200, output, debug);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << debug(i, j) << " ";
        }
        std::cout << std::endl;
    }

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout
                    << "("
                    << output(i, j, 0) << ", "
                    << output(i, j, 1) << ", "
                    << output(i, j, 2) << ", "
                    << output(i, j, 3) << ") ";
        }
        std::cout << std::endl;
    }

    /*float op[4][4];
    gluInvertMatrix(view_matrix, op);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << op[j][i] << " ";
        }
        std::cout << std::endl;
        }*/

    convert_and_save_image(output, "test.png");

    //plt::show();
}

#include <iostream>
#include <chrono>
#include <math.h>

#include "HalideBuffer.h"
#include "halide_image_io.h"
#include "HalideRuntime.h"
#include "HalideRuntimeCuda.h"

#include "gif.h"

#include "tracer_render.h"
#include "sdf_gen.h"
#include "optimizer.hpp"
#include "optimizer_gen.h"
#include "read_sdf.hpp"
#include "debug.hpp"

using namespace Halide::Runtime;
using namespace Halide::Tools;

void write_gifs(
    Buffer<float> buf,
    int iterations,
    int num_gifs = 1,
    uint32_t delay = 1) {

    int width = buf.dim(2).max() + 1;
    int height = buf.dim(3).max() + 1;
    int stride = width * height * 4;
    int gif_stride = stride * iterations;
    GifWriter g;

    uint8_t* buffer = new uint8_t[width * height * 4];
    for (int gif = 0; gif < num_gifs; gif++) {
        std::string filename = std::to_string(gif) + ".gif";
        GifBegin(&g, filename.c_str(), width, height, delay);
        for (int i = 0; i < iterations; i++) {
            //std::cout << "Writing frame " << i + 1 << " of " << filename << std::endl;
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    for (int c = 0; c < 4; c++) {
                        buffer[x * height * 4 + y * 4 + c] =
                            (uint8_t)(buf(gif, i, x, y, c) * 255.0f);
                        //std::cout << buf(gif, i, x, y, c) << std::endl;
                    }
                }
            }
            GifWriteFrame(&g,
                          buffer,
                          width, height, delay);
        }
        std::cout << "Finished writing 900 frames of " << filename << std::endl;
        GifEnd(&g);
    }
    free(buffer);
}

void mat_mul(float m1[4][4], float m2[4][4], float out[4][4]) {
    float tmp[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            tmp[i][j] = 0.0f;
            for (int k = 0; k < 4; k++) {
                tmp[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            out[i][j] = tmp[i][j];
        }
    }
}

void identity(float matrix[4][4]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i == j) {
                matrix[i][j] = 1.0f;
            } else {
                matrix[i][j] = 0.0f;
            }
        }
    }
}

void apply_rotation(float matrix[4][4], float heading, float attitude, float bank) {
    float ch = cos(heading);
    float sh = sin(heading);
    float ca = cos(attitude);
    float sa = sin(attitude);
    float cb = cos(bank);
    float sb = sin(bank);

    float m2[4][4] = {0.0f};
    m2[0][0] = ch * ca;
    m2[1][0] = sh*sb - ch*sa*cb;
    m2[2][0] = ch*sa*sb + sh*cb;
    m2[0][1] = sa;
    m2[1][1] = ca*cb;
    m2[2][1] = -ca*sb;
    m2[0][2] = -sh*ca;
    m2[1][2] = sh*sa*cb + ch*sb;
    m2[2][2] = -sh*sa*sb + ch*cb;
    m2[3][3] = 1.0f;

    mat_mul(matrix, m2, matrix);
}

void apply_translation(float matrix[4][4], float x, float y, float z) {
    float m2[4][4] = {0.0f};

    m2[3][0] = x;
    m2[3][1] = y;
    m2[3][2] = z;

    m2[0][0] = 1.0f;
    m2[1][1] = 1.0f;
    m2[2][2] = 1.0f;
    m2[3][3] = 1.0f;

    mat_mul(matrix, m2, matrix);
}

void print_matrix(Buffer<float> matrix) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << matrix(i, j) << " ";
        }
        std::cout << std::endl;
    }
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

    float target_transform_[4][4] = {0.0f};
    identity(target_transform_);
    apply_translation(target_transform_, 1.0f, 0.0f, 0.0f);
    //apply_rotation(target_transform_, 3.14f / 4.0f, 0.0f, 0.0f);

    /*const float model_translation_[3] = {
        0.0f, 0.0f, 0.0f
    };
    const float target_translation_[3] = {
        -1.0f, 0.0f, 2.0f
        };*/
    float model_transform_[4][4] = {0.0f};
    identity(model_transform_);

    const int32_t n_matrix[3] = {
        128, 128, 128
    };

    int width = 100;
    int height = 100;
    int iterations = 900;

    //float p0_x, p0_y, p0_z, p1_x, p1_y, p1_z;
    float p0_1_matrix[3];
    float p1_1_matrix[3];
    int32_t n_1_matrix[3];

    Buffer<float> lucy = read_sdf("bunny.sdf",
                                  p0_1_matrix[0], p0_1_matrix[1], p0_1_matrix[2],
                                  p1_1_matrix[0], p1_1_matrix[1], p1_1_matrix[2],
                                  n_1_matrix[0], n_1_matrix[1], n_1_matrix[2],
                                  true, true, 4.0f);

    Buffer<float> projection(projection_matrix);
    Buffer<float> view(view_matrix);
    Buffer<float> sdf_model(n_matrix[0], n_matrix[1], n_matrix[2]);
    Buffer<float> sdf_target(n_matrix[0], n_matrix[1], n_matrix[2]);
    Buffer<float> p0(3);
    Buffer<float> p1(3);
    Buffer<float> p0_1(3);
    Buffer<float> p1_1(3);
    Buffer<int32_t> n(n_matrix);
    Buffer<float> target_(width, height, 3);
    Buffer<float> loss_(1);
    Buffer<float> forward_(width, height, 3);
    Buffer<float> d_l_sdf_(n_matrix[0], n_matrix[1], n_matrix[2]);
    Buffer<float> d_l_transform_(4, 4);

    //Buffer<float> model_translation(model_translation_);
    //Buffer<float> target_translation(target_translation_);
    Buffer<float> model_transform(model_transform_);
    Buffer<float> target_transform(target_transform_);

    //ADAM adam(model_translation, d_l_translation_);
    ADAM adam(model_transform, d_l_transform_, 1e-2);

    projection.set_host_dirty();
    projection.copy_to_device(halide_cuda_device_interface());

    view.set_host_dirty();
    view.copy_to_device(halide_cuda_device_interface());

    //target_translation.set_host_dirty();
    //target_translation.copy_to_device(halide_cuda_device_interface());
    target_transform.set_host_dirty();
    target_transform.copy_to_device(halide_cuda_device_interface());

    sdf_model.set_host_dirty();
    sdf_model.copy_to_device(halide_cuda_device_interface());

    sdf_target.set_host_dirty();
    sdf_target.copy_to_device(halide_cuda_device_interface());

    loss_.set_host_dirty();
    loss_.copy_to_device(halide_cuda_device_interface());

    p0.set_host_dirty();
    p1.copy_to_device(halide_cuda_device_interface());

    auto start = std::chrono::steady_clock::now();
    sdf_gen(n_matrix[0], n_matrix[1], n_matrix[2], sdf_model, p0, p1);
    sdf_gen(n_matrix[0], n_matrix[1], n_matrix[2], sdf_target, p0, p1);

    auto end = std::chrono::steady_clock::now();

    auto diff = end - start;
    std::cout << "Generated 128x128x128 SDF in "
              << std::chrono::duration <float, std::milli> (diff).count()
              << " ms"
              << std::endl << std::endl;

#ifdef DEBUG_TRACER
    Buffer<float> debug(8, iterations, width, height, 4);
    Buffer<int32_t> num_debug(1);
#endif //DEBUG_TRACER

    start = std::chrono::steady_clock::now();

    tracer_render(projection, view, target_transform,
                  sdf_target, p0, p1,
                  forward_, // dummy placeholder
                  width, height, 0,
                  loss_,
                  target_, d_l_sdf_, d_l_transform_
#ifdef DEBUG_TRACER
                  , debug, num_debug
#endif //DEBUG_TRACER
                 );

    end = std::chrono::steady_clock::now();
    diff = end - start;

    std::cout << "rendered target; now going to begin optimization" << std::endl;

    std::cout
            << "target tracing took "
            << std::chrono::duration <float, std::milli> (diff).count()
            << " ms"
            << std::endl << std::endl;

    sdf_model.set_host_dirty();
    sdf_model.copy_to_device(halide_cuda_device_interface());
    for (int epoch = 0; epoch < 900; epoch++) {
        std::cout << "epoch " << epoch << std::endl;
        //model_translation.set_host_dirty();
        //model_translation.copy_to_device(halide_cuda_device_interface());
        model_transform.set_host_dirty();
        model_transform.copy_to_device(halide_cuda_device_interface());

        forward_.copy_to_device(halide_cuda_device_interface());

        start = std::chrono::steady_clock::now();
        tracer_render(projection, view, model_transform,
                      sdf_model, p0, p1,
                      target_,
                      width, height, 0,
                      loss_,
                      forward_, d_l_sdf_, d_l_transform_
#ifdef DEBUG_TRACER
                      , debug, num_debug
#endif //DEBUG_TRACER
                     );
        end = std::chrono::steady_clock::now();
        diff = end - start;

        std::cout << "done with rendering; copying back now" << std::endl;

        adam.step();

        forward_.copy_to_host();
        target_.copy_to_host();
        d_l_sdf_.copy_to_host();
        d_l_transform_.copy_to_host();

        //model_translation.copy_to_host();
        model_transform.copy_to_host();

        loss_.set_host_dirty();
        loss_.copy_to_device(halide_cuda_device_interface());

        std::cout << "loss " << loss_(0) << std::endl;

        /*std::cout << "d_l_translation " << d_l_translation_(0) << " "
                  << d_l_translation_(1) << " "
                  << d_l_translation_(2) << std::endl;*/
        print_matrix(d_l_transform_);

        d_l_transform_.set_host_dirty();
        d_l_transform_.copy_to_device(halide_cuda_device_interface());

        /*std::cout << "model_translation " << model_translation(0) << " "
                  << model_translation(1) << " "
                  << model_translation(2) << std::endl;*/
        print_matrix(model_transform);

#ifdef DEBUG_TRACER
        debug.copy_to_host();
        num_debug.copy_to_host();

        std::cout << "num_debug " << num_debug(0) << std::endl;
#endif //DEBUG_TRACER

        std::cout
                << "tracing took "
                << std::chrono::duration <float, std::milli> (diff).count()
                << " ms"
                << std::endl << std::endl;

#ifdef DEBUG_TRACER
        write_gifs(debug, iterations, num_debug(0));
#endif // DEBUG_TRACER

        convert_and_save_image(target_, "target.png");
        convert_and_save_image(forward_, "model/model_" + std::to_string(epoch) + ".png");
    }
}

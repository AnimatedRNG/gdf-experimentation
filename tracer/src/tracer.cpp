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
#include "fmm_gen.h"
#include "optimizer.hpp"
#include "optimizer_gen.h"
#include "read_sdf.hpp"
#include "debug.hpp"
#include "buffer_utils.hpp"

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

void debug_sdf(const std::string& directory,
               Buffer<float>& sdf,
               const halide_device_interface_t* interface) {
    to_host(sdf);
    std::vector<int> sizes = {sdf.dim(0).extent(), sdf.dim(1).extent()};

    for (int z = 0; z < sdf.dim(2).extent(); z++) {
        Buffer<float> buf(sizes);
        buf.copy_from(sdf.sliced(2, z));
        buf.for_each_value([] (float& v) {
            v = (v + 4.0f) / 8.0f;
        });
        convert_and_save_image(buf,
            directory + "/" + std::to_string(z) + ".png");
    }

    to_device(sdf, interface);
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

void print_matrix(const Buffer<float>& matrix) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << matrix(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

// this is awful
void copy(const Buffer<float>& a, Buffer<float>& b) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            b(i, j) = a(i, j);
        }
    }
}

int main() {
    const halide_device_interface_t* interface = halide_cuda_device_interface();

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
    //apply_translation(target_transform_, 1.0f, 0.0f, 0.0f);
    //apply_rotation(target_transform_, 0.0f, 0.5f, 0.0f);
    //apply_translation(target_transform_, 0.0f, 2.0f, -3.0f);

    /*const float model_translation_[3] = {
        0.0f, 0.0f, 0.0f
    };
    const float target_translation_[3] = {
        -1.0f, 0.0f, 2.0f
        };*/
    float model_transform_[4][4] = {0.0f};
    identity(model_transform_);

    const int32_t n_matrix[3] = {
        64, 64, 64
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
    Buffer<float> sdf_model_2(n_matrix[0], n_matrix[1], n_matrix[2]);
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
    //ADAM adam(model_transform, d_l_transform_, 1e-2f);

    to_device(projection, interface);

    to_device(view, interface);

    to_device(target_transform, interface);

    to_device(sdf_model, interface);
    to_device(sdf_model_2, interface);

    to_device(sdf_target, interface);

    //to_device(loss_, interface);

    ADAM adam(sdf_model, d_l_sdf_, 1e-3f);

    to_device(p0, interface);
    to_device(p1, interface);

    auto start = std::chrono::steady_clock::now();
    sdf_gen(n_matrix[0], n_matrix[1], n_matrix[2], 0, sdf_model, p0, p1);
    sdf_gen(n_matrix[0], n_matrix[1], n_matrix[2], 1, sdf_target, p0, p1);

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

    //to_device(sdf_model, interface);
    for (int epoch = 0; epoch < 900; epoch++) {
        std::cout << "epoch " << epoch << std::endl;

        /*std::cout << "before render " <<
            (float*) ((halide_buffer_t*) model_transform)->host << " " << std::endl;
            print_matrix(model_transform);

        Buffer<float> model_transform_copy(4, 4);
        copy(model_transform, model_transform_copy);*/

        to_device(model_transform, interface);
        //to_device(model_transform_copy, interface);
        to_device(forward_, interface);

        std::cout << "reinitializing distance field using FMM..." << std::endl;
        start = std::chrono::steady_clock::now();
        fmm_gen(sdf_model,
                p0, p1,
                n_matrix[0], n_matrix[1], n_matrix[2],
                sdf_model_2);
        // I would use std::swap, but that would mess with the optimizer :/
        interface->buffer_copy(nullptr, sdf_model_2, interface, sdf_model);
        end = std::chrono::steady_clock::now();
        diff = end - start;
        std::cout << "completed FMM in "
                  << std::chrono::duration <float, std::milli> (diff).count()
                  << " ms" << std::endl;
        //debug_sdf("sdf_uncorrected", sdf_model, interface);

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

        debug_sdf("d_l_sdf", d_l_sdf_, interface);

        //to_host(model_transform);
        //model_transform.set_host_dirty();
        //model_transform.copy_to_device(halide_cuda_device_interface());

        std::cout << "done with rendering; copying back now" << std::endl;

        //to_host(d_l_transform_);

        adam.step();

        to_host(forward_);
        to_host(target_);

        to_host(d_l_sdf_);
        float prod = 0.0f;
        for (int i = 0; i < n_matrix[0]; i++) {
            for (int j = 0; j < n_matrix[1]; j++) {
                for (int k = 0; k < n_matrix[2]; k++) {
                    prod += d_l_sdf_(i, j, k);
                }
            }
        }
        std::cout << "loss mag " << prod << std::endl;
        to_device(d_l_sdf_, interface);

        //to_host(loss_);

        std::cout << "loss " << loss_(0) << std::endl;

        //to_device(loss_, interface);

        /*std::cout << "d_l_translation " << d_l_translation_(0) << " "
                  << d_l_translation_(1) << " "
                  << d_l_translation_(2) << std::endl;*/
        //std::cout << "d_l_transform: " << std::endl;
        //print_matrix(d_l_transform_);

        //to_device(d_l_transform_, interface);

        /*std::cout << "model_translation " << model_translation(0) << " "
                  << model_translation(1) << " "
                  << model_translation(2) << std::endl;*/
        std::cout << "model: " << std::endl;
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

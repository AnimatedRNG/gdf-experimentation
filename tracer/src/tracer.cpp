#include <iostream>
#include <chrono>

#include "HalideBuffer.h"
#include "halide_image_io.h"
#include "HalideRuntime.h"
#include "HalideRuntimeCuda.h"

#include "gif.h"

#include "tracer_render.h"
#include "sdf_gen.h"
#include "read_sdf.hpp"

using namespace Halide::Runtime;
using namespace Halide::Tools;

void write_gifs(
    Buffer<uint8_t> buf,
    int iterations,
    int num_gifs = 1,
    float delay = 0.0001f) {

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
                        buffer[x * height * 4 + y * 4 + c] = buf(gif, i, x, y, c);
                    }
                }
            }
            GifWriteFrame(&g,
                          buffer,
                          width, height, delay);
        }
        std::cout << "Finished writing 300 frames of " << filename << std::endl;
        GifEnd(&g);
    }
    free(buffer);
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

    const int32_t n_matrix[3] = {
        32, 32, 32
    };
    //int n_matrix[3];

    int width = 150;
    int height = 150;
    int iterations = 400;

    float p0_matrix[3];
    float p1_matrix[3];
    /*Buffer<float> lucy = read_sdf("bunny.sdf",
                                  p0_matrix[0], p0_matrix[1], p0_matrix[2],
                                  p1_matrix[0], p1_matrix[1], p1_matrix[2],
                                  n_matrix[0], n_matrix[1], n_matrix[2],
                                  true, true, 10.0f);*/

    Buffer<float> projection(projection_matrix);
    Buffer<float> view(view_matrix);
    Buffer<float> sdf(n_matrix[0], n_matrix[1], n_matrix[2]);
    Buffer<float> p0(3);
    Buffer<float> p1(3);
    //Buffer<float> p0(p0_matrix);
    //Buffer<float> p1(p1_matrix);
    //Buffer<int32_t> n(n_matrix);
    Buffer<int32_t> n(n_matrix);
    Buffer<float> output(width, height, 3);
    Buffer<uint8_t> debug(10, iterations, width, height, 4);
    Buffer<int32_t> num_debug(1);

    projection.set_host_dirty();
    projection.copy_to_device(halide_cuda_device_interface());

    view.set_host_dirty();
    view.copy_to_device(halide_cuda_device_interface());

    sdf.set_host_dirty();
    sdf.copy_to_device(halide_cuda_device_interface());

    p0.set_host_dirty();
    p1.copy_to_device(halide_cuda_device_interface());

    auto start = std::chrono::steady_clock::now();
    sdf_gen(n_matrix[0], n_matrix[1], n_matrix[2], sdf, p0, p1);

    auto end = std::chrono::steady_clock::now();

    auto diff = end - start;
    std::cout << "Generated 128x128x128 SDF in "
              << std::chrono::duration <float, std::milli> (diff).count()
              << " ms"
              << std::endl;

    start = std::chrono::steady_clock::now();
    tracer_render(projection, view,
                  sdf, p0, p1,
                  true,
                  width, height,
                  0,
                  output, debug, num_debug);
    end = std::chrono::steady_clock::now();
    diff = end - start;

    std::cout << "done with rendering; copying back now" << std::endl;

    output.copy_to_host();
    debug.copy_to_host();
    num_debug.copy_to_host();

    std::cout
            << "tracing took "
            << std::chrono::duration <float, std::milli> (diff).count()
            << " ms"
            << std::endl;

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            std::cout << output(x, y, 0) << std::endl;
        }
    }

    write_gifs(debug, iterations, num_debug(0));

    convert_and_save_image(output, "test.png");
}

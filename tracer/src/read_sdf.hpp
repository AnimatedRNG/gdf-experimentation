#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "HalideBuffer.h"
#include "HalideRuntime.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

inline Buffer<float> read_sdf(const std::string& filename,
                              float& p0_x, float& p0_y, float& p0_z,
                              float& p1_x, float& p1_y, float& p1_z,
                              int& nx, int& ny, int& nz,
                              const bool& verbose = true,
                              const bool& rescale = true,
                              const float& scale_to = 1.0f) {
    auto error = [filename](const std::string & a) {
        throw std::runtime_error("error while parsing " + filename + ": " + a);
    };

    float dx;

    std::string line;

    std::ifstream infile(filename);

    std::getline(infile, line);
    std::istringstream line_n(line);
    if (!(line_n >> nx >> ny >> nz)) {
        error("invalid line 1");
    }

    std::getline(infile, line);
    std::istringstream line_p0(line);
    if (!(line_p0 >> p0_x >> p0_y >> p0_z)) {
        error("invalid line 2");
    }

    std::getline(infile, line);
    std::istringstream line_dx(line);
    if (!(line_dx >> dx)) {
        error("invalid line 3");
    }

    if (dx < 0.0f) {
        error("dx must be greater than 0.0f!");
    }

    if (nx < 0 || ny < 0 || nz < 0) {
        error("SDF dimensions must all be positive!");
    }

    p1_x = p0_x + dx * (float) nx;
    p1_y = p0_y + dx * (float) ny;
    p1_z = p0_z + dx * (float) nz;

    if (verbose) {
        std::cout << "Before: (" << p0_x << ", " << p0_y << ", " << p0_z << ") -> (" <<
                  p1_x << ", " << p1_y << ", " << p1_z << ") | " <<
                  nx << ", " << ny << ", " << nz << std::endl;
    }

    float scale_factor = 1.0f;
    if (rescale) {
        p1_x = (p1_x - p0_x) / 2.0f;
        p1_y = (p1_y - p0_y) / 2.0f;
        p1_z = (p1_z - p0_z) / 2.0f;

        p0_x = -p1_x;
        p0_y = -p1_y;
        p0_z = -p1_z;

        float extent = std::max(p1_x, std::max(p1_y, p1_z)) * 2.0f;
        scale_factor = scale_to / extent;

        p1_x *= scale_factor;
        p1_y *= scale_factor;
        p1_z *= scale_factor;

        p0_x *= scale_factor;
        p0_y *= scale_factor;
        p0_z *= scale_factor;
    }

    if (verbose) {
        std::cout << "After: (" << p0_x << ", " << p0_y << ", " << p0_z << ") -> (" <<
                  p1_x << ", " << p1_y << ", " << p1_z << ") | " <<
                  nx << ", " << ny << ", " << nz << std::endl;
    }

    Buffer<float> sdf_buffer(nx, ny, nz);

    int n = 0;
    float sdf_val;
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                std::getline(infile, line);
                std::istringstream line_val(line);
                if (!(line_val >> sdf_val)) {
                    error("malformed line " + std::to_string(n++ + 3 + 1));
                }
                sdf_buffer(i, j, k) = sdf_val * scale_factor;
            }
        }
    }

    return sdf_buffer;
}

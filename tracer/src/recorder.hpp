#pragma once

#include "Halide.h"

using namespace Halide;

inline void _record(Func f,
                    GeneratorOutput<Func>& debug_,
                    GeneratorOutput<Func>& num_debug,
                    GeneratorInput<int32_t>& initial_debug,
                    int& current_debug,
                    const std::string& visualization_type = "standard") {
    Var x("x"), y("y"), c("c"), t("t");

    Func _realize("_realize_" + f.name());
    _realize(t, x, y, c) = cast<uint8_t>(0);

    Func f_p("f_p");
    f_p(x, y, t, c) = 0.0f;
    if (visualization_type == "standard") {
        if (f.outputs() > 1) {
            f_p(x, y, t, 0) = f(x, y, t)[0];
            f_p(x, y, t, 1) = f(x, y, t)[1];
            f_p(x, y, t, 2) = f(x, y, t)[2];
        } else {
            f_p(x, y, t, c) = f(x, y, t);
        }
    } else if (visualization_type == "abs") {
        if (f.outputs() > 1) {
            f_p(x, y, t, 0) = Halide::abs(f(x, y, t)[0]);
            f_p(x, y, t, 1) = Halide::abs(f(x, y, t)[1]);
            f_p(x, y, t, 2) = Halide::abs(f(x, y, t)[2]);
        } else {
            f_p(x, y, t, c) = Halide::abs(f(x, y, t));
        }
    } else if (visualization_type == "log") {
        if (f.outputs() > 1) {
            f_p(x, y, t, 0) = Halide::abs(Halide::log(Halide::abs(f(x, y, t)[0]))) / 1e2f;
            f_p(x, y, t, 1) = Halide::abs(Halide::log(Halide::abs(f(x, y, t)[1]))) / 1e2f;
            f_p(x, y, t, 2) = Halide::abs(Halide::log(Halide::abs(f(x, y, t)[2]))) / 1e2f;
        } else {
            f_p(x, y, t, c) = Halide::abs(Halide::log(Halide::abs(f(x, y, t)))) / 1e2f;
        }
    } else if (visualization_type == "exists") {
        if (f.outputs() > 1) {
            f_p(x, y, t, 0) = select(f(x, y, t)[0] != 0.0f, 1.0f, 0.0f);
            f_p(x, y, t, 1) = select(f(x, y, t)[1] != 0.0f, 1.0f, 0.0f);
            f_p(x, y, t, 2) = select(f(x, y, t)[2] != 0.0f, 1.0f, 0.0f);
        } else {
            f_p(x, y, t, c) = select(f(x, y, t) != 0.0f, 1.0f, 0.0f);
        }
    }

    _realize(t, x, y, 2) = cast<uint8_t>(clamp(f_p(x, y, t, 0), 0.0f,
                                         1.0f) * 255.0f);
    _realize(t, x, y, 1) = cast<uint8_t>(clamp(f_p(x, y, t, 1), 0.0f,
                                         1.0f) * 255.0f);
    _realize(t, x, y, 0) = cast<uint8_t>(clamp(f_p(x, y, t, 2), 0.0f,
                                         1.0f) * 255.0f);

    debug_(initial_debug + current_debug++, t, x, y, c) = _realize(t, x, y, c);
}

#pragma once

#include "Halide.h"

using namespace Halide;

inline void _record(Func f,
                    GeneratorOutput<Func>& debug_,
                    GeneratorOutput<Func>& num_debug,
                    GeneratorInput<int32_t>& initial_debug,
                    int& current_debug) {
    Var x("x"), y("y"), c("c"), t("t");

    Func _realize("_realize_" + f.name());
    _realize(t, x, y, c) = cast<uint8_t>(0);
    if (f.outputs() > 1) {
        _realize(t, x, y, 2) = cast<uint8_t>(clamp(f(x, y, t)[0], 0.0f, 1.0f) * 255.0f);
        _realize(t, x, y, 1) = cast<uint8_t>(clamp(f(x, y, t)[1], 0.0f, 1.0f) * 255.0f);
        _realize(t, x, y, 0) = cast<uint8_t>(clamp(f(x, y, t)[2], 0.0f, 1.0f) * 255.0f);
    } else {
        _realize(t, x, y, 2) = cast<uint8_t>(clamp(f(x, y, t), 0.0f, 1.0f) * 255.0f);
        _realize(t, x, y, 1) = cast<uint8_t>(clamp(f(x, y, t), 0.0f, 1.0f) * 255.0f);
        _realize(t, x, y, 0) = cast<uint8_t>(clamp(f(x, y, t), 0.0f, 1.0f) * 255.0f);
    }

    debug_(initial_debug + current_debug++, t, x, y, c) = _realize(t, x, y, c);
}

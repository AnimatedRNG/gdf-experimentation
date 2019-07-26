#pragma once

#include "HalideBuffer.h"


using namespace Halide::Runtime;
using namespace Halide::Tools;

template <typename T>
inline void to_device(Buffer<T>& buf, const halide_device_interface_t* interface) {
    buf.set_host_dirty(true);
    buf.set_device_dirty(false);
    buf.copy_to_device(interface);
}

template <typename T>
inline void to_host(Buffer<T>& buf) {
    buf.set_host_dirty(false);
    buf.set_device_dirty(true);
    buf.copy_to_host();
}

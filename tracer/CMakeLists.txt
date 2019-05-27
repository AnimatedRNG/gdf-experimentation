cmake_minimum_required(VERSION 3.14)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/modules/")

include("${HALIDE_DISTRIB_DIR}/halide.cmake")

find_package(Threads)

add_executable(tracer src/tracer.cpp)
set_target_properties(tracer PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_include_directories(tracer PRIVATE "${HALIDE_INCLUDE_DIR}" "${HALIDE_TOOLS_DIR}")
halide_library(tracer_render SRCS src/tracer_generator.cpp)
target_link_libraries(tracer PUBLIC ${CMAKE_THREAD_LIBS_INIT} tracer_render)

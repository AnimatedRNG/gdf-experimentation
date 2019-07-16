cmake_minimum_required(VERSION 3.14)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/modules/")

include("${HALIDE_DISTRIB_DIR}/halide.cmake")

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")

find_package(Threads REQUIRED)

add_executable(tracer src/tracer.cpp)
set_target_properties(tracer PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)

halide_use_image_io(tracer)

set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
include_directories(${INCLUDE_DIR})

target_include_directories(tracer PRIVATE "${HALIDE_INCLUDE_DIR}" "${HALIDE_TOOLS_DIR}")

halide_generator(sobel.generator SRCS src/sobel_generator.cpp)
halide_library_from_generator(sobel GENERATOR sobel.generator
                                    GENERATOR_ARGS auto_schedule=true
                                    EXTRA_OUTPUTS schedule
)

halide_generator(projection.generator SRCS src/projection_generator.cpp)
halide_library_from_generator(projection
                                    GENERATOR projection.generator
                                    GENERATOR_ARGS auto_schedule=true
                                    EXTRA_OUTPUTS schedule
)

halide_generator(sdf_gen.generator SRCS src/sdf_generator.cpp)
halide_library_from_generator(sdf_gen
                              GENERATOR sdf_gen.generator
                              GENERATOR_ARGS auto_schedule=true
                              EXTRA_OUTPUTS schedule
)

halide_generator(tracer_render.generator
                 SRCS src/tracer_generator.cpp
                 DEPS sobel.generator projection.generator
)
halide_library_from_generator(tracer_render
                              GENERATOR tracer_render.generator
                              GENERATOR_DEPS sobel.generator projection.generator
                              GENERATOR_ARGS auto_schedule=true
                              EXTRA_OUTPUTS schedule
)

target_link_libraries(tracer PUBLIC
                             ${CMAKE_THREAD_LIBS_INIT}
                             sdf_gen
                             tracer_render)

cmake_minimum_required(VERSION 3.14)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/modules/")

include("${HALIDE_DISTRIB_DIR}/halide.cmake")

find_package(Threads REQUIRED)

add_executable(tracer src/tracer.cpp)
set_target_properties(tracer PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)

set(INCLUDE_PLOTS true)
if (INCLUDE_PLOTS)
   # this is bad
   set(PLOTTING_INCLUDE_DIRS "/usr/include/python2.7"
                             "/usr/lib/python2.7/site-packages/numpy/core/include")
   set(PLOTTING_LIBRARIES "-lpython2.7")
endif(INCLUDE_PLOTS)

halide_use_image_io(tracer)
target_include_directories(tracer PRIVATE "${HALIDE_INCLUDE_DIR}" "${HALIDE_TOOLS_DIR}" ${PLOTTING_INCLUDE_DIR} ${PLOTTING_INCLUDE_DIRS} "${CMAKE_SOURCE_DIR}/include")
halide_library(tracer_render SRCS src/tracer_generator.cpp)
target_link_libraries(tracer PUBLIC ${CMAKE_THREAD_LIBS_INIT} ${PLOTTING_LIBRARIES} tracer_render)

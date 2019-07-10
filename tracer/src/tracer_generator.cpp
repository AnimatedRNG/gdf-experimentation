#include <iostream>
#include <tuple>

#include "Halide.h"
#include "gif.h"

#include "matmul.hpp"
#include "grid_sdf.hpp"
#include "sobel.stub.h"
#include "projection.stub.h"

#include <stdio.h>

using namespace Halide;

constexpr static int iterations = 300;

Var x("x"), y("y"), c("c"), t("t");
Var num_img("num_img");
RDom tr;

// effectively converts a GridSDF into a regular SDF
template <unsigned int N>
TupleVec<N> trilinear(const GridSDF& sdf, TupleVec<3> position) {
    TupleVec<3> grid_space = ((position - sdf.p0) / (sdf.p1 - sdf.p0)) *
                             (cast<float>(sdf.n));

    // floor and ceil slow?
    TupleVec<3> lp = build<3>([grid_space, sdf](unsigned int i) {
        return clamp(cast<int32_t>(grid_space[i]), 0, sdf.n[i] - 1);
    });

    TupleVec<3> up = build<3>([grid_space, sdf](unsigned int i) {
        return clamp(cast<int32_t>(Halide::ceil(grid_space[i])), 0, sdf.n[i] - 1);
    });

    // why won't this work?
    /*Tuple up = {
        clamp(lp[0] + 1, 0, sdf.nx - 1),
        clamp(lp[1] + 1, 0, sdf.ny - 1),
        clamp(lp[2] + 1, 0, sdf.nz - 1)
        };*/

    /*TupleVec<3> alpha = {
        grid_space[0] - lp[0],
        grid_space[1] - lp[1],
        grid_space[2] - lp[2],
        };*/
    TupleVec<3> alpha = grid_space - lp;

    if (N == 1) {
        Expr c000 = sdf.buffer(lp[0], lp[1], lp[2]);
        Expr c001 = sdf.buffer(lp[0], lp[1], up[2]);
        Expr c010 = sdf.buffer(lp[0], up[1], lp[2]);
        Expr c011 = sdf.buffer(lp[0], up[1], up[2]);
        Expr c100 = sdf.buffer(up[0], lp[1], lp[2]);
        Expr c101 = sdf.buffer(up[0], lp[1], up[2]);
        Expr c110 = sdf.buffer(up[0], up[1], lp[2]);
        Expr c111 = sdf.buffer(up[0], up[1], up[2]);

        // interpolate on x
        Expr c00 = Halide::lerp(c000, c100, alpha[0]);
        Expr c01 = Halide::lerp(c001, c101, alpha[0]);
        Expr c10 = Halide::lerp(c010, c110, alpha[0]);
        Expr c11 = Halide::lerp(c011, c111, alpha[0]);

        // interpolate on y
        Expr c0 = Halide::lerp(c00, c10, alpha[1]);
        Expr c1 = Halide::lerp(c01, c11, alpha[1]);

        // interpolate on z
        Expr c = Halide::lerp(c0, c1, alpha[2]);

        return TupleVec<N>({c});
    } else {
        Tuple c000 = sdf.buffer(lp[0], lp[1], lp[2]);
        Tuple c001 = sdf.buffer(lp[0], lp[1], up[2]);
        Tuple c010 = sdf.buffer(lp[0], up[1], lp[2]);
        Tuple c011 = sdf.buffer(lp[0], up[1], up[2]);
        Tuple c100 = sdf.buffer(up[0], lp[1], lp[2]);
        Tuple c101 = sdf.buffer(up[0], lp[1], up[2]);
        Tuple c110 = sdf.buffer(up[0], up[1], lp[2]);
        Tuple c111 = sdf.buffer(up[0], up[1], up[2]);

        TupleVec<N> c00 = TupleVec<N>(c000) * (1.0f - alpha[0]) +
                          TupleVec<N>(c100) * alpha[0];
        TupleVec<N> c01 = TupleVec<N>(c001) * (1.0f - alpha[0]) +
                          TupleVec<N>(c101) * alpha[0];
        TupleVec<N> c10 = TupleVec<N>(c010) * (1.0f - alpha[0]) +
                          TupleVec<N>(c110) * alpha[0];
        TupleVec<N> c11 = TupleVec<N>(c011) * (1.0f - alpha[0]) +
                          TupleVec<N>(c111) * alpha[0];

        TupleVec<N> c0 = TupleVec<N>(c00) * (1.0f - alpha[1]) +
                         TupleVec<N>(c10) * alpha[1];
        TupleVec<N> c1 = TupleVec<N>(c01) * (1.0f - alpha[1]) +
                         TupleVec<N>(c11) * alpha[1];

        TupleVec<N> c = TupleVec<N>(c0) * (1.0f - alpha[2]) +
                        TupleVec<N>(c1) * alpha[2];

        return c;
    }
}



Expr example_sphere(TupleVec<3> position) {
    return norm(position) - 3.0f;
}

Expr vmax(TupleVec<3> v) {
    return Halide::max(Halide::max(v[0], v[1]), v[2]);
}

Expr example_box(TupleVec<3> position) {
    TupleVec<3> b = {3.0f, 3.0f, 3.0f};
    TupleVec<3> d = abs(position + Tuple(-0.22f, 0.19f, 0.34f)) - b;

    return norm(max(d, Expr(0.0f))) + vmax(min(d, Expr(0.0f)));
}

Expr to_render_dist(Expr dist) {
    return 1.0f / \
        (10.0f + (1.0f - Halide::clamp(Halide::abs(dist), 0.0f, 1.0f)) * 90.0f);
}

class TracerGenerator : public Halide::Generator<TracerGenerator> {
  public:

    Input<Buffer<float>> projection_{"projection", 2};
    Input<Buffer<float>> view_{"view", 2};
    Input<int32_t> width{"width"};
    Input<int32_t> height{"height"};
    Output<Buffer<float>> out_{"out", 3};
    Output<Buffer<uint8_t>> debug_{"debug", 5};
    Output<Buffer<int32_t>> num_debug{"num_debug"};

    int current_debug = 0;

    inline void record(Func f,
                       int32_t iterations = 300,
                       float delay = 0.03f) {
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

        debug_(current_debug++, t, x, y, c) = _realize(t, x, y, c);
    }

    Expr normal_pdf(Expr x, float sigma = 1e-7f, float mean = 0.0f) {
        return (1.0f / Halide::sqrt(2.0f * (float) M_PI * sigma * sigma)) *
               Halide::exp((x - mean) * (x - mean) / (-2.0f * sigma * sigma));
    }

    Expr relu(Expr a) {
        return Halide::max(a, 0);
    }

    Expr normal_pdf_rectified(Expr x, float sigma = 1e-2f, float mean = 0.0f) {
        return normal_pdf(relu(x), sigma, mean);
    }

    Func light_source(TupleVec<3> light_color,
                      Func positions,
                      TupleVec<3> light_position,
                      GridSDF normals,
                      float kd = 0.7f,
                      float ks = 0.3f,
                      float ka = 100.0f) {
        Func light_vec("light_vec");
        light_vec(x, y, t) = (light_position - Tuple(positions(x, y, t))).get();

        Func light_vec_norm("light_vec_norm");
        light_vec_norm(x, y, t) = norm(TupleVec<3>(light_vec(x, y, t)));

        Func light_vec_normalized("light_vec_normalized");
        light_vec_normalized(x, y, t) =
            (TupleVec<3>(light_vec(x, y, t))
             / Expr(light_vec_norm(x, y, t))).get();

        TupleVec<3> normal_sample =
            trilinear<3>(normals, TupleVec<3>(Tuple(positions(x, y, t))));
        Func diffuse("diffuse");
        diffuse(x, y, t) = (kd * clamp(dot(normal_sample,
                                           Tuple(light_vec_normalized(x, y, t))),
                                       0.0f, 1.0f) * light_color).get();
        //diffuse(x, y, tr) = ((kd * Expr(1.0f)) * light_color).get();

        //light_vec.compute_at(diffuse, x);
        //light_vec_norm.compute_at(light_vec_normalized, x);

        //light_vec_normalized.compute_at(diffuse, x);

        return diffuse;
    }

    Func shade(Func positions, TupleVec<3> origin, GridSDF normals) {
        TupleVec<3> top_light_color = {0.6f, 0.6f, 0.0f};
        TupleVec<3> self_light_color = {0.4f, 0.0f, 0.4f};

        TupleVec<3> top_light_pos = {10.0f, 30.0f, 0.0f};

        TupleVec<3> self_light_pos = origin;

        Func top_light("top_light"), self_light("self_light");
        top_light = light_source(top_light_color,
                                 positions,
                                 top_light_pos,
                                 normals);
        self_light = light_source(self_light_color,
                                  positions,
                                  self_light_pos,
                                  normals);

        Func total_light("total_light");
        total_light(x, y, t) = (TupleVec<3>(top_light(x, y, t))
                                + TupleVec<3>(self_light(x, y, t))).get();
        //total_light.trace_stores();

        //top_light.compute_at(total_light, x);
        //self_light.compute_at(total_light, x);
        //total_light.compute_root();

        //top_light.trace_stores();
        //top_light.trace_loads();

        //total_light.trace_stores();
        //total_light.trace_loads();

        return total_light;
    }

    TupleVec<3> step_back(TupleVec<3> positions, TupleVec<3> ray_vec,
                          float EPS = 1e-2f) {
        return (-1.0f * Expr(EPS)) * ray_vec + positions;
    }

    GridSDF call_sobel(GridSDF sdf) {
        Func sb = sobel::generate(Halide::GeneratorContext(this->get_target(),
                                  auto_schedule),
        {sdf.buffer, sdf.nx, sdf.ny, sdf.nz});
        //sb.compute_root();
        //sb.trace_loads();

        return GridSDF(sb, sdf.p0, sdf.p1, sdf.nx, sdf.ny, sdf.nz);
    }

    Func sphere_trace(const GridSDF& sdf,
                      float EPS = 1e-6f) {
        Func original_ray_pos("original_ray_pos");
        Func ray_vec("ray_vec");
        Func origin("origin");

        /*std::forward_as_tuple(std::tie(original_ray_pos,
                                       ray_vec), origin) =
                                           projection(projection_, view_);*/
        auto outputs = projection::generate(Halide::GeneratorContext(this->get_target(),
                                            auto_schedule),
        {projection_, view_, width, height});
        original_ray_pos = outputs.ray_pos;
        ray_vec = outputs.ray_vec;
        origin = outputs.origin;

        //original_ray_pos.compute_root();
        //ray_vec.compute_root();
        //origin.compute_root();

        GridSDF sb = call_sobel(sdf);

        Func pos("pos");
        Expr d("d");
        Func depth("depth");

        pos(x, y, t) = Tuple(0.0f, 0.0f, 0.0f);
        pos(x, y, 0) = original_ray_pos(x, y);
        d = trilinear<1>(sdf, TupleVec<3>(Tuple(pos(x, y, tr))))[0];
        d = to_render_dist(d);
        pos(x, y, tr + 1) = (TupleVec<3>(pos(x, y, tr)) +
                             d * TupleVec<3>(ray_vec(x, y))).get();
        Var xi, xo, yi, yo;
        depth(x, y, t) = 0.0f;
        depth(x, y, tr + 1) = d;
        //depth.trace_stores();

        Func normal_evaluation_position("normal_evaluation_position");
        normal_evaluation_position(x, y, t) = step_back(
                TupleVec<3>(pos(x, y, t)), TupleVec<3>(ray_vec(x, y))).get();

        Func shaded("shaded");
        shaded(x, y, t) = shade(normal_evaluation_position, {origin(0), origin(1), origin(2)},
                                sb)(x, y, t);

        Func normals_now("normals_now");
        normals_now(x, y, t) = (trilinear<3>(
                                    sb,
                                    TupleVec<3>(
                                        normal_evaluation_position(x, y, t)))).get();

        record(pos);
        record(shaded);
        record(depth);
        record(normals_now);

        //normal_evaluation_position.compute_at(shaded, x);
        //shaded.compute_root();

        Func endpoint("endpoint");
        //endpoint(x, y) = pos(x, y, iterations - 1);
        /*endpoint(x, y) = {depth(x, y, iterations),
                          depth(x, y, iterations),
                          depth(x, y, iterations)
                          };*/
        endpoint(x, y) = shaded(x, y, iterations - 1);
        //endpoint(x, y) = normals_now(x, y, iterations - 1);
        //shaded.trace_stores();
        //endpoint.compute_root();
        //apply_auto_schedule(endpoint);

        //pos.unroll(t);

        //pos.compute_at(endpoint, x);
        //pos.store_at(endpoint, x);
        //apply_auto_schedule(shaded);
        //apply_auto_schedule(pos);

        return endpoint;
    }

    void generate() {
        debug_(num_img, t, x, y, c) = cast<uint8_t>(0);
        tr = RDom(0, iterations);

        GridSDF grid_sdf = to_grid_sdf(example_box,
        {-4.0f, -4.0f, -4.0f},
        {4.0f, 4.0f, 4.0f}, 128, 128, 128);

        Func end("end");
        end(x, y) = sphere_trace(grid_sdf)(x, y);

        //std::cout << Buffer<float>(sobel(grid_sdf).realize(32, 32, 32)[0])(0, 0, 0)
        //          << std::endl;

        // flip image and RGB -> BGR to match OpenCV's output
        out_(x, y, c) = 0.0f;
        out_(x, y, 0) = clamp(end(y, x)[2], 0.0f, 1.0f);
        out_(x, y, 1) = clamp(end(y, x)[1], 0.0f, 1.0f);
        out_(x, y, 2) = clamp(end(y, x)[0], 0.0f, 1.0f);

        num_debug(x) = Func(Expr(current_debug))();

        if (auto_schedule) {
            projection_.dim(0).set_bounds_estimate(0, 4)
            .dim(1).set_bounds_estimate(0, 4);
            view_.dim(0).set_bounds_estimate(0, 4)
            .dim(1).set_bounds_estimate(0, 4);

            out_.dim(0).set_bounds_estimate(0, 1920)
            .dim(1).set_bounds_estimate(0, 1920)
            .dim(2).set_bounds_estimate(0, 3);

            debug_
            .estimate(num_img, 0, current_debug)
            .estimate(t, 0, 300)
            .estimate(x, 0, 1920)
            .estimate(y, 0, 1080)
            .estimate(c, 0, 3);
            num_debug.estimate(x, 0, 1);
        }
    }
};

HALIDE_REGISTER_GENERATOR(TracerGenerator, tracer_render)

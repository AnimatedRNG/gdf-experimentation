#include <iostream>
#include <tuple>
#include <string>

#include "Halide.h"
#include "gif.h"

#include "matmul.hpp"
#include "recorder.hpp"
#include "debug.hpp"
#include "grid_sdf.hpp"
#include "sobel.stub.h"
#include "projection.stub.h"

#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <stdio.h>

using namespace Halide;

constexpr static int iterations = 900;

Var x("x"), y("y"), c("c"), t("t");
Var i("i");
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

Expr to_render_dist(Expr dist, Expr scale_factor = Expr(1.0f)) {
    return scale_factor /                                             \
           (10.0f + (1.0f - Halide::clamp(Halide::abs(dist), 0.0f, 1.0f)) * 90.0f);
    //return scale_factor / 10.0f;
}

class TracerGenerator : public Halide::Generator<TracerGenerator> {
  public:

    Input<Buffer<float>> projection_{"projection", 2};
    Input<Buffer<float>> view_{"view", 2};

    Input<Buffer<float>> sdf_{"sdf_", 3};
    Input<Buffer<float>> p0_{"p0", 1};
    Input<Buffer<float>> p1_{"p1", 1};

    Input<bool> backwards_{"backwards"};

    Input<int32_t> width{"width"};
    Input<int32_t> height{"height"};
    Input<int32_t> initial_debug{"initial_debug"};

    Output<Func> out_ {"out", Float(32), 3};
#ifdef DEBUG_TRACER
    Output<Func> debug_ {"debug", UInt(8), 5};
    Output<Func> num_debug{"num_debug", Int(32), 1};
#endif // DEBUG_TRACER

    std::shared_ptr<GridSDF> sb;

    Func original_ray_pos{"original_ray_pos"};
    Func ray_vec{"ray_vec"};
    Func origin{"origin"};

    Func pos{"pos"};
    Func dist{"dist"};
    Func dist_render{"dist_render"};

    Func normal_evaluation_position{"normal_evaluation_position"};
    Func g_d{"g_d"};

    Func intensity{"intensity"};

    Func opc{"opc"};

    Func volumetric_shaded{"volumetric_shaded"};
    Func normals_debug{"normals_debug"};
    Func forward{"forward"};

    Func middle_func{"middle_func"};

    Func dLoss_dSDF{"dLoss_dSDF"};
    Derivative dr;

    std::map<std::string, std::vector<Func>> _tuplevec_unpacked;

    int current_debug = 0;

    // Unpacks a tuple valued function into a new channel,
    // stores it in _tuplevec_unpacked, packs it up again
    // and returns it.
    //
    // A hack to get gradients for tuple-valued functions
    template <unsigned int N>
    Func repack(Func other) {
        Var C("C");
        std::vector<Expr> args; //
        args.clear();
        for (auto arg : other.args()) {
            args.push_back(Expr(arg));
        }

        std::vector<Func> packed;
        packed.clear();
        for (int i = 0; i < N; i++) {
            Func packed_i(other.name() + "_packed_" + std::to_string(i));
            packed_i(args) = other(args)[i];
            packed.push_back(packed_i);
        }

        Func unpacked(other.name() + "_unpacked");
        std::vector<Expr> unpacked_tuple;
        unpacked_tuple.clear();
        for (int i = 0; i < N; i++) {
            unpacked_tuple.push_back(packed[i](args));
        }

        unpacked(args) = Tuple(unpacked_tuple);

        _tuplevec_unpacked[other.name()] = std::move(packed);

        return unpacked;
    }

    Func& get_packed(const std::string& name, const int& index) {
        return _tuplevec_unpacked.at(name).at(index);
    }

    void record(Func f) {
#ifdef DEBUG_TRACER
        _record(f, debug_, num_debug, initial_debug, current_debug);
#endif //DEBUG_TRACER
    }

    void record(std::vector<Internal::Function>& funcs) {
        Func mix(funcs.at(0).name());

        Var C("C");
        std::vector<Expr> args;
        args.clear();
        for (auto arg : funcs.at(0).args()) {
            args.push_back(Expr(arg));
        }

        std::vector<Expr> pure(args);
        pure.push_back(C);

        mix(pure) = 0.0f;
        for (auto func_it = funcs.begin();
                func_it != funcs.end();
                func_it++) {
            std::vector<Expr> impure(args);
            impure.push_back(Expr((int32_t)(func_it - funcs.begin())));
            mix(impure) = Func(*func_it)(args);
        }

        record(mix);
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

    GridSDF create_grid_sdf() {
        return GridSDF(Func(sdf_),
                       TupleVec<3>({p0_(0), p0_(1), p0_(2)}),
                       TupleVec<3>({p1_(0), p1_(1), p1_(2)}), TupleVec<3>(Tuple(
                                   sdf_.dim(0).extent(),
                                   sdf_.dim(1).extent(),
                                   sdf_.dim(2).extent()
                               )));
    }

    void call_sobel(GridSDF sdf) {
        Func sb_func = sobel::generate(Halide::GeneratorContext(this->get_target(),
                                       true),
        {sdf.buffer, sdf.n[0], sdf.n[1], sdf.n[2]});

        this->sb = std::shared_ptr<GridSDF>(new GridSDF(sb_func, sdf.p0, sdf.p1,
                                            sdf.n));
    }

    Func forward_pass(const GridSDF& sdf,
                      float EPS = 1e-6f) {
        auto outputs = projection::generate(Halide::GeneratorContext(this->get_target(),
                                            true),
        {projection_, view_, width, height});
        original_ray_pos = outputs.ray_pos;
        ray_vec = outputs.ray_vec;
        origin = outputs.origin;

        call_sobel(sdf);

        Expr d("d");
        Expr ds("ds");
        Expr step("step");
        Func repacked_ray_vec = repack<3>(ray_vec);

        pos(x, y, t) = repack<3>(original_ray_pos)(x, y);
        d = trilinear<1>(sdf, TupleVec<3>(Tuple(pos(x, y, tr))))[0];
        //d = example_box(TupleVec<3>(pos(x, y, tr)));
        ds = to_render_dist(d);
        step = 1.0f / 100.0f;
        pos(x, y, tr + 1) = (TupleVec<3>(pos(x, y, tr)) +
                             step * TupleVec<3>(repacked_ray_vec(x, y))).get();

        Var xi, xo, yi, yo;
        dist(x, y, t) = 0.0f;
        dist(x, y, tr + 1) = d;

        dist_render(x, y, t) = 0.0f;
        dist_render(x, y, tr + 1) = ds;

        normal_evaluation_position(x, y, t) = step_back(
                TupleVec<3>(pos(x, y, t)), TupleVec<3>(ray_vec(x, y))).get();

        g_d(x, y, t) = normal_pdf_rectified(dist(x, y, t));

        intensity(x, y, t) = shade(normal_evaluation_position, {origin(0), origin(1), origin(2)},
                                   *sb)(x, y, t);

        opc(x, y, t) = 0.0f;
        opc(x, y, tr + 1) = opc(x, y, tr) + g_d(x, y, tr) * dist_render(x, y, tr);

        float u_s = 1.0f;
        float k = -1.0f;

        Expr scattering("scattering");
        scattering = g_d(x, y, tr) * u_s;

        volumetric_shaded(x, y, t) = {0.0f, 0.0f, 0.0f};
        volumetric_shaded(x, y, tr + 1) =
            (TupleVec<3>(volumetric_shaded(x, y, tr)) + (scattering * Halide::exp(k * opc(x,
                    y, tr))) *
             TupleVec<3>(intensity(x, y, tr)) * Expr(dist_render(x, y, tr))).get();

        normals_debug(x, y, t) = (trilinear<3>(
                                      *sb,
                                      TupleVec<3>(
                                          normal_evaluation_position(x, y, t)))).get();

        forward(x, y) = volumetric_shaded(x, y, iterations);
        //forward(x, y) = repack<3>(pos)(x, y, iterations);
        //forward(x, y) = repack<3>(pos)(x, y, iterations);

        /*record(pos);
        record(intensity);
        record(dist);
        record(normals_debug);
        record(g_d);
        record(opc);
        record(volumetric_shaded);*/

        return forward;
    }

    Func backwards_pass(GridSDF sdf, Func forward, float EPS = 1e-6f) {
        Func loss_xyc("loss_xyc");
        Func loss_xy("loss_xy");
        Func loss_y("loss_y");
        Func loss("loss");
        RDom rx(0, width);
        RDom ry(0, height);
        loss_xyc(x, y) = (1.0f - TupleVec<3>(forward(x, y))).get();
        loss_xy(x, y) = norm(repack<3>(loss_xyc)(x, y));
        loss_y(y) = 0.0f;
        loss_y(y) += loss_xy(rx, y);
        loss() = 0.0f;
        loss() += loss_y(ry);

        /*Func a("a"), b("b"), d("d");
        RDom z(0, 3);
        d() = 5.0f;
        a(x) = d();
        a(z + 1) = a(z) * 2.0f;
        // b() = a(3) = a(2) * 5 = (a(1) * 5) * 5 = ((a(0) * 5) * 5) * 5 = 375
        // db/da(0) = ((1 * 5) * 5) * 5 = 125
        b() = a(3);

        Func dbdd = propagate_adjoints(b)(d);

        Buffer<float> test = b.realize();
        std::cout << "test is " << test() << std::endl;
        Buffer<float> deriv = dbdd.realize();
        std::cout << "dbda is " << deriv() << std::endl;*/

        //Buffer<float> test = da_dd.realize();
        //std::cout << "test is " << test() << std::endl;

        dr = propagate_adjoints(loss);
        dLoss_dSDF = dr(sdf_);

        Func backwards("backwards");
        //backwards(x, y) = {dLoss_dSDF(x, y, 50), 0.0f, 0.0f};
        Func de("de");
        /*de(x, y, c) = {
            select(Halide::abs(dLoss_dSDF(x, y, c)) >= 1e-9f, 1.0f, 0.0f),
            select(Halide::abs(dLoss_dSDF(x, y, c)) >= 1e-9f, 1.0f, 0.0f),
            select(Halide::abs(dLoss_dSDF(x, y, c)) >= 1e-9f, 1.0f, 0.0f)
            };*/
        de(x, y, c) = {
            Halide::log(abs(dLoss_dSDF(x, y, c))) / -1e2f,
            Halide::log(abs(dLoss_dSDF(x, y, c))) / -1e2f,
            Halide::log(abs(dLoss_dSDF(x, y, c))) / -1e2f,
        };
        /*de(x, y, c) = {select(dr(get_packed("ray_pos$1", 0))(x, y) > 1e-2f, 1.0f, 0.0f),
                       select(dr(get_packed("ray_pos$1", 1))(x, y) > 1e-2f, 1.0f, 0.0f),
                       select(dr(get_packed("ray_pos$1", 2))(x, y) > 1e-2f, 1.0f, 0.0f)
                       };*/
        /*de(x, y, c) = {dr(get_packed("ray_vec$1", 0))(x, y),
                       dr(get_packed("ray_vec$1", 1))(x, y),
                       dr(get_packed("ray_vec$1", 2))(x, y)};*/
        backwards(x, y) = de(x, y, 50);

        Func debug_gradient("debug_gradient");
        debug_gradient(x, y, c) = de(x, y, cast<int32_t>((c / 900.0f) * 128.0f));
        record(debug_gradient);
        //print_func(debug_gradient);

        return backwards;
    }

    void generate() {
#ifdef DEBUG_TRACER
        debug_(i, t, x, y, c) = cast<uint8_t>(0);
#endif //DEBUG_TRACER

        tr = RDom(0, iterations);

        /*GridSDF grid_sdf = to_grid_sdf(example_box,
        {-4.0f, -4.0f, -4.0f},
        {4.0f, 4.0f, 4.0f}, 128, 128, 128);*/
        GridSDF grid_sdf = create_grid_sdf();

        Func end("end");
        /*end(x, y) =
            select(backwards_, backwards_pass(grid_sdf, forward_pass(grid_sdf))(x, y),
            forward_pass(grid_sdf)(x, y))*/
        Func fw_pass = forward_pass(grid_sdf);

        //end(x, y) = fw_pass(x, y);
        end(x, y) = backwards_pass(grid_sdf, fw_pass)(x, y);

        // flip image and RGB -> BGR to match OpenCV's output
        /*out_(x, y, c) = 0.0f;
        out_(x, y, 0) = clamp(end(y, x)[2], 0.0f, 1.0f);
        out_(x, y, 1) = clamp(end(y, x)[1], 0.0f, 1.0f);
        out_(x, y, 2) = clamp(end(y, x)[0], 0.0f, 1.0f);*/
        out_(x, y, c) = 0.0f;
        out_(x, y, 0) = end(y, x)[0];
        out_(x, y, 1) = end(y, x)[1];
        out_(x, y, 2) = end(y, x)[2];

#ifdef DEBUG_TRACER
        num_debug(x) = Func(Expr(current_debug) + initial_debug)();
#endif // DEBUG_TRACER

        if (/*auto_schedule*/ true) {
            sdf_.dim(0).set_bounds_estimate(0, 32)
            .dim(1).set_bounds_estimate(0, 32)
            .dim(2).set_bounds_estimate(0, 32);

            projection_.dim(0).set_bounds_estimate(0, 4)
            .dim(1).set_bounds_estimate(0, 4);
            view_.dim(0).set_bounds_estimate(0, 4)
            .dim(1).set_bounds_estimate(0, 4);

            p0_.dim(0).set_bounds_estimate(0, 3);
            p1_.dim(0).set_bounds_estimate(0, 3);

            out_.estimate(x, 0, 1920)
            .estimate(y, 0, 1080)
            .estimate(c, 0, 3);

#ifdef DEBUG_TRACER
            debug_
            .estimate(i, 0, current_debug)
            .estimate(t, 0, 300)
            .estimate(x, 0, 1920)
            .estimate(y, 0, 1080)
            .estimate(c, 0, 3);

            num_debug.estimate(x, 0, 1);
#endif //DEBUG_TRACER

            //Pipeline p(std::vector<Func>({out_, debug_, num_debug}));
            //p.auto_schedule(this->get_target());

            Halide::SimpleAutoscheduleOptions options;
            options.gpu = get_target().has_gpu_feature();
            options.gpu_tile_channel = 1;

#ifdef DEBUG_TRACER
            std::vector<Func> output_func({out_, debug_, num_debug});
#else
            std::vector<Func> output_func({out_});
#endif // DEBUG_TRACER
            Halide::simple_autoschedule(
            output_func, {
                {"sdf_.min.0", 0},
                {"sdf_.extent.0", 128},
                {"sdf_.min.1", 0},
                {"sdf_.extent.1", 128},
                {"sdf_.min.2", 0},
                {"sdf_.extent.2", 128},
                {"projection_.min.0", 0},
                {"projection_.extent.0", 4},
                {"projection_.min.1", 0},
                {"projection_.extent.1", 4},
                {"view_.min.0", 0},
                {"view_.extent.0", 4},
                {"view_.min.1", 0},
                {"view_.extent.1", 4},
                {"p0_.min.0", 0},
                {"p0_.extent.0", 4},
                {"p1_.min.0", 0},
                {"p1_.extent.0", 4},
                {"width", 1920},
                {"height", 1080},
            }, {
                {
                    {0, 1920},
                    {0, 1080},
                    {0, 3}
                },
#ifdef DEBUG_TRACER
                {
                    {0, current_debug},
                    {0, 300},
                    {0, 1920},
                    {0, 1080},
                    {0, 3}
                },
                {
                    {0, 1}
                }
#endif //DEBUG_TRACER
            },
            options);

            //apply_auto_schedule(sb->buffer);
            //apply_auto_schedule(projection_);
            //apply_auto_schedule(view_);
        }
    }
};

HALIDE_REGISTER_GENERATOR(TracerGenerator, tracer_render)

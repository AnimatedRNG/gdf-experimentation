#include <iostream>
#include <tuple>
#include <string>

#include "Halide.h"
#include "gif.h"

#include "matmul.hpp"
#include "recorder.hpp"
#include "debug.hpp"
#include "utils.hpp"
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
Var xo("xo"), yo("yo"), xi("xi"), yi("yi");
Var to("to"), ti("to");
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
    Input<Buffer<float>> translation_{"model", 1};

    Input<Buffer<float>> sdf_{"sdf_", 3};
    Input<Buffer<float>> p0_{"p0", 1};
    Input<Buffer<float>> p1_{"p1", 1};

    Input<Buffer<float>> target_{"target_", 3};

    Input<int32_t> width{"width"};
    Input<int32_t> height{"height"};
    Input<int32_t> initial_debug{"initial_debug"};

    Output<Buffer<float>> forward_{"forward_", 3};
    Output<Buffer<float>> d_l_sdf_{"d_l_sdf_", 3};
    Output<Buffer<float>> d_l_translation_{"d_l_translation_", 1};
#ifdef DEBUG_TRACER
    Output<Func> debug_ {"debug", UInt(8), 5};
    Output<Func> num_debug{"num_debug", Int(32), 1};
#endif // DEBUG_TRACER

    std::unordered_map<std::string, std::shared_ptr<GridSDF>> sb;

    Func middle_func{"middle_func"};

    Func model_transform{"identity"};
    Func dLoss_dSDF{"dLoss_dSDF"};

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

    void record(Func f, bool wrap_c = false) {
#ifdef DEBUG_TRACER
        if (wrap_c) {
            Func new_f;
            auto args = f.args();
            std::vector<Var> new_ch(args);
            new_ch.push_back(Var("c"));
            new_f(new_ch) = f(args);
            _record(new_f, debug_, num_debug, initial_debug, current_debug);
        } else {
            _record(f, debug_, num_debug, initial_debug, current_debug);
        }
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

    void call_sobel(const std::string& name, GridSDF sdf) {
        Func sb_func = sobel::generate(Halide::GeneratorContext(this->get_target(),
                                       true),
        {sdf.buffer, sdf.n[0], sdf.n[1], sdf.n[2]});

        this->sb[name] = std::shared_ptr<GridSDF>(new GridSDF(sb_func, sdf.p0, sdf.p1,
                         sdf.n));
    }

    parameter_map forward_pass(
        std::string name,
        const GridSDF& sdf,
        Func ray_transform,
        float EPS = 1e-6f) {

        Func original_ray_pos("original_ray_pos_" + name);
        Func ray_vec("ray_vec_" + name);
        Func repacked_ray_vec("repacked_ray_vec_" + name);
        Func origin("origin_" + name);

        Func pos("pos_" + name);
        Func dist("dist_" + name);
        Func dist_render("dist_render_" + name);

        Func normal_evaluation_position("normal_evaluation_position_" + name);
        Func g_d("g_d_" + name);

        Func intensity("intensity_" + name);

        Func opc("opc_" + name);

        Func volumetric_shaded("volumetric_shaded_" + name);
        Func normals_debug("normals_debug_" + name);
        Func forward("forward_" + name);

        auto outputs = projection::generate(Halide::GeneratorContext(this->get_target(),
                                            true),
        {projection_, view_, width, height});
        original_ray_pos = outputs.ray_pos;
        ray_vec = outputs.ray_vec;
        origin = outputs.origin;

        call_sobel(name, sdf);

        Expr d("d_" + name);
        Expr ds("ds_" + name);
        Expr step("step_" + name);
        repacked_ray_vec = repack<3>(ray_vec);

        Func transformed_ray_pos("transformed_ray_pos_" + name);
        transformed_ray_pos(x, y) = {
            original_ray_pos(x, y)[0] + ray_transform(0),
            original_ray_pos(x, y)[1] + ray_transform(1),
            original_ray_pos(x, y)[2] + ray_transform(2)
        };

        pos(x, y, t) = repack<3>(transformed_ray_pos)(x, y);
        d = trilinear<1>(sdf, TupleVec<3>(Tuple(pos(x, y, tr))))[0];
        //d = example_box(TupleVec<3>(pos(x, y, tr)));
        ds = to_render_dist(d);
        step = 1.0f / 100.0f;
        pos(x, y, tr + 1) = (TupleVec<3>(pos(x, y, tr)) +
                             step * TupleVec<3>(repacked_ray_vec(x, y))).get();

        dist(x, y, t) = 0.0f;
        dist(x, y, tr + 1) = d;

        dist_render(x, y, t) = 0.0f;
        dist_render(x, y, tr + 1) = ds;

        normal_evaluation_position(x, y, t) = step_back(
                TupleVec<3>(pos(x, y, t)), TupleVec<3>(ray_vec(x, y))).get();

        g_d(x, y, t) = normal_pdf_rectified(dist(x, y, t));

        intensity(x, y, t) = shade(normal_evaluation_position, {origin(0), origin(1), origin(2)},
                                   *(sb[name]))(x, y, t);

        opc(x, y, t) = 0.0f;
        opc(x, y, tr + 1) = opc(x, y, tr) + g_d(x, y, tr) * dist_render(x, y, tr);

        float u_s = 1.0f;
        float k = -1.0f;

        Expr scattering("scattering_" + name);
        scattering = g_d(x, y, tr) * u_s;

        volumetric_shaded(x, y, t) = {0.0f, 0.0f, 0.0f};
        volumetric_shaded(x, y, tr + 1) =
            (TupleVec<3>(volumetric_shaded(x, y, tr)) + (scattering * Halide::exp(k * opc(x,
                    y, tr))) *
             TupleVec<3>(intensity(x, y, tr)) * Expr(dist_render(x, y, tr))).get();

        normals_debug(x, y, t) = (trilinear<3>(
                                      *(sb[name]),
                                      TupleVec<3>(
                                          normal_evaluation_position(x, y, t)))).get();

        forward(x, y) = {
            clamp(volumetric_shaded(y, x, iterations)[2], 0.0f, 1.0f),
            clamp(volumetric_shaded(y, x, iterations)[1], 0.0f, 1.0f),
            clamp(volumetric_shaded(y, x, iterations)[0], 0.0f, 1.0f)
        };
        /*forward(x, y) = {
            clamp(pos(x, y, iterations)[0], 0.0f, 1.0f),
            clamp(pos(x, y, iterations)[1], 0.0f, 1.0f),
            clamp(pos(x, y, iterations)[2], 0.0f, 1.0f)
            };*/

        return {
            {"original_ray_pos", original_ray_pos},
            {"ray_vec", ray_vec},
            {"origin", origin},

            {"ray_transform", ray_transform},

            {"pos", pos},
            {"dist", dist},
            {"dist_render", dist_render},

            {"normal_evaluation_position", normal_evaluation_position},
            {"g_d", g_d},

            {"intensity", intensity},

            {"opc", opc},

            {"volumetric_shaded", volumetric_shaded},
            {"normals_debug", normals_debug},
            {"forward", forward},

            {"transformed_ray_pos", transformed_ray_pos}
        };
    }

    parameter_map backwards_pass(
        GridSDF sdf, parameter_map& forward_map, Func target, float EPS = 1e-6f) {
        Func loss_xyc("loss_xyc");
        Func loss_xy("loss_xy");
        Func loss_y("loss_y");
        Func loss("loss");
        RDom rx(0, width);
        RDom ry(0, height);

        Func forward = forward_map.at("forward");
        loss_xyc(x, y) = (1.0f - TupleVec<3>(forward(x, y))).get();
        loss_xy(x, y) = norm(loss_xyc(x, y));
        loss_y(y) = 0.0f;
        loss_y(y) += loss_xy(rx, y);
        loss() = 0.0f;
        loss() += loss_y(ry);

        Derivative dr = propagate_adjoints(loss);

        Func dLoss_dSDF;
        Func dLoss_dTranslation;
        Func dLoss_dRayVec;
        dLoss_dSDF = dr(sdf.buffer);
        dLoss_dTranslation = dr(forward_map.at("ray_transform"));
        dLoss_dRayVec = dr(forward_map.at("ray_vec"));

        Func gradient_cont("gradient_cont");
        Func debug_gradient("debug_gradient");
        gradient_cont(x, y, c) =
            dLoss_dSDF(x, y, cast<int32_t>((c / 900.0f) * 128.0f));
        /*debug_gradient(x, y, c) =
          Halide::log(abs(gradient_cont(x, y, c))) / 1e2f;*/
        debug_gradient(x, y, c) =
            select(debug_gradient(x, y, c) != 0.0f, 1.0f, 0.0f);
        //print_func(debug_gradient);

        return {
            {"loss_xyc", loss_xyc},
            {"loss_xy", loss_xy},
            {"loss_y", loss_y},
            {"loss", loss},
            {"dLoss_dSDF", dLoss_dSDF},
            {"dLoss_dTranslation", dLoss_dTranslation},
            {"dLoss_dRayVec", dLoss_dRayVec},
            {"debug_gradient", debug_gradient}
        };
    }

    void schedule_forward_pass(parameter_map& fw) {
        // projection stuff performance doesn't matter, we just
        // DO NOT want to inline it!
        apply_auto_schedule(fw.at("original_ray_pos"));
        apply_auto_schedule(fw.at("ray_vec"));
        apply_auto_schedule(fw.at("origin"));

        fw.at("original_ray_pos").reorder_storage(y, x);
        fw.at("ray_vec").reorder_storage(y, x);;

        fw.at("ray_transform").compute_root();
        fw.at("transformed_ray_pos").compute_root();

        fw.at("pos").reorder(t, y, x).reorder_storage(t, y, x)
        .bound(t, 0, iterations + 1)
        .compute_at(fw.at("volumetric_shaded"), y);
        fw.at("dist_render").reorder(t, y, x).reorder_storage(t, y, x)
        .bound(t, 0, iterations + 1)
        .compute_at(fw.at("volumetric_shaded"), y);
        fw.at("normal_evaluation_position").reorder(t, y, x).reorder_storage(t, y, x)
        .bound(t, 0, iterations + 1)
        .compute_at(fw.at("volumetric_shaded"), y);
        fw.at("g_d").reorder(t, y, x).reorder_storage(t, y, x)
        .bound(t, 0, iterations + 1)
        .compute_at(fw.at("volumetric_shaded"), y);
        fw.at("intensity").reorder(t, y, x).reorder_storage(t, y, x)
        .bound(t, 0, iterations + 1)
        .compute_at(fw.at("volumetric_shaded"), y);

        fw.at("opc").reorder(t, y, x).reorder_storage(t, y, x)
        .bound(t, 0, iterations + 1)
        .compute_at(fw.at("volumetric_shaded"), y);

        fw.at("volumetric_shaded").reorder(t, y, x).reorder_storage(t, y, x)
        .bound(t, 0, iterations + 1)
        .compute_at(fw.at("forward"), yi);

        fw.at("forward").reorder(y, x).reorder_storage(y, x)
        .gpu_tile(y, x, yo, xo, yi, xi, 64, 64).vectorize(yi);
        fw.at("normals_debug").reorder(y, x).reorder_storage(y, x)
        .gpu_tile(y, x, yo, xo, yi, xi, 64, 64).vectorize(yi);
    }

    void generate() {
#ifdef DEBUG_TRACER
        debug_(i, t, x, y, c) = cast<uint8_t>(0);
#endif //DEBUG_TRACER
        sb = std::unordered_map<std::string, std::shared_ptr<GridSDF>>();

        tr = RDom(0, iterations);

        GridSDF grid_sdf = create_grid_sdf();

        Func end("end");

        model_transform(x) = translation_(x);

        /*Func target_transform("target_transform");
        target_transform(x) = 0.0f;
        target_transform(0) = 1.0f;
        target_transform(1) = 2.0f;
        target_transform(2) = 3.0f;

        GridSDF target_sdf = to_grid_sdf(example_box,
        {-4.0f, -4.0f, -4.0f},
        {4.0f, 4.0f, 4.0f}, 128, 128, 128);*/

        // controls forward vs backwards pass
        //end(x, y) = fw_pass(x, y);
        //end(x, y) = gradient(x, y, 10);

        parameter_map fw_pass = forward_pass("model", grid_sdf, model_transform);
        //parameter_map target = forward_pass("target", target_sdf, target_transform);

        //parameter_map bw_pass = backwards_pass(grid_sdf, fw_pass, target.at("forward"));

        Func target("target");
        target(x, y) = {
            target_(x, y, 0),
            target_(x, y, 1),
            target_(x, y, 2)
        };
        parameter_map bw_pass = backwards_pass(grid_sdf, fw_pass, target);

        Func fw_pass_fwd = fw_pass.at("forward");

        forward_(x, y, c) = 0.0f;
        forward_(x, y, 0) = fw_pass_fwd(x, y)[0];
        forward_(x, y, 1) = fw_pass_fwd(x, y)[1];
        forward_(x, y, 2) = fw_pass_fwd(x, y)[2];

        //schedule_forward_pass(fw_pass);
        //fw_pass_fwd.compute_root();
        //forward_.reorder(c, y, x).reorder_storage(c, y, x);

        d_l_sdf_(x, y, c) = bw_pass.at("dLoss_dSDF")(x, y, c);
        d_l_translation_(x) = bw_pass.at("dLoss_dTranslation")(x);
        //d_l_sdf_(x, y, c) = 0.0f;
        //d_l_translation_(x) = 0.0f;

        //record(fw_pass.at("volumetric_shaded"));
        //record(fw_pass.at("forward"), true);
        //record(target.at("volumetric_shaded"));
        //record(bw_pass.at("dLoss_dRayVec"), true);
        record(bw_pass.at("debug_gradient"));

        // optimizing
        /*float lr = 1e-1f;
        Func optimization("optimization");
        optimization(x, y, c) = {0.0f, 0.0f, 0.0f};
        for (int epoch = 0; epoch < 128; epoch++) {
            Func new_sdf("new_sdf_" + std::to_string(epoch));
            Func gradient("gradient" + std::to_string(epoch));
            gradient(x, y, c) = backwards_pass(grid_sdf, fw_pass, target)(x, y, c);
            new_sdf(x, y, c) = grid_sdf.buffer(x, y, c) - gradient(x, y, c) * lr;
            optimization(x, y, epoch) = fw_pass(x, y);

            grid_sdf.buffer = new_sdf;
        }
        record(optimization);*/

        // flip image and RGB -> BGR to match OpenCV's output
        /*out_(x, y, c) = 0.0f;
        out_(x, y, 0) = clamp(end(y, x)[2], 0.0f, 1.0f);
        out_(x, y, 1) = clamp(end(y, x)[1], 0.0f, 1.0f);
        out_(x, y, 2) = clamp(end(y, x)[0], 0.0f, 1.0f);*/
        //out_(x, y, c) = 0.0f;
        //out_(x, y, 0) = end(y, x)[0];
        //out_(x, y, 1) = end(y, x)[1];
        //out_(x, y, 2) = end(y, x)[2];

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
            translation_.dim(0).set_bounds_estimate(0, 3);

            p0_.dim(0).set_bounds_estimate(0, 3);
            p1_.dim(0).set_bounds_estimate(0, 3);

            forward_.dim(0).set_bounds_estimate(0, 1920)
            .dim(1).set_bounds_estimate(0, 1080)
            .dim(2).set_bounds_estimate(0, 3);

            d_l_sdf_.dim(0).set_bounds_estimate(0, 128)
            .dim(1).set_bounds_estimate(0, 128)
            .dim(2).set_bounds_estimate(0, 128);

            d_l_translation_.dim(0).set_bounds_estimate(0, 3);

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
            //options.gpu_tile_channel = 1;
            //options.unroll_rvar_size = 10;

#ifdef DEBUG_TRACER
            std::vector<Func> output_func({forward_, d_l_sdf_, d_l_translation_, debug_, num_debug});
#else
            std::vector<Func> output_func({forward_, d_l_sdf_, d_l_translation_});
#endif // DEBUG_TRACER
            // applies simple GPU autoschedule to everything
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
                {"translation_.extent.0", 3},
                {"translation_.min.0", 0},
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
                {
                    {0, 128},
                    {0, 128},
                    {0, 128}
                },
                {
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

            //print_func_dependencies(dr.funcs(sb->buffer).at(0).function());

            // sets every dependency to compute_root its ancestor
            for (auto entry : sb) {
                apply_auto_schedule(entry.second->buffer);
            }
            apply_auto_schedule(projection_);
            apply_auto_schedule(view_);
        }
    }
};

HALIDE_REGISTER_GENERATOR(TracerGenerator, tracer_render)

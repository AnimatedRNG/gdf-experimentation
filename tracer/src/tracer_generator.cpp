#include <iostream>
#include <tuple>

#include "Halide.h"

#include "matmul.hpp"

#include <stdio.h>

using namespace Halide;

constexpr static int iterations = 300;

Var x("x"), y("y"), c("c"), t("t");
Var dx("dx"), dy("dy"), dz("dz");
RDom tr;

class GridSDF {
  public:

    GridSDF(Halide::Func _buffer, TupleVec<3> _p0, TupleVec<3> _p1, int n0, int n1,
            int n2) :
        buffer(_buffer),
        p0(_p0),
        p1(_p1),
        n({n0, n1, n2}), nx(n0), ny(n1), nz(n2) { }

    Halide::Func buffer;
    TupleVec<3> p0;
    TupleVec<3> p1;
    TupleVec<3> n;

    int nx, ny, nz;
};

// For debugging analytical functions
GridSDF to_grid_sdf(std::function<Expr(TupleVec<3>)> sdf,
                    TupleVec<3> p0,
                    TupleVec<3> p1,
                    int nx, int ny, int nz) {
    Func field_func("field_func");
    field_func(dx, dy, dz) = sdf(TupleVec<3>({
        (dx / cast<float>(nx)) * (p1[0] - p0[0]) + p0[0],
        (dy / cast<float>(ny)) * (p1[1] - p0[1]) + p0[1],
        (dz / cast<float>(nz)) * (p1[2] - p0[2]) + p0[2]
    }));
    Halide::Buffer<float> buffer = field_func.realize(nx, ny, nz);

    return GridSDF(Func(buffer), p0, p1, nx, ny, nz);
}

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

void apply_auto_schedule(Func F) {
    std::map<std::string, Internal::Function> flist =
        Internal::find_transitive_calls(
            F.function());
    flist.insert(std::make_pair(F.name(), F.function()));
    std::map<std::string, Internal::Function>::iterator fit;
    for (fit = flist.begin(); fit != flist.end(); fit++) {
        Func f(fit->second);
        f.compute_root();
        std::cout << "Warning: applying default schedule to " << f.name() << std::endl;
    }
    std::cout << std::endl;
}

/*Expr example_sphere(Tuple position) {
    return Halide::sqrt(
               position[0] * position[0] +
               position[1] * position[1] +
               position[2] * position[2]) - 3.0f;
               }*/

Expr example_sphere(TupleVec<3> position) {
    return norm(position) - 3.0f;
}

class TracerGenerator : public Halide::Generator<TracerGenerator> {
  public:

    Input<Buffer<float>> projection_{"projection", 2};
    Input<Buffer<float>> view_{"view", 2};
    Input<int32_t> width{"width"};
    Input<int32_t> height{"height"};
    Output<Buffer<float>> out_{"out", 3};
    Output<Buffer<float>> debug_{"debug", 2};

    std::tuple<std::tuple<Func, Func>, Func> projection(Func proj_matrix,
            Func view_matrix,
            float near = 0.1f) {
        // TODO: Get rid of scheduling calls
        Func rays("rays");

        Func ss_norm;
        ss_norm(x, y) = {
            cast<float>(x) / (cast<float>(width)),
            cast<float>(y) / (cast<float>(height))
        };

        Func clip_space("clip_space");
        clip_space(x, y, c) = 1.0f;
        clip_space(x, y, 0) = ss_norm(x, y)[0] * 2.0f - 1.0f;
        clip_space(x, y, 1) = ss_norm(x, y)[1] * 2.0f - 1.0f;
        clip_space.bound(c, 0, 4).unroll(c);

        Func viewproj_inv("viewproj_inv");
        viewproj_inv =
            matmul::inverse(matmul::product(proj_matrix, view_matrix));

        Func view_inv("view_inv");
        view_inv =
            matmul::inverse(view_matrix);
        debug_(x, y) = 0.0f;

        RDom k(0, 4);
        Func homogeneous("homogeneous");
        homogeneous(x, y, c) = sum(viewproj_inv(c, k) *
                                   clip_space(x, y, k));
        homogeneous.bound(c, 0, 4).unroll(c);
        viewproj_inv.compute_at(homogeneous, x);

        Func origin("origin");
        origin(c) = view_inv(c, 3);
        origin.bound(c, 0, 4).unroll(c);

        Func projected("projected");
        projected(x, y, c) = (homogeneous(x, y, c) / homogeneous(x, y, 3))
                             - origin(c);
        projected.bound(c, 0, 4).unroll(c);
        homogeneous.compute_at(projected, x);
        origin.compute_at(projected, x);

        Func ray_vec("ray_vec");
        RDom norm_k(0, 3);
        // could use fast inverse sqrt, but not worth accuracy loss
        Expr ray_vec_norm = Halide::sqrt(
                                sum(
                                    projected(x, y, norm_k) *
                                    projected(x, y, norm_k)));
        ray_vec(x, y) = Tuple(projected(x, y, 0) / ray_vec_norm,
                              projected(x, y, 1) / ray_vec_norm,
                              projected(x, y, 2) / ray_vec_norm);
        projected.compute_at(ray_vec, x);

        Func ray_pos("ray_pos");
        //ray_pos(x, y, c) = origin(c) + ray_vec(x, y, c) * near;
        ray_pos(x, y) = Tuple(origin(0) + ray_vec(x, y)[0] * near,
                              origin(1) + ray_vec(x, y)[1] * near,
                              origin(2) + ray_vec(x, y)[2] * near);

        //Func projection_result("projection_result");
        //projection_result(x, y) = {ray_pos(x, y), ray_vec(x, y)};
        //ray_vec.compute_at(projection_result, x);
        //ray_pos.compute_at(projection_result, x);

        //apply_auto_schedule(projection_result);

        std::tuple<Func, Func> projection_result(ray_pos, ray_vec);
        apply_auto_schedule(ray_pos);
        apply_auto_schedule(ray_vec);

        return std::make_tuple(projection_result, origin);
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
        light_vec(x, y, t) = {0.0f, 0.0f, 0.0f};
        light_vec(x, y, tr) = (light_position - Tuple(positions(x, y, tr))).get();

        Func light_vec_norm("light_vec_norm");
        light_vec_norm(x, y, t) = 0.0f;
        light_vec_norm(x, y, tr) = norm(TupleVec<3>(light_vec(x, y, tr)));

        Func light_vec_normalized("light_vec_normalized");
        light_vec_normalized(x, y, t) = {0.0f, 0.0f, 0.0f};
        light_vec_normalized(x, y, tr) =
            (TupleVec<3>(light_vec(x, y, tr))
             / Expr(light_vec_norm(x, y, tr))).get();

        Func ray_position("ray_position");
        ray_position(x, y, t) = {0.0f, 0.0f, 0.0f};
        ray_position(x, y, tr) = (TupleVec<3>(positions(x, y, tr)) +
                                  Tuple(light_vec_normalized(x, y, tr))).get();

        TupleVec<3> normal_sample =
            trilinear<3>(normals, TupleVec<3>(Tuple(positions(x, y, tr))));
        Func diffuse("diffuse");
        diffuse(x, y, t) = {0.0f, 0.0f, 0.0f};
        diffuse(x, y, tr) = (kd * clamp(dot(normal_sample,
                                            Tuple(light_vec(x, y, tr))),
                                        0.0f, 1.0f) * light_color).get();

        return diffuse;
    }

    Func shade(Func positions, TupleVec<3> origin, GridSDF normals) {
        TupleVec<3> top_light_color = {0.6f, 0.6f, 0.4f};
        TupleVec<3> self_light_color = {0.4f, 0.0f, 0.4f};

        TupleVec<3> top_light_pos = {10.0f, 30.0f, 0.0f};

        TupleVec<3> self_light_pos = origin;

        Func top_light("top_light"), self_light("self_light");
        top_light(x, y, t) = {0.0f, 0.0f, 0.0f};
        top_light(x, y, tr) = light_source(top_light_color,
                                           positions,
                                           top_light_pos,
                                           normals)(x, y, tr);
        self_light(x, y, t) = {0.0f, 0.0f, 0.0f};
        self_light(x, y, tr) = light_source(self_light_color,
                                            positions,
                                            self_light_pos,
                                            normals)(x, y, tr);

        Func total_light("total_light");
        total_light(x, y, t) = {0.0f, 0.0f, 0.0f};
        total_light(x, y, tr) = (TupleVec<3>(top_light(x, y, tr))
                                 + TupleVec<3>(self_light(x, y, tr))).get();
        //total_light.trace_stores();

        return total_light;
    }

    Func h(GridSDF sdf, unsigned int dim) {
        float h_kern[3] = {1.f, 2.f, 1.f};
        Func h_conv("h_conv");

        switch (dim) {
            case 0:
                h_conv(x, y, c) =
                    sdf.buffer(max(x - 1, 0), y, c) * h_kern[0] +
                    sdf.buffer(x, y, c) * h_kern[1] +
                    sdf.buffer(min(x + 1, sdf.n[0] - 1), y, c) * h_kern[2];
                break;
            case 1:
                h_conv(x, y, c) =
                    sdf.buffer(x, max(y - 1, 0), c) * h_kern[0] +
                    sdf.buffer(x, y, c) * h_kern[1] +
                    sdf.buffer(x, min(y + 1, sdf.n[1] - 1), c) * h_kern[2];
                break;
            case 2:
                h_conv(x, y, c) =
                    sdf.buffer(x, y, max(c - 1, 0)) * h_kern[0] +
                    sdf.buffer(x, y, c) * h_kern[1] +
                    sdf.buffer(x, y, min(c + 1, sdf.n[2] - 1)) * h_kern[2];
                break;
            default:
                throw std::out_of_range("invalid dim for h");
        };

        return h_conv;
    }

    Func h_p(GridSDF sdf, unsigned int dim) {
        float h_p_kern[2] = {1.f, -1.f};
        Func h_p_conv("h_p_conv");

        switch (dim) {
            case 0:
                h_p_conv(x, y, c) =
                    sdf.buffer(max(x - 1, 0), y, c) * h_p_kern[0] +
                    sdf.buffer(min(x + 1, sdf.n[0] - 1), y, c) * h_p_kern[1];
                break;
            case 1:
                h_p_conv(x, y, c) =
                    sdf.buffer(x, max(y - 1, 0), c) * h_p_kern[0] +
                    sdf.buffer(x, min(y + 1, sdf.n[1] - 1), c) * h_p_kern[1];
                break;
            case 2:
                h_p_conv(x, y, c) =
                    sdf.buffer(x, y, max(c - 1, 0)) * h_p_kern[0] +
                    sdf.buffer(x, y, min(c + 1, sdf.n[2] - 1)) * h_p_kern[1];
                break;
            default:
                throw std::out_of_range("invalid dim for h_p");
        };

        return h_p_conv;
    }

    GridSDF sobel(GridSDF sdf) {
        Func sb("sobel");

        Func h_x("h_x"), h_y("h_y"), h_z("h_z");
        Func h_p_x("h_p_x"), h_p_y("h_p_y"), h_p_z("h_p_z");

        h_x(x, y, c) = h(sdf, 0)(x, y, c);
        h_y(x, y, c) = h(sdf, 1)(x, y, c);
        h_z(x, y, c) = h(sdf, 2)(x, y, c);

        h_p_x(x, y, c) = h_p(sdf, 0)(x, y, c);
        h_p_y(x, y, c) = h_p(sdf, 1)(x, y, c);
        h_p_z(x, y, c) = h_p(sdf, 2)(x, y, c);

        sb(x, y, c) = {
            max(h_p_x(x, y, c) * h_y(x, y, c) * h_z(x, y, c), 1e-6f),
            max(h_p_y(x, y, c) * h_z(x, y, c) * h_x(x, y, c), 1e-6f),
            max(h_p_z(x, y, c) * h_x(x, y, c) * h_y(x, y, c), 1e-6f)
        };

        // TODO: come up with a better schedule at some point
        h_x.compute_at(sb, x);
        h_y.compute_at(sb, y);
        h_z.compute_at(sb, c);

        h_p_x.compute_at(sb, x);
        h_p_y.compute_at(sb, y);
        h_p_z.compute_at(sb, c);

        Func sobel_norm("sobel_norm");
        sobel_norm(x, y, c) = norm(sb(x, y, c));

        Func sobel_normalized("sobel_normalized");
        sobel_normalized(x, y, c) = (TupleVec<3>(sb(x, y, c))
                                     / Expr(sobel_norm(x, y, c))).get();

        sb.compute_at(sobel_normalized, x);
        sobel_norm.compute_at(sobel_normalized, x);

        sobel_normalized.compute_root();

        return GridSDF(sobel_normalized, sdf.p0, sdf.p1, sdf.nx, sdf.ny, sdf.nz);
    }

    Func sphere_trace(const GridSDF& sdf,
                      float EPS = 1e-6) {
        Func original_ray_pos("original_ray_pos");
        Func ray_vec("ray_vec");
        Func origin("origin");

        std::forward_as_tuple(std::tie(original_ray_pos,
                                       ray_vec), origin) =
                                           projection(projection_, view_);
        Func pos("pos");
        Expr d("d");
        Func depth("depth");

        GridSDF sb = sobel(sdf);

        // Remember how update definitions work
        pos(x, y, t) = Tuple(0.0f, 0.0f, 0.0f);
        pos(x, y, 0) = original_ray_pos(x, y);
        d = trilinear<1>(sdf, TupleVec<3>(Tuple(pos(x, y, tr))))[0];
        pos(x, y, tr + 1) = (TupleVec<3>(pos(x, y, tr)) +
                             d * TupleVec<3>(ray_vec(x, y))).get();
        Var xi, xo, yi, yo;
        depth(x, y, t) = 0.0f;
        depth(x, y, tr + 1) = d;

        Func shaded("shaded");
        shaded(x, y, t) = {0.0f, 0.0f, 0.0f};
        shaded(x, y, tr) = (TupleVec<3>(
                                shade(pos, {origin(0), origin(1), origin(2)}, sb)(x, y, tr)) *
                            1.0f).get();

        Func endpoint("endpoint");
        //endpoint(x, y) = pos(x, y, iterations - 1);
        /*endpoint(x, y) = {depth(x, y, iterations),
                          depth(x, y, iterations),
                          depth(x, y, iterations)
                          };*/
        endpoint(x, y) = shaded(x, y, iterations - 1);
        endpoint.compute_root();
        pos.unroll(t);
        //pos.compute_at(endpoint, x);
        //pos.store_at(endpoint, x);
        apply_auto_schedule(shaded);
        //apply_auto_schedule(pos);

        return endpoint;
    }

    void generate() {
        tr = RDom(0, iterations);

        Func original_ray_pos("original_ray_pos");
        Func original_ray_vec("original_ray_vec");
        Func origin("origin");

        std::forward_as_tuple(std::tie(original_ray_pos,
                                       original_ray_vec), origin) =
                                           projection(projection_, view_);
        original_ray_pos.compute_root();
        original_ray_vec.compute_root();

        GridSDF grid_sdf = to_grid_sdf(example_sphere,
        {-4.0f, -4.0f, -4.0f},
        {4.0f, 4.0f, 4.0f}, 32, 32, 32);

        Func end("end");
        end(x, y) = sphere_trace(grid_sdf)(x, y);

        //std::cout << Buffer<float>(sobel(grid_sdf).realize(32, 32, 32)[0])(0, 0, 0)
        //          << std::endl;

        /*Func val("val");
        val(dx) = trilinear<1>(grid_sdf, {-3.5f, -3.5f, -3.5f})[0];
        std::cout << "at (-3.5, -3.5, -3.5) " << Buffer<float>(val.realize(1))(0) <<
        std::endl;*/

        //out_(x, y, c) =
        //    matmul::product(matmul::product(Func(b), 3.0f), Func(b))(x, y);
        //out_(x, y, c) = matmul::inverse(Func(a))(x, y);
        //out_(x, y, c) = clamp(rays(x, y, 2 - c)[1], 0.0f, 1.0f);
        //out_(x, y, c) = rays(x, y, c)[1];

        out_(x, y, c) = 0.0f;
        out_(x, y, 0) = clamp(end(x, y)[0], 0.0f, 1.0f);
        out_(x, y, 1) = clamp(end(x, y)[1], 0.0f, 1.0f);
        out_(x, y, 2) = clamp(end(x, y)[2], 0.0f, 1.0f);
        //out_(x, y, 0) = clamp(sobel(grid_sdf)(x / 7, y / 7, 10)[0], 0.0f, 1.0f);
        //out_(x, y, 1) = clamp(sobel(grid_sdf)(x / 7, y / 7, 10)[1], 0.0f, 1.0f);
        //out_(x, y, 2) = clamp(sobel(grid_sdf)(x / 7, y / 7, 10)[2], 0.0f, 1.0f);
    }
};

HALIDE_REGISTER_GENERATOR(TracerGenerator, tracer_render)

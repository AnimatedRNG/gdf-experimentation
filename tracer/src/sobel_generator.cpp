#include <iostream>
#include <tuple>

#include "Halide.h"

#include "grid_sdf.hpp"
#include "matmul.hpp"

#include <stdio.h>

using namespace Halide;

namespace {

    class SobelGenerator : public Halide::Generator<SobelGenerator> {
      public:
        Input<Func> sdf_{"sdf", 3};
        Input<int32_t> d0{"d0"};
        Input<int32_t> d1{"d1"};
        Input<int32_t> d2{"d2"};
        Output<Func> normals_{"normals", {Float(32), Float(32), Float(32)}, 3};

        Func h_x{"h_x"}, h_y{"h_y"}, h_z{"h_z"};
        Func h_p_x{"h_p_x"}, h_p_y{"h_p_y"}, h_p_z{"h_p_z"};

        Func sb{"sobel"};
        Func sobel_norm{"sobel_norm"};
        Func sobel_normalized{"sobel_normalized"};

        std::vector<Func> intermediates;

        Func h(Func clamped, unsigned int dim) {
            Func h_conv("h_conv");
            float h_kern[3] = {1.f, 2.f, 1.f};

            switch (dim) {
                case 0:
                    h_conv(x, y, z) =
                        clamped(x - 1, y, z) * h_kern[0] +
                        clamped(x, y, z) * h_kern[1] +
                        clamped(x + 1, y, z) * h_kern[2];
                    break;
                case 1:
                    h_conv(x, y, z) =
                        clamped(x, y - 1, z) * h_kern[0] +
                        clamped(x, y, z) * h_kern[1] +
                        clamped(x, y + 1, z) * h_kern[2];
                    break;
                case 2:
                    h_conv(x, y, z) =
                        clamped(x, y, z - 1) * h_kern[0] +
                        clamped(x, y, z) * h_kern[1] +
                        clamped(x, y, z + 1) * h_kern[2];
                    break;
                default:
                    throw std::out_of_range("invalid dim for h");
            };

            intermediates.push_back(h_conv);

            return h_conv;
        }

        Func h_p(Func clamped, unsigned int dim) {
            Func h_p_conv("h_p_conv");
            float h_p_kern[2] = {1.f, -1.f};

            switch (dim) {
                case 0:
                    h_p_conv(x, y, z) =
                        clamped(x - 1, y, z) * h_p_kern[0] +
                        clamped(x + 1, y, z) * h_p_kern[1];
                    break;
                case 1:
                    h_p_conv(x, y, z) =
                        clamped(x, y - 1, z) * h_p_kern[0] +
                        clamped(x, y + 1, z) * h_p_kern[1];
                    break;
                case 2:
                    h_p_conv(x, y, z) =
                        clamped(x, y, z - 1) * h_p_kern[0] +
                        clamped(x, y, z + 1) * h_p_kern[1];
                    break;
                default:
                    throw std::out_of_range("invalid dim for h_p");
            };

            intermediates.push_back(h_p_conv);

            return h_p_conv;
        }

        void generate() {
            Func clamped = BoundaryConditions::repeat_edge(sdf_, {{0, d0}, {0, d1}, {0, d2}});

            h_x = h(clamped, 0);
            h_y = h(clamped, 1);
            h_z = h(clamped, 2);

            h_p_x = h_p(clamped, 0);
            h_p_y = h_p(clamped, 1);
            h_p_z = h_p(clamped, 2);

            sb(x, y, z) = {
                h_p_x(x, y, z)* h_y(x, y, z)* h_z(x, y, z),
                h_p_y(x, y, z)* h_z(x, y, z)* h_x(x, y, z),
                h_p_z(x, y, z)* h_x(x, y, z)* h_y(x, y, z)
            };
            sb(x, y, z) = {
                select(abs(sb(x, y, z)[0]) < 1e-6f, 1e-6f, sb(x, y, z)[0]),
                select(abs(sb(x, y, z)[1]) < 1e-6f, 1e-6f, sb(x, y, z)[1]),
                select(abs(sb(x, y, z)[2]) < 1e-6f, 1e-6f, sb(x, y, z)[2])
            };

            intermediates.push_back(h_x);
            intermediates.push_back(h_y);
            intermediates.push_back(h_z);

            intermediates.push_back(h_p_x);
            intermediates.push_back(h_p_y);
            intermediates.push_back(h_p_z);

            sobel_norm(x, y, z) = norm(sb(x, y, z)) * -1.0f;
            intermediates.push_back(sobel_norm);

            normals_(x, y, z) = (TupleVec<3>(sb(x, y, z))
                                     / Expr(sobel_norm(x, y, z))).get();

            if (auto_schedule) {
                /*sdf_.estimate(x, 0, 64)
                .estimate(y, 0, 64)
                .estimate(z, 0, 64);
                normals_.estimate(x, 0, 64)
                .estimate(y, 0, 64)
                .estimate(z, 0, 64);*/
                std::cout << "auto schedule sobel" << std::endl;
            } else {
                // TODO: come up with a better schedule at some point
                h_x.compute_at(sb, x);
                h_y.compute_at(sb, y);
                h_z.compute_at(sb, z);

                h_p_x.compute_at(sb, x);
                h_p_y.compute_at(sb, y);
                h_p_z.compute_at(sb, z);

                sb.compute_at(normals_, x);
                sobel_norm.compute_at(normals_, x);
                normals_.compute_root();

                std::cout << "not auto schedule sobel" << std::endl;
            }
        }

      private:
        Var x, y, z;
    };

} // namespace

HALIDE_REGISTER_GENERATOR(SobelGenerator, sobel)

#!/usr/bin/env python3

from sympy import *

init_printing()


def lerp(a, b, alpha):
    return (1 - alpha) * a + alpha * b


def trilinear(alpha_x, alpha_y, alpha_z,
              c000, c001, c010, c011, c100, c101, c110, c111):
    c00 = lerp(c000, c100, alpha_x)
    c01 = lerp(c001, c101, alpha_x)
    c10 = lerp(c010, c110, alpha_x)
    c11 = lerp(c011, c111, alpha_x)

    c0 = lerp(c00, c10, alpha_y)
    c1 = lerp(c01, c11, alpha_y)

    return lerp(c0, c1, alpha_z)


def gen_neighbors(name="SDF"):
    neighbors = {}
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                neighbors[(i, j, k)] = symbols(
                    "{}{}{}{}".format(name, i, j, k))
    return neighbors


def h(sdf, dim):
    if dim == 0:
        return sdf[(-1, 0, 0)] * 1 + sdf[(0, 0, 0)] * 2 + sdf[(1, 0, 0)] * 1
    elif dim == 1:
        return sdf[(0, -1, 0)] * 1 + sdf[(0, 0, 0)] * 2 + sdf[(0, 1, 0)] * 1
    elif dim == 2:
        return sdf[(0, 0, -1)] * 1 + sdf[(0, 0, 0)] * 2 + sdf[(0, 0, 1)] * 1


def h_p(sdf, dim):
    if dim == 0:
        return sdf[(-1, 0, 0)] * 1 + sdf[(1, 0, 0)] * -1
    elif dim == 1:
        return sdf[(0, -1, 0)] * 1 + sdf[(0, 1, 0)] * -1
    elif dim == 2:
        return sdf[(0, 0, -1)] * 1 + sdf[(0, 0, 1)] * -1


def sobel(neighbors):
    h_x = h(neighbors, 0)
    h_y = h(neighbors, 1)
    h_z = h(neighbors, 2)

    h_p_x = h_p(neighbors, 0)
    h_p_y = h_p(neighbors, 1)
    h_p_z = h_p(neighbors, 2)

    sb = {}
    sb["sb_0"] = h_p_x * h_y * h_z
    sb["sb_1"] = h_p_y * h_z * h_x
    sb["sb_2"] = h_p_z * h_x * h_y

    return sb


def jacobian(inputs, outputs):
    for output_name, output in outputs.items():
        print("d{}/d?: ".format(output_name))
        for input_name, inp in inputs.items():
            print("\td{}/d{}: {}".format(output_name, input_name,
                                         diff(output, inp)))


def trilinear_derivatives():
    alpha_x, alpha_y, alpha_z = symbols('alpha_x alpha_y alpha_z')
    c000, c001, c010, c011, c100, c101, c110, c111 = \
        symbols('c000 c001 c010 c011 c100 c101 c110 c111')
    tri = trilinear(alpha_x, alpha_y, alpha_z, c000, c001,
                    c010, c011, c100, c101, c110, c111)

    print('derivative of {} is {}'.format(c000, diff(tri, c000)))


if __name__ == '__main__':
    # trilinear_derivatives()
    neighbors = gen_neighbors()
    jacobian(neighbors, sobel(neighbors))

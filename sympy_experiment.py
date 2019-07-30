#!/usr/bin/env python3

from sympy import *

init_printing()


def lerp(a, b, alpha):
    return (1 - alpha) * a + alpha * b


def lerp_vec(va, vb, alpha):
    assert(len(va) == len(vb))
    va_name = next(iter(va.keys()))
    va_name = va_name[:va_name.find("_")]
    vb_name = next(iter(vb.keys()))
    vb_name = vb_name[:vb_name.find("_")]

    va_vec = {int(name[name.find("_") + 1:]): (name, symbol)
              for name, symbol in va.items()}
    vb_vec = {int(name[name.find("_") + 1:]): (name, symbol)
              for name, symbol in vb.items()}

    lerped_vec = {"lerp({},{})_{}".format(va_name, vb_name, i): lerp(
        va_vec[i][1], vb_vec[i][1], alpha) for i in range(len(va_vec))}
    return lerped_vec


def trilinear(alpha, c):
    c00 = lerp(c[(0, 0, 0)], c[1, 0, 0], alpha[0])
    c01 = lerp(c[(0, 0, 1)], c[1, 0, 1], alpha[0])
    c10 = lerp(c[(0, 1, 0)], c[(1, 1, 0)], alpha[0])
    c11 = lerp(c[(0, 1, 1)], c[(1, 1, 1)], alpha[0])

    c0 = lerp(c00, c10, alpha[1])
    c1 = lerp(c01, c11, alpha[1])

    return lerp(c0, c1, alpha[2])


def trilinear_vec(alpha, c):
    c00 = lerp_vec(c[(0, 0, 0)], c[1, 0, 0], alpha[0])
    c01 = lerp_vec(c[(0, 0, 1)], c[1, 0, 1], alpha[0])
    c10 = lerp_vec(c[(0, 1, 0)], c[(1, 1, 0)], alpha[0])
    c11 = lerp_vec(c[(0, 1, 1)], c[(1, 1, 1)], alpha[0])

    c0 = lerp_vec(c00, c10, alpha[1])
    c1 = lerp_vec(c01, c11, alpha[1])

    return lerp_vec(c0, c1, alpha[2])


def gen_neighbors(name="SDF", r=1):
    neighbors = {}
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            for k in range(-r, r + 1):
                neighbors[(i, j, k)] = symbols(
                    "{}{}{}{}".format(name, i, j, k))
    return neighbors


def get_sobel_neighbors(sdf_neighbors):
    sobel_neighbors = {}

    sobel_offsets = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                sobel_offsets.append((i, j, k))

    for offset in sobel_offsets:
        ni, nj, nk = offset
        for i in range(ni-1, ni+2):
            for j in range(nj-1, nj+2):
                for k in range(nk-1, nk+2):
                    if (ni, nj, nk) not in sobel_neighbors:
                        sobel_neighbors[(ni, nj, nk)] = {
                            (i - ni, j - nj, k - nk): sdf_neighbors[(i, j, k)]}
                    else:
                        sobel_neighbors[(ni, nj, nk)][(i - ni, j - nj, k - nk)] = \
                            sdf_neighbors[(i, j, k)]
    return sobel_neighbors


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


def sobel(neighbors, offset):
    print("running sobel on: {}".format(neighbors))
    h_x = h(neighbors, 0)
    h_y = h(neighbors, 1)
    h_z = h(neighbors, 2)

    h_p_x = h_p(neighbors, 0)
    h_p_y = h_p(neighbors, 1)
    h_p_z = h_p(neighbors, 2)

    sb = {}
    sb["sb{}_0".format(offset)] = h_p_x * h_y * h_z
    sb["sb{}_1".format(offset)] = h_p_y * h_z * h_x
    sb["sb{}_2".format(offset)] = h_p_z * h_x * h_y

    return sb


def trilinear_sobel(alpha, all_sdf_neighbors):
    # map from sobel evaluation position -> all the sdf grid cells that it touches
    sobel_neighbors = get_sobel_neighbors(all_sdf_neighbors)

    # sobel offset maps each sobel evaluation based on trilinear interpolation
    # sdf_neighbors is a dictionary mapping from position to the symbol
    # it's complicated :/
    return trilinear_vec(alpha, {sobel_offset: sobel(sdf_neighbors, sobel_offset)
                                 for sobel_offset, sdf_neighbors in sobel_neighbors.items()})


def jacobian(inputs, outputs):
    for output_name, output in outputs.items():
        print("d{}/d?: ".format(output_name))
        for input_name, inp in inputs.items():
            print("\t/d{}: {}".format(input_name,
                                      diff(output, inp)))


def trilinear_derivatives():
    alphas = [symbols("alpha{}".format(i)) for i in range(3)]
    cs = {(i, j, k): symbols("c{}{}{}".format(i, j, k))
          for i in range(2) for j in range(2) for k in range(2)}
    tri = trilinear(alphas, cs)

    print('derivative of {} is {}'.format(
        cs[(0, 0, 0)], diff(tri, cs[(0, 0, 0)])))


def trilinear_sobel_derivatives():
    neighbors = gen_neighbors("SDF", 2)
    alphas = [symbols("alpha{}".format(i)) for i in range(3)]
    tsb = trilinear_sobel(alphas, neighbors)
    jacobian(neighbors, tsb)


if __name__ == '__main__':
    # trilinear_derivatives()
    #neighbors = gen_neighbors()
    #jacobian(neighbors, sobel(neighbors))

    trilinear_sobel_derivatives()

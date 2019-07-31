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
                    "{}!{}!{}!{}".format(name, i, j, k))
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


def jacobian(inputs, outputs, ignore_zeros=False):
    for output_name, output in outputs.items():
        print("\nd{}/d?: ".format(output_name))
        for input_name, inp in inputs.items():
            deriv = diff(output, inp)
            if not (ignore_zeros and simplify(deriv) == 0):
                print("\t/d{}: {}".format(input_name,
                                          deriv))


index_to_xyzw = {0: "x", 1: "y", 2: "z", 3: "w"}


def vec_elem_to_mapping(vec_str, vec_name):
    global index_to_xyzw
    return "{name}.{index}".format(name=vec_name, index=index_to_xyzw[int(vec_str.split("_")[1])])


def matrix_elem_to_mapping(matrix_str, matrix_name):
    global index_to_xyzw
    if isinstance(matrix_str, str):
        return "index({name}, {indices})".format(name=matrix_name, indices=", ".join(
            "offset.{} + {}".format(index_to_xyzw[actual_index], str_index)
            for actual_index, str_index in enumerate(matrix_str.split("!")[1:])
        ))
    else:
        return "index({name}, {indices})".format(name=matrix_name, indices=", ".join(
            "offset.{} + {}".format(index_to_xyzw[actual_index], index)
            for actual_index, index in enumerate(matrix_str)
        ))


def jacobian_to_code(inputs, outputs, mappings):
    code = ""
    for output_name, output in outputs.items():
        output_mapped = mappings[output_name]
        for input_name, inp in inputs.items():
            input_mapped = mappings[input_name]
            code += "    {} = {};\n".format(output_mapped, input_mapped)
    print(code)


def trilinear_derivatives():
    alphas = [symbols("alpha{}".format(i)) for i in range(3)]
    cs = {(i, j, k): symbols("c!{}!{}!{}".format(i, j, k))
          for i in range(2) for j in range(2) for k in range(2)}
    tri = trilinear(alphas, cs)

    print(jacobian(cs, {"tri": tri}))
    # print('derivative of {} is {}'.format(
    #    cs[(0, 0, 0)], expand(diff(tri, cs[(0, 0, 0)]))
    # ))


def trilinear_sobel_derivatives():
    neighbors = gen_neighbors("SDF", 2)
    alphas = [symbols("alpha{}".format(i)) for i in range(3)]
    tsb = trilinear_sobel(alphas, neighbors)
    jacobian(neighbors, tsb, True)

    # mappings = {tsb_name: vec_elem_to_mapping(tsb_name, "dNormal_dSDF")
    #            for tsb_name in tsb.keys()}
    # mappings.update({neighbor_name: matrix_elem_to_mapping(neighbor_name, "SDF")
    #                 for neighbor_name in neighbors.keys()})
    #print(jacobian_to_code(neighbors, tsb, mappings))


def opc_test():
    SDF = symbols("SDF")
    opc_t = Function("opc_t")(SDF)
    g_d = Function("g_d")(SDF)

    opc_t1 = opc_t + g_d
    pprint(diff(opc_t1, SDF))


def vs_test():
    SDF = symbols("SDF")
    vs_t = Function("vs_t")(SDF)

    scattering_t = Function("scattering_t")(SDF)
    opc_t1 = Function("opc_t1")(SDF)
    intensity = Function("intensity")(SDF)
    k, step = symbols("k step")

    vs_t1 = vs_t + scattering_t * exp(k * opc_t1) * intensity * step
    pprint(diff(vs_t1, SDF))


if __name__ == '__main__':
    trilinear_derivatives()
    #neighbors = gen_neighbors()
    #jacobian(neighbors, sobel(neighbors))

    # trilinear_sobel_derivatives()
    # vs_test()
    # opc_test()

    #cs = "cs!0!2!0"
    #print(matrix_elem_to_mapping(cs, "cs"))

    #cs = "cs_1"
    #print(vec_elem_to_mapping(cs, "cs"))

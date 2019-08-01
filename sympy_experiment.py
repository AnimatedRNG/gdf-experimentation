#!/usr/bin/env python3

from sympy import *

init_printing()


def clamp(a, b, c):
    return Min(Max(a, b), c)


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


def matrix_elem_to_mapping(matrix_str, matrix_name, offset_name="offset"):
    global index_to_xyzw
    if isinstance(matrix_str, str):
        return "index({name}, {indices})".format(name=matrix_name, indices=", ".join(
            "{}.{} + {}".format(offset_name,
                                index_to_xyzw[actual_index], str_index)
            for actual_index, str_index in enumerate(matrix_str.split("!")[1:])
        ))
    else:
        return "index({name}, {indices})".format(name=matrix_name, indices=", ".join(
            "{}.{} + {}".format(offset_name,
                                index_to_xyzw[actual_index], index)
            for actual_index, index in enumerate(matrix_str)
        ))


def jacobian_to_code(inputs, outputs, mappings):
    code = ""
    for output_name, output in outputs.items():
        output_mapped = mappings[output_name]
        for input_name, inp in inputs.items():
            input_mapped = mappings[input_name]
            deriv = str(diff(output, inp))
            for name, replacement in mappings.items():
                deriv = deriv.replace(str(name), replacement)
            code += "    {}{} = {};\n".format(input_mapped,
                                              output_mapped, deriv)
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
    jacobian(neighbors, tsb, False)

    '''mappings = {tsb_name: vec_elem_to_mapping(tsb_name, "")
                for tsb_name in tsb.keys()}
    mappings.update({"alpha" + str(index): "alpha.{}".format(index_to_xyzw[index])
                     for index, value in enumerate(alphas)})
    mappings.update({neighbor_name: matrix_elem_to_mapping(neighbor_name, "normals_vals", "delete_me")
                     for neighbor_name in neighbors.keys()})
    sdf_names = {"SDF!{}!{}!{}".format(i, j, k): "index(sdf_vals, offset.x + {}, offset.y + {}, offset.z + {})"
                 .format(i, j, k)
                 for i in range(-1, 3) for j in range(-1, 3) for k in range(-1, 3)}
    mappings.update(sdf_names)

    print(jacobian_to_code(neighbors, tsb, mappings))'''


def opc_test():
    SDF = symbols("SDF")
    opc_t = Function("opc_t")(SDF)
    g_d = Function("g_d")(SDF)

    opc_t1 = opc_t + g_d
    pprint(diff(opc_t1, SDF))


def vs_test():
    SDF = symbols("SDF")
    vs_t = Function("vs_t")(SDF)

    g_d_t = Function("g_d_t")(SDF)
    opc_t = Function("opc_t")(SDF)
    opc_t1_e = opc_t + g_d_t
    opc_t1_diff = diff(opc_t1_e, SDF)

    scattering_t = Function("scattering_t")(SDF)
    opc_t1 = Function("opc_t1")(SDF)
    intensity = Function("intensity")(SDF)
    k, step = symbols("k step")

    vs_t1 = vs_t + scattering_t * exp(k * opc_t1) * intensity * step
    deriv = diff(vs_t1, SDF)
    print("\n\nbase derivative")
    pprint(deriv)
    deriv = deriv.subs(Derivative(opc_t1), opc_t1_diff)
    print("\n\nderivative after substituting opc_t1")
    pprint(deriv)


def light_source(light_color,
                 position,
                 light_position,
                 normal,
                 kd=symbols("kd"),
                 ks=symbols("ks"),
                 ka=symbols("ka")):
    light_vec = (Matrix(light_position) - Matrix(position)).normalized()
    ray_position = position + light_vec
    diffuse = kd * clamp(Matrix(normal).dot(Matrix(light_vec)),
                         0.0, 1.0) * Matrix(light_color)
    return Matrix(diffuse)


def shade(position, origin, normal):
    top_light_color = MatrixSymbol("toplightcolor", 3, 1)
    self_light_color = MatrixSymbol("selflightcolor", 3, 1)

    top_light_pos = MatrixSymbol("toplightpos", 3, 1)
    self_light_pos = MatrixSymbol("selflightpos", 3, 1)

    top_light = light_source(top_light_color, position, top_light_pos, normal)
    self_light = light_source(
        self_light_color, position, self_light_pos, normal)

    total_light = top_light + self_light

    return total_light


def intensity_derivative(position, origin, normal, SDF):
    return diff(simplify(shade(position, origin, normal)), SDF)


def whole_pipeline_test():
    '''SDF = symbols("SDF")
    vs_t = Function("vs_t")(SDF)

    scattering_t = Function("scattering_t")(SDF)
    opc_t1 = Function("opc_t1")(SDF)
    intensity = Function("intensity")(SDF)
    k, step = symbols("k step")

    vs_t1 = vs_t + scattering_t * exp(k * opc_t1) * intensity * step
    pprint(diff(vs_t1, SDF))'''

    SDF = symbols("SDF")
    # position_placeholder = FunctionMatrix(
    #    3, 1, lambda i, j: Function("position_{}{}".format(i, j), real=True)(SDF))
    position_placeholder = FunctionMatrix(
        3, 1, lambda i, j: Function("position_{}{}".format(i, j), real=True)())
    origin = symbols("origin")
    normal_placeholder = FunctionMatrix(
        3, 1, lambda i, j: Function("normal_{}{}".format(i, j), real=True)(SDF))

    print(intensity_derivative(position_placeholder,
                               origin, normal_placeholder, SDF))


if __name__ == '__main__':
    # trilinear_derivatives()
    # neighbors = gen_neighbors()
    # jacobian(neighbors, sobel(neighbors))

    # trilinear_sobel_derivatives()
    vs_test()
    # opc_test()

    # cs = "cs!0!2!0"
    # print(matrix_elem_to_mapping(cs, "cs"))

    # cs = "cs_1"
    # print(vec_elem_to_mapping(cs, "cs"))

    # whole_pipeline_test()

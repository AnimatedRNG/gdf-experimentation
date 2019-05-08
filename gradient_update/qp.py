#!/usr/bin/env python3

import osqp
from scipy.sparse import identity, csc_matrix, coo_matrix
from scipy.signal import correlate2d
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=2.0)


def gen_sdf(f, dims, *args):
    sdf = np.zeros(dims, dtype=np.float64)
    for i in range(sdf.shape[0]):
        for j in range(sdf.shape[1]):
            sdf[i, j] = f(np.array([i, j], dtype=np.float64), *args)
    return sdf


def simple_kernels():
    #kern = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=np.float64)
    kern = np.zeros((11, 11), dtype=np.float64)
    kern[5] = [0, 1/280, -4/105, 1/5, -4/5, 0,
               4/5, -1/5, 4/105, -1/280, 0]
    return kern, kern.T


def sobel_kernels(size):
    assert size % 2 == 1
    gx = np.zeros((size, size), dtype=np.float64)
    gy = np.zeros((size, size), dtype=np.float64)
    for i_raw in range(size):
        for j_raw in range(size):
            i = i_raw - size // 2
            j = j_raw - size // 2
            denom = (i * i + j * j)
            if denom != 0:
                gy[i_raw, j_raw] = i / (i * i + j * j)
                gx[i_raw, j_raw] = j / (i * i + j * j)
            else:
                gy[i_raw, j_raw] = 0
                gx[i_raw, j_raw] = 0

    # I'm too lazy to rederive this
    weights = {
        3: 4,
        5: 12,
        7: 24,
        9: 40,
    }
    total = weights[size]

    print("total: {}".format(total))
    return (gx / total, gy / total)


def correlate(f1, f2):
    '''not used, just an example of how it's supposed to work'''
    output = np.zeros_like(f1)

    def access(p):
        return f1[
            max(min(p[0], f1.shape[0] - 1), 0),
            max(min(p[1], f1.shape[1] - 1), 0)
        ]

    hw = f2.shape[0] // 2
    r_f2 = f2.ravel()
    for i in range(f1.shape[0]):
        for j in range(f1.shape[1]):
            kern_index = 0
            accum = 0.0
            for s_i in range(i - hw, i + hw + 1):
                for s_j in range(j - hw, j + hw + 1):
                    pixel = access((s_i, s_j))
                    accum += pixel * r_f2[kern_index]
                    kern_index += 1
            output[i, j] = accum
    return output


scale = (-4, 19)


def plot(sdf, scale=scale, center=0, fmt=".0f"):
    sns.heatmap(sdf, annot=True, fmt=fmt,
                vmin=scale[0], vmax=scale[1], center=center,
                cmap="RdBu_r")


def validity(sd_field, kernels, C=1.0):
    #gx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
    #gy_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.0
    gx_kernel, gy_kernel = kernels

    Gx = correlate2d(sd_field, gx_kernel, mode='same', boundary='symm')
    Gy = correlate2d(sd_field, gy_kernel, mode='same', boundary='symm')

    #plot(Gx, [-1, 1], 0, ".2f")
    # plt.show()

    # return np.max(np.abs(Gx)) < C and np.max(np.abs(Gy))
    return (np.abs(Gx).max(), np.abs(Gy).max())


def index(i, j, d2):
    return i * d2 + j


def unindex(pos, d2):
    return (pos // d2, pos % d2)


def gauss(n, sigma=1):
    r = range(-int(n/2), int(n/2)+1)
    x = np.arange(r[0], r[1], (r[1] - r[0]) / n, dtype=np.float64)

    return 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-x**2/(2*sigma**2))


def optimize_correction(sd_field, gradient_update, C=1.0):
    assert(sd_field.shape == gradient_update.shape)

    #kernels = sobel_kernels(3)
    kernels = simple_kernels()

    plot(sd_field)
    plt.show()

    plot(sd_field + gradient_update)
    plt.show()

    print("sd_field validity: {}".format(validity(sd_field, kernels)))
    print("sd_field + gradient_update validity: {}".format(
        validity(sd_field + gradient_update, kernels)))

    dims = 2
    N = sd_field.shape[0] * sd_field.shape[1]

    Q = identity(N, dtype=np.float64, format='csc')

    threshold = 2.0
    near_surface_mask = \
        (abs(sd_field + gradient_update).ravel()) > threshold

    Q[near_surface_mask] = 0.0

    Sx = correlate2d(sd_field, kernels[0], mode='same', boundary='symm')
    Sy = correlate2d(sd_field, kernels[1], mode='same', boundary='symm')

    # we can write +- constraint in one,
    # so it's not 2x dims and we also have the two L1 constraints
    E_rows = (dims + 2) * N
    E_cols = N

    def clamp(a, b): return (
        min(max(a, 0), sd_field.shape[0] - 1),
        min(max(b, 0), sd_field.shape[1] - 1),
    )

    dc = {}

    def up(coord, val):
        if val != 0:
            if coord in dc.keys():
                dc[coord] += val
            else:
                dc[coord] = val

    hw = (kernels[0].shape[0] // 2, kernels[0].shape[1] // 2)
    for i in range(sd_field.shape[0]):
        for j in range(sd_field.shape[1]):
            base_index = index(i, j, sd_field.shape[1])

            kern_index = 0
            for si in range(i - hw[0], i + hw[1] + 1):
                for sj in range(j - hw[0], j + hw[1] + 1):
                    a_si, a_sj = clamp(si, sj)
                    sb_index = index(a_si, a_sj, sd_field.shape[1])

                    up((base_index, sb_index),
                       kernels[0].ravel()[kern_index])
                    up((N + base_index, sb_index),
                       kernels[1].ravel()[kern_index])

                    up((2 * N + base_index, sb_index),
                       kernels[0].ravel()[kern_index])
                    up((2 * N + base_index, sb_index),
                       kernels[1].ravel()[kern_index])
                    up((3 * N + base_index, sb_index),
                       kernels[0].ravel()[kern_index])
                    up((3 * N + base_index, sb_index),
                       kernels[1].ravel()[kern_index])

                    kern_index += 1
    row_ind = []
    col_ind = []
    data = []

    for (i, j), val in dc.items():
        row_ind.append(i)
        col_ind.append(j)
        data.append(val)
    E = coo_matrix((data, (row_ind, col_ind)))

    # |d(d)/dx| <= 1, |d(d)/dy| <= C
    # recall that d = G + x
    # |d(G + x)/dx| <= C
    #
    # we approximate this as
    # |Sobel_x(G + x)| = |Sobel_x(G) + Sobel_x(x)| <= C
    #
    # -C - Sobel_x(G) <= Sobel_x(x) <= C - Sobel_x(G)
    # -C - Sobel_y(G) <= Sobel_y(x) <= C - Sobel_y(G)

    u = np.zeros((E_rows,), dtype=np.float64)
    u[:N] = C - Sx.ravel()
    u[N:2*N] = C - Sy.ravel()

    l = np.zeros((E_rows), dtype=np.float64)
    l[:N] = -C - Sx.ravel()
    l[N:2*N] = -C - Sy.ravel()

    # we also have the L1 norm constraint
    # |d(d)/dx + d(d)/dy| >= 1
    # Using the same logic as above
    # |Sobel_x(G + x) + Sobel_y(G + x)| =
    #     |Sobel_x(G) + Sobel_x(x) + Sobel_y(G) + Sobel_y(x)| >= 1
    # -1 >= Sobel_x(G) + Sobel_x(x) + Sobel_y(G) + Sobel_y(x) >= 1
    # -1 - Sobel_x(G) - Sobel_y(G) >= Sobel_x(x) + Sobel_y(x)
    #                                    >= 1 - Sobel_x(G) - Sobel_y(G)

    eps = 1e-5
    u[2*N:3*N] = (1 - eps) - Sx.ravel() - Sy.ravel()
    l[3*N:4*N] = -(1 - eps) - Sx.ravel() - Sy.ravel()

    # these are harder to explain -- look at diagram
    l[2*N:3*N] = -(2 ** 0.5 + 0.1) - Sx.ravel() - Sy.ravel()
    u[3*N:4*N] = (2 ** 0.5 + 0.1) - Sx.ravel() - Sy.ravel()

    # for i in range(N):
    #    print("{} <= Sobel_x <= {}".format(l[i], u[i]))
    #    print("{} <= Sobel_y <= {}".format(l[2 * i], u[2 * i]))

    prob = osqp.OSQP()

    print("solving problem...")

    q = np.zeros(N)

    # interesting idea, but just means
    # that we tend to reverse the gradient
    q = -gradient_update.ravel()
    q[near_surface_mask] = 0.0

    prob.setup(Q, q, E, l, u)

    res = prob.solve()
    X = res.x.reshape(sd_field.shape)
    print(X)

    print("Final validity: {}".format(validity(sd_field + X, kernels)))

    plot(sd_field + X)
    plt.show()


if __name__ == '__main__':
    dims = (32, 32)

    def sphere(p, off, radius):
        return np.linalg.norm(p - off) - radius
    gradient = np.zeros(dims)
    sdf = gen_sdf(sphere, dims,
                  np.array((dims[0] / 2, dims[1] / 2)),
                  int(dims[0] * 0.3))

    #gradient[18:22, 18:22] = 20.0
    gradient += (np.random.random((dims[0], dims[1])) - 0.5) * 1.0

    optimize_correction(sdf, gradient)

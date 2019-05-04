#!/usr/bin/env python3

import osqp
from scipy.sparse import identity, csc_matrix, coo_matrix
from scipy.signal import correlate2d
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


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


def validity(sd_field, kernels, C=1.0):
    #gx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
    #gy_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.0
    gx_kernel, gy_kernel = kernels

    Gx = correlate2d(sd_field, gx_kernel, mode='same', boundary='symm')
    Gy = correlate2d(sd_field, gy_kernel, mode='same', boundary='symm')

    # return np.max(np.abs(Gx)) < C and np.max(np.abs(Gy))
    return (np.abs(Gx).max(), np.abs(Gy).max())


def index(i, j, d2):
    return i * d2 + j


def unindex(pos, d2):
    return (pos // d2, pos % d2)


scale = (-4, 19)


def plot(sdf):
    sns.heatmap(sdf, annot=True, fmt=".0f",
                vmin=scale[0], vmax=scale[1], center=0,
                cmap="RdBu_r")


def optimize_correction(sd_field, gradient_update, C=1.0):
    assert(sd_field.shape == gradient_update.shape)

    G = sd_field + gradient_update

    #kernels = sobel_kernels(3)
    kernels = simple_kernels()

    plot(sd_field)
    plt.show()

    plot(G)
    plt.show()

    print("sd_field validity: {}".format(validity(sd_field, kernels)))
    print("G validity: {}".format(validity(G, kernels)))

    dims = 2
    N = sd_field.shape[0] * sd_field.shape[1]

    Q = identity(N, dtype=np.float64, format='csc')

    Gx = correlate2d(G, kernels[0], mode='same', boundary='symm')
    Gy = correlate2d(G, kernels[1], mode='same', boundary='symm')

    # we can write +- constraint in one,
    # so it's not 2x dims
    E_rows = dims * N
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
    u[:N] = C - Gx.ravel()
    u[N:2*N] = C - Gy.ravel()

    l = np.zeros((E_rows), dtype=np.float64)
    l[:N] = -C - Gx.ravel()
    l[N:2*N] = -C - Gy.ravel()

    # for i in range(N):
    #    print("{} <= Sobel_x <= {}".format(l[i], u[i]))
    #    print("{} <= Sobel_y <= {}".format(l[2 * i], u[2 * i]))

    prob = osqp.OSQP()

    print("solving problem...")

    q = np.zeros(N)

    # interesting idea, but just means
    # that we tend to reverse the gradient
    #q = -gradient_update.ravel()

    prob.setup(Q, q, E, l, u)

    res = prob.solve()
    X = res.x.reshape(sd_field.shape)
    print(X)

    print("Final validity: {}".format(validity(G + X, kernels)))

    plot(G + X)
    plt.show()


if __name__ == '__main__':
    dims = (32, 32)

    def sphere(p, off, radius):
        return np.linalg.norm(p - off) - radius
    gradient = np.zeros(dims)
    sdf = gen_sdf(sphere, dims,
                  np.array((dims[0] / 2, dims[1] / 2)),
                  int(dims[0] * 0.3))

    gradient[18:22, 18:22] = 20.0
    #gradient -= np.random.random((dims[0], dims[1])) * 10.0

    optimize_correction(sdf, gradient)

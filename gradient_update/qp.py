#!/usr/bin/env python3

import osqp
from scipy.sparse import identity, csc_matrix
from scipy.signal import correlate2d
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


def gen_sdf(f, dims):
    sdf = np.zeros(dims, dtype=np.float64)
    for i in range(sdf.shape[0]):
        for j in range(sdf.shape[1]):
            sdf[i, j] = f(np.array([i, j], dtype=np.float64))
    return sdf


def correlate(f1, f2):
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


def validity(sd_field, C=1.0):
    gx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
    gy_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.0

    Gx = correlate2d(sd_field, gx_kernel, mode='same', boundary='symm')
    Gy = correlate2d(sd_field, gy_kernel, mode='same', boundary='symm')

    # return np.max(np.abs(Gx)) < C and np.max(np.abs(Gy))
    return (np.abs(Gx).max(), np.abs(Gy).max())


def optimize_correction(sd_field, gradient_update, C=1):
    assert(sd_field.shape == gradient_update.shape)

    G = sd_field + gradient_update

    #sns.heatmap(G, annot=True, fmt=".1f")
    sns.heatmap(G, annot=False, fmt=".1f")
    plt.show()

    print("sd_field validity: {}".format(validity(sd_field)))
    print("G validity: {}".format(validity(G)))

    dims = 2
    N = sd_field.shape[0] * sd_field.shape[1]

    Q = identity(N, dtype=np.float64, format='csc')

    gx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
    gy_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.0
    Gx = correlate2d(G, gx_kernel, mode='same', boundary='symm')
    Gy = correlate2d(G, gy_kernel, mode='same', boundary='symm')

    # we can write +- constraint in one,
    # so it's not 2x dims
    E_rows = dims * N
    E_cols = N

    def clamp(a, b): return (
        min(max(a, 0), sd_field.shape[0] - 1),
        min(max(b, 0), sd_field.shape[1] - 1),
    )

    E = csc_matrix((E_rows, E_cols), dtype=np.float64)
    for i in range(sd_field.shape[0]):
        for j in range(sd_field.shape[1]):
            base_index = j * sd_field.shape[0] + i

            kern_index = 0
            for si in range(i - 1, i + 2):
                for sj in range(j - 1, j + 2):
                    a_si, a_sj = clamp(si, sj)
                    sb_index = a_sj * sd_field.shape[0] + a_si

                    '''print("{} -- {} || (ai, aj) {}: {} | {} --> {}".format(
                        base_index,
                        (i, j),
                        (a_si, a_sj),
                        sb_index,
                        kern_index,
                        gx_kernel.ravel()[kern_index]))'''

                    E[base_index, sb_index] += \
                        gx_kernel.ravel()[kern_index]
                    E[N + base_index, sb_index] += \
                        gy_kernel.ravel()[kern_index]

                    kern_index += 1

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

    prob = osqp.OSQP()

    print("solving problem...")
    q = np.zeros(N)
    prob.setup(Q, q, E, l, u)

    res = prob.solve()
    X = res.x.reshape(sd_field.shape)
    print(X)

    print("Final validity: {}".format(validity(G + X)))

    #sns.heatmap(G + X, annot=True, fmt=".1f")
    sns.heatmap(G + X, annot=False, fmt=".1f")
    plt.show()


if __name__ == '__main__':
    dims = (100, 100)
    def sphere(p): return np.linalg.norm(p) - int(dims[0] * 0.3)
    gradient = np.zeros(dims)
    sdf = gen_sdf(sphere, dims)

    for i in range(int(0.6 * dims[0]), int(0.7 * dims[1])):
        for j in range(int(0.6 * dims[0]), int(0.7 * dims[1])):
            gradient[i,  j] = -10.0

    optimize_correction(sdf, gradient)

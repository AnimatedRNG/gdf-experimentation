#!/usr/bin/env python3

import numpy as np
from mesh import write_sdf_to_file
import torch
from renderable import Function


def p_clamp(p):
    #min_clamp = torch.min(p, torch.tensor([40.0], dtype=torch.float32))
    #both_clamp = torch.max(min_clamp, torch.tensor([1.0], dtype=torch.float32))
    # return both_clamp
    min_clamp = p.min(Function(40.0))
    both_clamp = min_clamp.max(Function(1.0))
    return both_clamp


def gdf_raw(tree, p):
    '''Returns a Generalized Distance Function'''
    # return lambda v: sum(
    #    gdf_raw(t[0], p_clamp(t[1]))(v) if isinstance(
    #        t, tuple) else torch.abs(torch.dot(t, v)) ** p
    #    for t in tree) ** (1 / p)

    return lambda v: sum(
        gdf_raw(t[0], p_clamp(t[1]))(v) if isinstance(
            t, tuple) else t.dot(v).abs().pow(p)
        for t in tree).pow(Function(1.0) / p)


'''def gdf(tree, p,
        transpose=torch.tensor(np.array([-0.5, -0.5, -0.5],
                                        dtype=np.float32)), signed_offset=-0.2):
    #Cleaner interface that transposes and offsets the output
    return lambda v: gdf_raw(tree, p_clamp(p))(v + transpose) + signed_offset'''


def gdf(tree, p,
        transpose=Function([-0.5, -0.5, -0.5]), signed_offset=Function(-0.2)):
    '''Cleaner interface that transposes and offsets the output'''
    return lambda v: gdf_raw(tree, p_clamp(p))(v + transpose) + signed_offset


def sphere_sdf(radius=0.1, origin=np.array([0.5, 0.5, 0.5])):
    '''Example SDF'''
    return lambda a: np.linalg.norm(a - origin) - radius


def generate_volume(sdf, resolution=32):
    x_grid = np.linspace(0, 1, resolution)
    y_grid = np.linspace(0, 1, resolution)
    z_grid = np.linspace(0, 1, resolution)
    xv, yv, zv = np.meshgrid(x_grid, y_grid, z_grid)

    sdf_buffer = np.zeros(
        (resolution, resolution, resolution), dtype=np.float32)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                position = torch.tensor(np.array(
                    [xv[i, j, k], yv[i, j, k], zv[i, j, k]], dtype=np.float32))
                #dist = sdf(position).detach().numpy()
                dist = sdf.generate_model({'pos': position}).detach().numpy()
                sdf_buffer[j, resolution - i - 1, k] = dist
    return (sdf_buffer.flatten(), np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32))


if __name__ == '__main__':
    resolution = 32

    def vec3(a, b, c): return torch.tensor(np.array((a, b, c), np.float32))

    g = gdf([vec3(0.577, 0.577, 0.577),
             vec3(-0.577, 0.577, 0.577),
             vec3(0.577, -0.577, 0.577),
             vec3(0.577, 0.577, -0.577)],
            torch.tensor([9.0], dtype=torch.float32))

    volume, header = generate_volume(g, resolution)
    write_sdf_to_file("model.sdf", header, volume, resolution)

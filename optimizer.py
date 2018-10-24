#!/usr/bin/env python3

import torch
from sys import argv
import threading
import numpy as np

from numpy import meshgrid
from model import gdf, generate_volume
from mesh import mesh_to_sdf, load_sdf_from_file, write_sdf_to_file
import visualizer

import pymesh


def decompose_tree(tree):
    vs, p = tree
    flat = [p]
    for t in vs:
        if isinstance(t, list) or isinstance(t, tuple):
            flat.extend(decompose_tree(t))
        else:
            flat.append(t)
    return flat


def optimize_to_point_cloud(tree, pointcloud):
    ps = decompose_tree(tree)
    vs, p = tree
    model = gdf(vs, p)
    optimizer = torch.optim.Adam(ps)
    # LBFGS
    loss_fn = torch.nn.MSELoss(size_average=False)
    for epoch in range(10):
        for row_index in range(pointcloud.shape[0]):
            xyz = pointcloud[row_index][:3]
            d = pointcloud[row_index][3]
            prediction = model(torch.tensor(xyz, dtype=torch.float32))
            expected = torch.tensor(d)
            loss = loss_fn(prediction, expected)

            #print("Prediction {}; expected: {}".format(prediction, expected))
            # print(ps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(epoch, loss.item())
    return model


def sdf_to_point_cloud(sdf, header, just_level_set=True):
    x_grid = np.linspace(0, 1, 32)
    y_grid = np.linspace(0, 1, 32)
    z_grid = np.linspace(0, 1, 32)
    xv, yv, zv = meshgrid(x_grid, y_grid, z_grid)
    coords = np.stack(
        (xv.ravel(), yv.ravel(), zv.ravel(), sdf.ravel()), axis=1)
    if just_level_set:
        coords = coords[np.abs(coords[:, 3]) < 1e-2]

    return coords


def build_tree(depth, breadth):
    if depth == 0:
        return ([torch.rand(3, dtype=torch.float32, requires_grad=True) for i in range(breadth)],
                torch.tensor(np.array([3], np.float32), requires_grad=True))
    else:
        return ([build_tree(depth - 1, breadth) for i in range(breadth)],
                torch.tensor(np.array([3], np.float32), requires_grad=True))


def main():
    header, sdf = load_sdf_from_file(argv[1])
    pc = sdf_to_point_cloud(sdf, header)

    def vec3(a, b, c): return torch.tensor(np.array((a, b, c), np.float32),
                                           requires_grad=True)
    '''tree = ([vec3(0.577, 0.577, 0.577),
             vec3(-0.577, 0.577, 0.577),
             vec3(0.577, -0.577, 0.577),
             vec3(0.577, 0.577, -0.577)],
            torch.tensor(np.array([3], np.float32), requires_grad=True))'''
    tree = build_tree(2, 4)

    fitted = optimize_to_point_cloud(tree, pc)
    generated_sdf, generated_header = generate_volume(fitted, resolution=32)
    write_sdf_to_file("model.sdf", generated_header, generated_sdf, 32)


if __name__ == '__main__':
    t1 = threading.Thread(target=visualizer.run)
    t1.start()
    t1.join()

    main()

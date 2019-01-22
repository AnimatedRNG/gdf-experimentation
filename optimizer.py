#!/usr/bin/env python3

import torch
from sys import argv
import threading
import numpy as np
import os

from numpy import meshgrid
from model import gdf, generate_volume
from mesh import mesh_to_sdf, load_sdf_from_file, write_sdf_to_file
from renderable import Function
import visualizer

import pymesh


def decompose_tree(tree):
    vs, p = tree
    flat = [p.model]
    for t in vs:
        if isinstance(t, list) or isinstance(t, tuple):
            flat.extend(decompose_tree(t))
        else:
            flat.append(t.model)
    return flat


def print_tree(tree, depth=0):
    vs, p = tree
    flat = [p]
    print("P at depth {} is {}; grad: {}".format(depth, p.model, p.model.grad))
    for t in vs:
        if isinstance(t, list) or isinstance(t, tuple):
            print_tree(t, depth+1)
        else:
            print("Row {} is {}; grad: {}".format(
                depth, t.model, t.model.grad))


def optimize_to_sdf(tree, sdf):
    ps = decompose_tree(tree)
    vs, p = tree
    model = gdf(vs, p)(Function('pos', 3))
    optimizer = torch.optim.Adam(ps)

    loss_fn = torch.nn.MSELoss(size_average=False)
    for epoch in range(100):

        #print_tree(tree, 0)

        for i in range(1000):
            xyz = torch.randn(3)
            d = sdf.generate_model(
                {'pos': torch.tensor(xyz, dtype=torch.float32)})

            prediction = model.generate_model(
                {'pos': torch.tensor(xyz, dtype=torch.float32)})
            expected = torch.tensor(d).unsqueeze(0)
            # print("{}: prediction: {}, expected: {}".format(
            #    xyz, prediction, expected))
            # print("3")
            # print("4")
            loss = loss_fn(prediction, expected)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            model.update()

            if loss < 1e-14:
                break
        print(loss)

        with open("model.glsl", "w") as f:
            f.write(model.generate_shader())

    return model


def optimize_to_point_cloud(tree, pointcloud):
    ps = decompose_tree(tree)
    vs, p = tree
    model = gdf(vs, p)(Function('pos', 3))
    optimizer = torch.optim.Adam(ps)
    # LBFGS
    loss_fn = torch.nn.MSELoss(size_average=False)
    for epoch in range(100):
        for row_index in range(pointcloud.shape[0]):

            xyz = pointcloud[row_index][:3]
            d = pointcloud[row_index][3]
            # prediction = model(torch.tensor(xyz, dtype=torch.float32))
            # prediction = model(Function(xyz.tolist()))
            prediction = model.generate_model(
                {'pos': torch.tensor(xyz, dtype=torch.float32)})
            expected = torch.tensor(d).unsqueeze(0)
            loss = loss_fn(prediction, expected)

            #print("Prediction {}; expected: {}".format(prediction, expected))
            # print(ps)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            model.update()

        with open("model.glsl", "w") as f:
            f.write(model.generate_shader())

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
        return ([Function((np.random.rand(3) - 0.5).tolist(), 'requires_grad') for i in range(breadth)],
                Function(3.0, 'requires_grad'))
    else:
        return ([build_tree(depth - 1, breadth) for i in range(breadth)],
                Function(3.0, 'requires_grad'))


def main():
    header, sdf = load_sdf_from_file(argv[1])
    # visualizer.visualize_sdf_as_mesh(sdf)
    # visualizer.start_visualization()
    pc = sdf_to_point_cloud(sdf, header)
    '''def vec3(a, b, c): return Function((a, b, c), 'requires_grad')
    tree = ([vec3(0.577, 0.577, 0.577),
             vec3(-0.577, 0.577, 0.577),
             vec3(0.577, -0.577, 0.577),
             vec3(0.577, 0.577, -0.577)],
            Function([9], 'requires_grad'))'''

    torch.backends.cudnn.deterministic = True
    torch.backends.mkl.deterministic = True
    tree = build_tree(2, 4)

    def vec3(a, b, c): return Function((a, b, c), 'requires_grad')
    sdf = ([vec3(0.577, 0.577, 0.577),
            vec3(-0.577, 0.577, 0.577),
            vec3(0.577, -0.577, 0.577),
            vec3(0.577, 0.577, -0.577)],
           Function([9], 'requires_grad'))

    #fitted = optimize_to_point_cloud(tree, pc)
    fitted = optimize_to_sdf(tree, gdf(*sdf)(Function('pos', 3)))
    # generated_sdf, generated_header = generate_volume(fitted, resolution=32)
    # write_sdf_to_file("model.sdf", generated_header, generated_sdf, 32)


if __name__ == '__main__':
    main()

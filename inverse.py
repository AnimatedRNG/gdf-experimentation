#!/usr/bin/env python3

import tracer
import cv2
import torch
import numpy as np

from tracer import ImageRenderer, forward_pass, Tiling


def tiled_iterator(width, height, x_tile, y_tile, f_pass):
    for x in range(0, width, x_tile):
        for y in range(0, height, y_tile):
            tiling = Tiling(width, height, x, y, x_tile, y_tile)
            yield (tiling, f_pass(tiling))


def inverse(m1,
            m2,
            renderer,
            num_epochs=300,
            width=512,
            height=384,
            tracing_iterations=220,
            EPS=1e-6):
    current_model, current_params = m1
    target_model, target_params = m2

    loss_fn = torch.nn.MSELoss(size_average=False)
    target_output, _ = forward_pass(target_model, renderer,
                                    Tiling(width, height, 0, 0, width, height),
                                    tracing_iterations,
                                    EPS, True, "target")

    def forward_pass_model(tiling):
        return forward_pass(current_model, renderer,
                            tiling, tracing_iterations,
                            EPS, True, "model")

    optimizer = torch.optim.Adam([current_params])
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(num_epochs):
            for tiling, (tile_output, _) in tiled_iterator(width, height,
                                                           64, 64,
                                                           forward_pass_model):
                _, _, x_off, y_off, x_tile, y_tile = tiling
                loss = loss_fn(tile_output,
                               target_output[y_off:y_off+y_tile,
                                             x_off:x_off+x_tile])
                print(tile_output)
                print(target_output)
                print("Loss: {}".format(loss))

                optimizer.zero_grad()
                loss.backward(retain_graph=False)
                optimizer.step()
                current_model.update()

                if loss < 1e-14:
                    break


if __name__ == '__main__':
    from tracer import create_analytic, sphere_model
    import scene

    r1 = torch.tensor((3.0,), dtype=torch.float64, requires_grad=True)
    r2 = torch.tensor((6.0,), dtype=torch.float64, requires_grad=False)
    def sphere_1(position): return sphere_model(position, r1)
    def sphere_2(position): return sphere_model(position, r2)
    test_scene = scene.load('test.hdf5')

    renderer = ImageRenderer()
    inverse((create_analytic(test_scene, sphere_1), r1),
            (create_analytic(test_scene, sphere_2), r2),
            renderer)

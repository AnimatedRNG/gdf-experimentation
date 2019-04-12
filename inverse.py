#!/usr/bin/env python3

import tracer
import cv2
import torch
import numpy as np

from tracer import ImageRenderer, forward_pass, Tiling


def inverse(m1,
            m2,
            renderer,
            num_epochs=300,
            width=512,
            height=384,
            tracing_iterations=100,
            EPS=1e-6):
    current_model, current_params = m1
    target_model, target_params = m2

    loss_fn = torch.nn.MSELoss(size_average=False)
    target_output, _ = forward_pass(target_model, renderer,
                                    Tiling(width, height, 0, 0, width, height),
                                    tracing_iterations,
                                    EPS, False, "target")
    optimizer = torch.optim.Adam([current_params])
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(num_epochs):
            model_output, _ = forward_pass(current_model, renderer,
                                           width, height, tracing_iterations,
                                           EPS, True, "model")
            loss = loss_fn(model_output, target_output)
            print(model_output)
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

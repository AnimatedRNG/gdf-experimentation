#!/usr/bin/env python3

import torch
import numpy as np

device = torch.device('cuda:0')


def projection(proj_matrix, view_matrix, width, height):
    rays = torch.zeros((2, width, height, 3), dtype=torch.float32)
    inv = torch.inverse(proj_matrix @ view_matrix)
    origin = (torch.inverse(view_matrix) @ torch.tensor(
        (0.0, 0.0, 0.0, 1.0)))[:3]
    near = 0.1
    grid = torch.meshgrid(torch.linspace(-1.0, 1.0, width),
                          torch.linspace(-1.0, 1.0, height))
    clip_space = torch.stack(
        (grid[0], grid[1], torch.ones((width, height)), torch.ones((width, height))), dim=-1)
    tmp = torch.matmul(inv, clip_space.view(width, height, 4, 1)).squeeze()
    tmp /= tmp[:, :, 3:]
    tmp = tmp[:, :, :3]
    tmp -= torch.tensor((origin[0], origin[1], origin[2]),
                        dtype=torch.float32)
    ray_vec = tmp / torch.norm(tmp, p=2, dim=2).unsqueeze(-1)
    rays[0, :, :] = origin + ray_vec * near
    rays[1, :, :] = ray_vec

    return rays


def to_render_dist(dist):
    # TODO: Proper clamping to grid cell boundaries
    return torch.min(dist, torch.ones(dist.shape, dtype=dist.dtype))


def sdf_model(position):
    return (torch.norm(position, p=2, dim=2) - 3.0).unsqueeze(-1)


def sdf_iteration(ray_matrix, model):
    '''
    ray_matrix: [2, ALL_RAYS, RAY_DIMS]
    '''
    xyz = ray_matrix[0, :, :, :]
    vec = ray_matrix[1, :, :, :]
    # d = model.generate_model(
    #    {'pos': torch.tensor(xyz, dtype=torch.float32, device=device)})
    d = sdf_model(xyz)
    return (xyz + to_render_dist(d) * vec, d)


def hit_mask(d):
    hit = np.zeros_like(d)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            if d[i, j] < 1e-1:
                hit[i, j] = 1.0


def forward_pass(grid_sdf, width=500, height=500):
    import cv2
    projection_matrix = grid_sdf.perspective
    view_matrix = grid_sdf.view
    rays = projection(projection_matrix, view_matrix, width, height)
    for i in range(200):
        print("traced iteration {}".format(i))
        pos_2, d = sdf_iteration(rays, None)
        #print(pos_2[50:53, 50:53])
        #print(d[50:53, 50:53])
        rays[0] = pos_2
    #print(torch.norm(rays[0, :, :, :3], p=2, dim=2) / 100.0)
    cv2.imshow('rays', rays.numpy()[1, :, :, :3])
    cv2.imshow('d', d.numpy() / 30.0)
    #cv2.imshow('hit', hit)
    cv2.waitKey(0)


if __name__ == '__main__':
    import scene
    forward_pass(scene.load('test.hdf5'))

#!/usr/bin/env python3

import torch
import numpy as np
import cv2

device = torch.device('cuda:0')


class ImageRenderer:
    def __init__(self, _width=1600, _height=800):
        self.width = _width
        self.height = _height
        self.current_position = [0, 0]
        self.ideal = (350, 350)
        self.vbuf = 60

    def show(self, name, img):
        cv2.imshow(name, cv2.resize(img, self.ideal,
                                    interpolation=cv2.INTER_AREA))
        cv2.moveWindow(name, *(self.current_position))
        if self.width - self.current_position[0] < 2 * (self.ideal[0]):
            self.current_position[0] = 0
            self.current_position[1] += self.ideal[1] + self.vbuf
        else:
            self.current_position[0] += self.ideal[0]

    def render_all_images(self, pause=1):
        cv2.waitKey(pause)
        self.current_position = [0, 0]


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

    return rays, origin


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


def h(positions, index, sdf, EPS=1e-6):
    forward = positions.clone()
    backward = positions.clone()
    forward[:, :, index] += EPS
    backward[:, :, index] -= EPS
    top = torch.stack((sdf(backward), sdf(positions), sdf(forward)), dim=-1)
    bottom = torch.tensor((1.0, 2.0, 1.0)).view(1, -1, 1)

    dim_0 = positions.shape[0]
    dim_1 = positions.shape[1]
    return torch.bmm(top.view(dim_0 * dim_1, 1, 3),
                     bottom.expand(dim_0 * dim_1, 3, 1)).view(dim_0, dim_1)


def h_p(positions, index, sdf, EPS=1e-6):
    forward = positions.clone()
    backward = positions.clone()
    forward[:, :, index] += EPS
    backward[:, :, index] -= EPS
    top = torch.stack((sdf(backward), sdf(forward)), dim=-1)
    bottom = torch.tensor((1.0, -1.0)).view(1, -1, 1)

    dim_0 = positions.shape[0]
    dim_1 = positions.shape[1]
    return torch.bmm(top.view(dim_0 * dim_1, 1, 2),
                     bottom.expand(dim_0 * dim_1, 2, 1)).view(dim_0, dim_1)


def sobel(positions, m):
    h_x = h_p(positions, 0, m) * h(positions, 1, m) * h(positions, 2, m)
    h_y = h_p(positions, 1, m) * h(positions, 2, m) * h(positions, 0, m)
    h_z = h_p(positions, 2, m) * h(positions, 0, m) * h(positions, 1, m)

    h_all = torch.stack((h_x, h_y, h_z), -1)
    return (-1.0 * h_all) / torch.norm(h_all, p=2, dim=-1).unsqueeze(-1)


def light_source(light_color,
                 positions,
                 light_position,
                 normals,
                 kd=0.7,
                 ks=0.3,
                 ka=100.0):
    dim_0 = positions.shape[0]
    dim_1 = positions.shape[1]

    light_vec = light_position - positions
    light_vec /= torch.norm(light_vec, p=2, dim=-1).unsqueeze(-1)
    ray_position = positions + light_vec
    diffuse = kd * torch.clamp(
        torch.bmm(normals.view(dim_0 * dim_1, 1, 3),
                  light_vec.view(dim_0 * dim_1, 3, 1),
                  ).view(dim_0, dim_1, 1) * light_color, 0.0, 1.0)
    # TODO: Figure out specular
    return diffuse


def shade(rays, origin, normals, EPS=1e-6):
    top_light_color = torch.tensor((0.9, 0.9, 0.0))
    self_light_color = torch.tensor((0.1, 0.0, 0.1))

    top_light_pos = torch.tensor((10.0, 30.0, 0.0))
    self_light_pos = origin

    top_light = light_source(
        top_light_color, rays[0], top_light_pos, normals)
    self_light = light_source(
        self_light_color, rays[0], self_light_pos, normals)
    return top_light + self_light


def normal_pdf(x, sigma=1e-5, mean=0.0):
    return (1.0 / np.sqrt(2.0 * np.pi * sigma * sigma)) * \
        torch.exp((x - mean) ** 2 / (-2.0 * sigma * sigma))


def forward_pass(grid_sdf, width=500, height=500, EPS=1e-6):
    projection_matrix = grid_sdf.perspective
    view_matrix = grid_sdf.view
    rays, origin = projection(projection_matrix, view_matrix, width, height)
    energy = torch.ones((width, height, 1), dtype=torch.float32)
    total_intensity = torch.zeros((width, height, 3), dtype=torch.float32)
    energy_denom = torch.ones((width, height, 1), dtype=torch.float32)

    num = torch.zeros((width, height, 3), dtype=torch.float32)
    denom = torch.zeros((width, height, 1), dtype=torch.float32)

    renderer = ImageRenderer()

    for i in range(200):
        print("traced iteration {}".format(i))

        pos_2, d = sdf_iteration(rays, None)
        normals = sobel(rays[0] - rays[1] * EPS, sdf_model)
        g_d = normal_pdf(d)

        intensity = shade(rays, origin, normals)
        total_intensity += energy * g_d * intensity
        energy_denom += energy * g_d
        energy = torch.clamp(energy - g_d, 0.0, 1.0)

        num += g_d * intensity
        denom += g_d

        rays[0] = pos_2

        renderer.show('rays', rays.numpy()[1])
        renderer.show('normals', normals.numpy())
        renderer.show('d', d.numpy() / 30.0)
        renderer.show('g_d', g_d.numpy())
        renderer.show('energy', energy.numpy())
        renderer.show('intensity', intensity.numpy())
        renderer.show('shaded_old', (num / denom).numpy())
        renderer.show('energy_shaded',
                      (total_intensity / energy_denom).numpy())

        renderer.render_all_images(100)


if __name__ == '__main__':
    import scene
    forward_pass(scene.load('test.hdf5'))

#!/usr/bin/env python3

import torch
import numpy as np
import cv2
import imageio
import os
from collections import namedtuple

device = torch.device('cuda:0')


class ImageRenderer:
    def __init__(self, _width=1600, _height=800):
        self.width = _width
        self.height = _height
        self.current_position = [0, 0]
        #self.ideal = (256, 192)
        self.ideal = (384, 288)
        #self.ideal = (256, 256)
        self.vbuf = 60
        self.recordings = {}

    def record(self, name, img):
        clipped = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
        clipped = cv2.cvtColor(clipped, cv2.COLOR_RGB2BGR)
        if name in self.recordings.keys():
            self.recordings[name].append(clipped)
        else:
            self.recordings[name] = [clipped]

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

    def save_gifs(self):
        if not os.path.isdir("./screenshots"):
            os.mkdir("./screenshots")
        for gif_name, gif_images in self.recordings.items():
            filename = os.path.join("screenshots", gif_name + ".gif")
            imageio.mimwrite(filename, gif_images, 'GIF')
            try:
                os.popen(
                    "gifsicle -O3 {filename} -o {filename}".format(filename=filename))
            except Exception as e:
                print("unable to optimize gif {}: {}".format(filename, e))


def sanitize(t, set_value=0):
    n = t.clone()
    n[t != t] = set_value
    return n


def projection(proj_matrix, view_matrix, tiling):
    width, height, x_off, y_off, x_tile, y_tile = tiling

    rays = torch.zeros((2, height, width, 3), dtype=torch.float64)
    inv = torch.inverse(proj_matrix @ view_matrix)
    origin = (torch.inverse(view_matrix) @ torch.tensor(
        (0.0, 0.0, 0.0, 1.0), dtype=torch.float64))[:3]
    near = 0.1
    grid = torch.meshgrid(torch.linspace(-1.0, 1.0, height, dtype=torch.float64),
                          torch.linspace(-1.0, 1.0, width, dtype=torch.float64))
    clip_space = torch.stack(
        (grid[0], grid[1],
         torch.ones((height, width), dtype=torch.float64),
         torch.ones((height, width), dtype=torch.float64)), dim=-1)
    tmp = torch.matmul(inv, clip_space.view(height, width, 4, 1)).squeeze()
    tmp = tmp / tmp[:, :, 3:]
    tmp = tmp[:, :, :3]
    tmp = tmp - torch.tensor((origin[0], origin[1], origin[2]),
                             dtype=torch.float64)
    ray_vec = tmp / torch.norm(tmp, p=2, dim=2).unsqueeze(-1)
    rays[0, :, :] = origin + ray_vec * near
    rays[1, :, :] = ray_vec

    return rays[:, y_off:y_off+y_tile, x_off:x_off+x_tile, :], origin


def to_render_dist(dist):
    # TODO: Proper clamping to grid cell boundaries
    # return dist
    # return torch.min(dist, torch.ones(dist.shape, dtype=dist.dtype))
    # return torch.ones_like(dist, dtype=torch.float64) / 3
    step = torch.ones_like(dist, dtype=torch.float64)
    interpolant = (1.0 - torch.clamp(torch.abs(dist), 0.0, 1.0)) * 90
    return step / (10 + interpolant)


def vmax(v):
    return torch.max(torch.max(v[:, :, 0], v[:, :, 1]), v[:, :, 2])


def box_model(position, b=torch.tensor((3.0, 3.0, 3.0), dtype=torch.float64)):
    # Warning: evil discontinuity!
    d = torch.abs(position) - b
    zero_tensor = torch.zeros_like(d)
    return (torch.norm(torch.max(d, zero_tensor), p=2, dim=2) +
            vmax(torch.min(d, zero_tensor))).unsqueeze(-1)


def sphere_model(position,
                 radius=torch.tensor((3.0,), requires_grad=True)):
    return (torch.norm(position, p=2, dim=2) - radius).unsqueeze(-1)


def grid_sdf_model(position, grid_sdf):
    span = grid_sdf.end - grid_sdf.start
    transposed_position = position - grid_sdf.start
    #print("span: {}".format(span.numpy()))
    #print("start: {}".format(grid_sdf.start))
    box_dist = box_model(transposed_position, span / 2)

    # mask out the rays that are within the bounding box
    box_mask = torch.zeros_like(box_dist)
    box_mask[box_dist < 1.0] = 1.0

    # This is a massive hack!
    if box_dist < 1.0:
        resolution = grid_sdf.data.shape
        coords = torch.floor((transposed_position / span)
                             * torch.tensor(resolution))
        return torch.gather(grid_sdf.data, )
    else:
        return box_dist


def sdf_iteration(ray_matrix, model):
    '''
    ray_matrix: [2, ALL_RAYS, RAY_DIMS]
    '''
    xyz = ray_matrix[0, :, :, :]
    vec = ray_matrix[1, :, :, :]
    # d = model.generate_model(
    #    {'pos': torch.tensor(xyz, dtype=torch.float64, device=device)})
    d = model(xyz)
    return (xyz + to_render_dist(d) * vec, d)


def hit_mask(d):
    hit = np.zeros_like(d)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            if d[i, j] < 1e-1:
                hit[i, j] = 1.0


def h(positions, index, sdf, EPS=1e-5):
    forward = positions.clone()
    backward = positions.clone()
    forward[:, :, index] = forward[:, :, index] + EPS
    backward[:, :, index] = backward[:, :, index] - EPS
    top = torch.stack((sdf(backward), sdf(positions), sdf(forward)), dim=-1)
    bottom = torch.tensor((1.0, 2.0, 1.0), dtype=torch.float64).view(1, -1, 1)

    dim_0 = positions.shape[0]
    dim_1 = positions.shape[1]
    return torch.bmm(top.view(dim_0 * dim_1, 1, 3),
                     bottom.expand(dim_0 * dim_1, 3, 1)).view(dim_0, dim_1)


def h_p(positions, index, sdf, EPS=1e-6):
    forward = positions.clone()
    backward = positions.clone()
    forward[:, :, index] = forward[:, :, index] + EPS
    backward[:, :, index] = backward[:, :, index] - EPS
    top = torch.stack((sdf(backward), sdf(forward)), dim=-1)
    bottom = torch.tensor((1.0, -1.0), dtype=torch.float64).view(1, -1, 1)

    dim_0 = positions.shape[0]
    dim_1 = positions.shape[1]
    return torch.bmm(top.view(dim_0 * dim_1, 1, 2),
                     bottom.expand(dim_0 * dim_1, 2, 1)).view(dim_0, dim_1)


def sobel(positions, m):
    h_x = h_p(positions, 0, m) * h(positions, 1, m) * h(positions, 2, m)
    h_y = h_p(positions, 1, m) * h(positions, 2, m) * h(positions, 0, m)
    h_z = h_p(positions, 2, m) * h(positions, 0, m) * h(positions, 1, m)

    h_all = torch.stack((h_x, h_y, h_z), -1)
    norm = sanitize(torch.norm(h_all, p=2, dim=-1))
    norm[norm == 0.0] = 1e-6
    return (-1.0 * h_all) / (norm).unsqueeze(-1)


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
    light_vec = light_vec / torch.norm(light_vec, p=2, dim=-1).unsqueeze(-1)
    ray_position = positions + light_vec
    diffuse = kd * torch.clamp(
        torch.bmm(normals.view(dim_0 * dim_1, 1, 3),
                  light_vec.view(dim_0 * dim_1, 3, 1),
                  ).view(dim_0, dim_1, 1) * light_color, 0.0, 1.0)
    # TODO: Figure out specular
    return diffuse


def shade(rays, origin, normals, EPS=1e-6):
    top_light_color = torch.tensor((0.6, 0.6, 0.0), dtype=torch.float64)
    self_light_color = torch.tensor((0.4, 0.0, 0.4), dtype=torch.float64)

    top_light_pos = torch.tensor((10.0, 30.0, 0.0), dtype=torch.float64)
    self_light_pos = origin

    top_light = light_source(
        top_light_color, rays[0], top_light_pos, normals)
    self_light = light_source(
        self_light_color, rays[0], self_light_pos, normals)
    return top_light + self_light


def normal_pdf(x, sigma=1e-7, mean=0.0):
    return (1.0 / np.sqrt(2.0 * np.pi * sigma * sigma)) * \
        torch.exp((x - mean) ** 2 / (-2.0 * sigma * sigma))


def normal_pdf_rectified(x, sigma=1e-2, mean=0.0):
    return normal_pdf(torch.relu(x), sigma, mean)


Tiling = namedtuple('Tiling', ['width', 'height',
                               'x_off', 'y_off',
                               'x_tile', 'y_tile'])


def forward_pass(grid_sdf,
                 renderer,
                 tiling=Tiling(
                     width=512,
                     height=384,
                     x_off=0,
                     y_off=0,
                     x_tile=512,
                     y_tile=384
                 ),
                 iterations=300, EPS=1e-6, verbose=True, prefix=""):
    _, _, x_off, y_off, x_tile, y_tile = tiling
    projection_matrix = grid_sdf.perspective
    view_matrix = grid_sdf.view
    rays, origin = projection(projection_matrix, view_matrix, tiling)
    rays = rays.unsqueeze(0).repeat(iterations + 1, 1, 1, 1, 1)
    opc = torch.zeros((iterations + 1, y_tile, x_tile, 1),
                      dtype=torch.float64)
    c = torch.zeros(
        (iterations + 1, y_tile, x_tile, 3), dtype=torch.float64)
    k = -1.0
    u_s = 1.0

    #num = torch.zeros((height, width, 3), dtype=torch.float64)
    #denom = torch.zeros((height, width, 1), dtype=torch.float64)

    #def model(position): return grid_sdf_model(position, grid_sdf)
    if isinstance(grid_sdf.data, torch.Tensor):
        raise Exception
    else:
        model = grid_sdf.data

    def fmt(st): return "{}_{}".format(prefix, st)
    for i in range(1, iterations + 1):
        print("traced iteration {}".format(i))

        pos_2, d = sdf_iteration(rays[i - 1], model)
        ds = to_render_dist(d)
        normals = sanitize(sobel(rays[i - 1, 0] - rays[i - 1, 1] * EPS, model))
        g_d = sanitize(normal_pdf_rectified(d))

        intensity = sanitize(shade(rays[i - 1], origin, normals))
        opc[i] = opc[i - 1] + g_d * ds
        c[i] = c[i - 1] + (g_d * u_s) * torch.exp(k * opc[i]) * intensity * ds

        #num += g_d * intensity
        #denom += g_d

        rays[i, 0] = pos_2

        if verbose:
            renderer.show(fmt('rays'), rays[i].detach().numpy()[1])
            renderer.show(fmt('normals'), normals.detach().numpy())
            renderer.show(fmt('d'), d.detach().numpy() / 30.0)
            renderer.show(fmt('g_d'), g_d.detach().numpy())
            renderer.show(fmt('opacity'), opc[i].detach().numpy())
            renderer.show(fmt('ds'), ds.detach().numpy())
            renderer.show(fmt('intensity'), intensity.detach().numpy())

            renderer.record(fmt('intensity'),
                            intensity.detach().numpy())

            energy_shaded_img = (c[i]).detach().numpy()
            renderer.record(fmt('opacity_shaded'),
                            energy_shaded_img)
            renderer.show(fmt('opacity_shaded'), energy_shaded_img)

        renderer.render_all_images(1)

    intensity = c[iterations]
    # renderer.save_gifs()
    return (intensity, d)


def create_analytic(sc, model):
    import scene
    params = sc._asdict()
    params['data'] = model
    return scene.GridSDF(**params)


if __name__ == '__main__':
    import scene
    renderer = ImageRenderer()
    test_scene = scene.load('test.hdf5')
    params = test_scene._asdict()
    params['data'] = box_model
    print(params)
    adjusted_scene = scene.GridSDF(**params)
    forward_pass(adjusted_scene, renderer)

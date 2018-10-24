#!/usr/bin/env python3

import pymesh
from sys import argv
from numpy import array, zeros, float32, int32, amin, amax, meshgrid
import numpy as np
from ctypes import c_int, c_float, c_size_t
import ctypes
import matplotlib.pyplot as plt
import struct
from os.path import basename, abspath

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule


class MeshEvaluator:

    def __init__(self, mesh):
        vertices_t = mesh.vertices.T
        faces = mesh.faces
        faces_flattened = faces.flatten()

        self.c_int_p = ctypes.POINTER(ctypes.c_int)
        self.c_float_p = ctypes.POINTER(ctypes.c_float)

        self.vertices_x = array(vertices_t[0], copy=True, dtype=float32)
        self.vertices_y = array(vertices_t[1], copy=True, dtype=float32)
        self.vertices_z = array(vertices_t[2], copy=True, dtype=float32)
        self.faces_view = array(faces_flattened, copy=True, dtype=int32)

        margin = 1e-2
        top, bottom = amax(self.vertices_y) + \
            margin, amin(self.vertices_y) - margin
        left, right = amin(self.vertices_x) - margin, \
            amax(self.vertices_x) + margin
        front, back = amin(self.vertices_z) - margin, \
            amax(self.vertices_z) + margin

        dim_diff = max(right - left, top - bottom, back - front)

        self.vertices_x = (self.vertices_x - left) / dim_diff
        self.vertices_y = (self.vertices_y - bottom) / dim_diff
        self.vertices_z = (self.vertices_z - front) / dim_diff

        self.vertices_x_ctypes = self.to_ctype_float(self.vertices_x)
        self.vertices_y_ctypes = self.to_ctype_float(self.vertices_y)
        self.vertices_z_ctypes = self.to_ctype_float(self.vertices_z)

        self.vertices_x_cuda = drv.In(self.vertices_x)
        self.vertices_y_cuda = drv.In(self.vertices_y)
        self.vertices_z_cuda = drv.In(self.vertices_z)

        self.faces_ctypes = self.to_ctype_int(self.faces_view)
        self.num_faces_ctypes = c_size_t(len(faces))

        self.faces_cuda = drv.In(self.faces_view)
        self.num_faces_cuda = int32(len(faces))

        lib = ctypes.cdll.LoadLibrary(
            './triangle_dist/triangle_distance.so')
        self.fun = lib.closest_intersection
        self.fun.argtypes = [self.c_float_p,
                             self.c_float_p, self.c_float_p, self.c_float_p,
                             self.c_int_p, c_size_t]
        self.fun.restype = c_float

        with open('triangle_dist/triangle_distance.cu', 'r') as f:
            self.mod = SourceModule(f.read(), include_dirs=[
                                    abspath('./triangle_dist')],
                                    no_extern_c=True)
            self.grid_execution = self.mod.get_function('grid_execution')

    def to_ctype_float(self, a):
        return a.ctypes.data_as(self.c_float_p).contents

    def to_ctype_int(self, a):
        return a.ctypes.data_as(self.c_int_p).contents

    def eval_at(self, position):
        position_ctypes = self.to_ctype_float(position)

        return self.fun(position_ctypes,
                        self.vertices_x_ctypes,
                        self.vertices_y_ctypes,
                        self.vertices_z_ctypes,
                        self.faces_ctypes, self.num_faces_ctypes)

    def eval_grid(self, grid_object):
        fig = plt.figure()
        first = True

        xv, yv, zv = grid_object
        assert(xv.shape == yv.shape == zv.shape)
        assert(xv.shape[0] == xv.shape[1] == xv.shape[2])
        resolution = xv.shape[0]

        scalar_field = zeros(
            (resolution, resolution, resolution), dtype=float32)

        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution):
                    position = array(
                        [xv[i, j, k], yv[i, j, k], zv[i, j, k], 0], dtype=float32)
                    result = self.eval_at(position)

                    print("Distance from position {} is {}"
                          .format(position, result))
                    # Ugly hack to do the coordinate transform
                    scalar_field[j, resolution - i - 1, k] = result
            slice_data = scalar_field[:, resolution - i - 1, :]
            if first:
                im = plt.imshow(slice_data, cmap=plt.cm.gray)
            else:
                im.set_data(slice_data)
            plt.pause(1e-3)
            plt.draw()
        plt.close('all')
        return scalar_field

    def eval_grid_gpu(self, grid_object):
        xv, yv, zv = grid_object
        assert(xv.shape == yv.shape == zv.shape)
        assert(xv.shape[0] == xv.shape[1] == xv.shape[2])
        resolution = xv.shape[0]

        x_size, y_size, z_size = \
            int32(resolution), \
            int32(resolution), \
            int32(resolution)

        positions = np.vstack(
            [xv.ravel(), yv.ravel(), zv.ravel()]).T.astype(float32).flatten()
        positions_cuda = drv.In(positions)

        dest = zeros((resolution, resolution, resolution), dtype=float32)

        bs = 8

        self.grid_execution(
            drv.Out(dest),
            x_size, y_size, z_size,
            positions_cuda,
            self.vertices_x_cuda, self.vertices_y_cuda, self.vertices_z_cuda,
            self.faces_cuda, self.num_faces_cuda,
            block=(bs, bs, bs),
            grid=(resolution // bs, resolution // bs, resolution // bs))

        return dest


def mesh_to_sdf(mesh, resolution):
    m = MeshEvaluator(mesh)

    x_grid = np.linspace(0, 1, resolution)
    y_grid = np.linspace(0, 1, resolution)
    z_grid = np.linspace(0, 1, resolution)
    xv, yv, zv = meshgrid(x_grid, y_grid, z_grid)

    sdf_buffer = zeros(resolution * resolution * resolution, dtype=float32)
    sdf_buffer = m.eval_grid((xv, yv, zv)).flatten()
    #sdf_buffer = m.eval_grid_gpu((xv, yv, zv)).flatten()
    header = array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=float32)
    return (sdf_buffer, header)


def mesh_to_big_sdf(mesh, resolution):
    m = MeshEvaluator(mesh)

    final_buffer = zeros((resolution, resolution, resolution), dtype=float32)
    eval_resolution = 32
    block_dim = resolution // eval_resolution
    edge_size = eval_resolution / resolution
    for i in range(block_dim):
        for j in range(block_dim):
            for k in range(block_dim):
                region_start = array([i, j, k], dtype=float32) / \
                    block_dim
                region_end = array([i + 1, j + 1, k + 1], dtype=float32) / \
                    block_dim
                x_grid = np.linspace(
                    region_start[0], region_end[0], eval_resolution)
                y_grid = np.linspace(
                    region_start[1], region_end[1], eval_resolution)
                z_grid = np.linspace(
                    region_start[2], region_end[2], eval_resolution)
                xv, yv, zv = meshgrid(x_grid, y_grid, z_grid)

                print("Finished region {}:{}".format(region_start, region_end))

                sdf_buffer = m.eval_grid((xv, yv, zv)).flatten()
                #sdf_buffer = m.eval_grid_gpu((xv, yv, zv))
                final_buffer[j * eval_resolution: (j + 1) * eval_resolution,
                             i * eval_resolution: (i + 1) * eval_resolution,
                             k * eval_resolution: (k + 1) * eval_resolution] = \
                    sdf_buffer
    header = array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=float32)
    return (final_buffer, header)


def write_sdf_to_file(file_name, header, sdf, resolution):
    header_buffer = header.tobytes()
    sdf_buffer = sdf.tobytes()
    with open(file_name, 'wb') as f:
        f.write(header_buffer)
        f.write(resolution.to_bytes(4, 'little'))
        f.write(resolution.to_bytes(4, 'little'))
        f.write(resolution.to_bytes(4, 'little'))
        f.write(sdf_buffer)


def load_sdf_from_file(file_name):
    with open(file_name, 'rb') as f:
        header = np.frombuffer(f.read(4 * 6), dtype=np.float32, count=6)
        resolution_x = struct.unpack('<I', f.read(4))[0]
        resolution_y = struct.unpack('<I', f.read(4))[0]
        resolution_z = struct.unpack('<I', f.read(4))[0]
        size = resolution_x * resolution_y * resolution_z
        sdf_linear = np.frombuffer(
            f.read(size * 4), dtype=np.float32, count=size)
        sdf = sdf_linear.reshape((resolution_x, resolution_y, resolution_z))
    return (header, sdf)


if __name__ == '__main__':
    assert(len(argv) == 2)
    resolution = 32
    sdf, header = mesh_to_sdf(
        pymesh.load_mesh(argv[1]), resolution)
    write_sdf_to_file(basename(argv[1])[:-3] + 'sdf', header, sdf, resolution)

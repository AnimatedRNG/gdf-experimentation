#!/usr/bin/env python3

from multiprocessing import Process, Queue
from model import generate_volume, gdf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from skimage import measure
from numpy import sin, cos, pi
import numpy as np


def visualize_sdf_as_mesh(sdf, resolution=32):
    s = (1.0 / resolution) ** 2
    try:
        verts, faces = measure.marching_cubes(sdf, 0, spacing=(s, s, s))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                        cmap='Spectral', lw=1)
        plt.show()
    except:
        print("Failed")
        print(sdf)


queue = Queue()


def start_visualization():
    def vis(q):
        while True:
            model, resolution = q.get()
            sdf, _ = generate_volume(gdf(*model), resolution=resolution)
            sdf = np.reshape(sdf, (resolution, resolution, resolution))
            print(sdf.shape)
            visualize_sdf_as_mesh(sdf, resolution)
    p = Process(target=vis, args=(queue,))
    p.start()


def sdf_visualization(model, resolution=20):
    #queue.put((model, resolution))
    sdf, _ = generate_volume(gdf(*model), resolution=resolution)
    sdf = np.reshape(sdf, (resolution, resolution, resolution))
    visualize_sdf_as_mesh(sdf, resolution)

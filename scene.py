#!/usr/bin/env python3

import h5py
import numpy as np
import torch
from collections import namedtuple

GridSDF = namedtuple('GridSDF', 'start end perspective view data')


def load(filename):
    h5_file = h5py.File(filename)
    metadata = h5_file.get("metadata")
    resolution = torch.tensor((int(metadata['resolution_x'][0]),
                               int(metadata['resolution_y'][0]),
                               int(metadata['resolution_z'][0])))
    start = torch.tensor((float(metadata['start_x'][0]),
                          float(metadata['start_y'][0]),
                          float(metadata['start_z'][0])))
    end = torch.tensor((float(metadata['end_x'][0]),
                        float(metadata['end_y'][0]),
                        float(metadata['end_z'][0])))
    perspective = torch.tensor(metadata['perspective_matrix']).view(4, 4)
    view = torch.tensor(metadata['view_matrix']).view(4, 4)
    data = torch.from_numpy(h5_file.get("data").value).view(
        resolution[0], resolution[1], resolution[2])

    return GridSDF(start, end, perspective, view, data)


if __name__ == '__main__':
    load("test.hdf5")

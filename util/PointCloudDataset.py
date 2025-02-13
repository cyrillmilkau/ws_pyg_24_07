# Copyright (c) 2025 Cyrill Milkau
# Licensed under the Apache License, Version 2.0
# See LICENSE file for details.

import sys
sys.path.append(r"/workspace/")

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import laspy
import numpy as np

from util.las_util import load_las_file

class PointCloudDataset(Dataset):
    def __init__(self, file_list, max_points=4096, use_random=False):
        '''
        :param file_list: list of input file (format: las or laz)
        :param max_points: maximum points used from cloud, -1 will take all points (default is 4096)
        :param use_random: if maximum points is reasonable defined (!= -1) this decides whether random points are taken or the first max_point points (default is False; i.e. use the FIRST max_point points instead)
        '''
        
        self.file_list = file_list
        self.max_points = max_points
        self.use_random = use_random

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        points, labels = load_las_file(file_path)

        # # Only keep points labeled as "Building" (assuming class 6 is building)
        # building_mask = self.labels == 6
        # self.points = self.points[building_mask]
        # self.labels = self.labels[building_mask]

        return preprocess_point_cloud(points, labels, self.max_points, self.use_random)
    
def preprocess_point_cloud(points, labels, max_points, use_random):
    points = (points - points.mean(axis=0)) / points.std(axis=0)

    if max_points > 0 and len(points) > max_points:
        if use_random:
            indices = np.random.choice(len(points), max_points, replace=False)
        else:
            indices = np.arange(max_points)
        
        points, labels = points[indices], labels[indices]
    else:
        indices = np.arange(len(points))

    pos_t = torch.tensor(points, dtype=torch.float)
    lab_t = torch.tensor(labels, dtype=torch.long)
    idc_t = torch.tensor(indices, dtype=torch.long)
    
    return Data(pos=pos_t, y=lab_t, idx = idc_t)
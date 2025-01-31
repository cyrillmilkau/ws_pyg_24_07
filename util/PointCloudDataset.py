import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import laspy
import numpy as np

class PointCloudDataset(Dataset):
    def __init__(self, file_list, max_points=4096):
        self.file_list = file_list
        self.max_points = max_points

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        points, labels = load_las_file(file_path)
        return preprocess_point_cloud(points, labels, self.max_points)
    
def preprocess_point_cloud(points, labels, max_points):
    points = (points - points.mean(axis=0)) / points.std(axis=0)

    if len(points) > max_points:
        # idx = np.random.choice(len(points), max_points, replace=False)
        idx = np.arange(max_points)
        points, labels = points[idx], labels[idx]

    pos = torch.tensor(points, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    return Data(pos=pos, y=y)

def load_las_file(las_path):
    las = laspy.read(las_path)
    points = np.vstack((las.x, las.y, las.z)).T  # Extract XYZ coordinates
    labels = las.classification  # Extract classification labels if available
    return points, labels

"""
# las_file = "/workspace/train2/pc2011_10245101_sub.las"
# points, labels = load_las_file(las_file)
# print(points.shape, labels.shape)
# data = preprocess_point_cloud(points, labels)
# print(data)
"""
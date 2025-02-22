import torch
from torch_geometric.data import InMemoryDataset, Data
from pathlib import Path
import laspy
import numpy as np


def read_las(pointcloudfile,get_attributes=False,useevery=1):
    '''
    :param pointcloudfile: specification of input file (format: las or laz)
    :param get_attributes: if True, will return all attributes in file, otherwise will only return XYZ (default is False)
    :param useevery: value specifies every n-th point to use from input, i.e. simple subsampling (default is 1, i.e. returning every point)
    :return: 3D array of points (x,y,z) of length number of points in input file (or subsampled by 'useevery')
    '''

    # read the file
    inFile = laspy.read(pointcloudfile)
    # get the coordinates (XYZ)
    coords = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
    coords = coords[::useevery, :]
    if get_attributes == False:
        return (coords)
    else:
        las_fields= [info.name for info in inFile.points.point_format.dimensions]
        attributes = {}
        for las_field in las_fields[3:]: # skip the X,Y,Z fields
            attributes[las_field] = inFile.points[las_field][::useevery]
        return (coords, attributes)

class PointCloudsInFiles(InMemoryDataset):
    """Point cloud dataset where one data point is a file."""

    def __init__(self, root_dir, glob='*', column_name='', max_points=200_000, use_columns=None):
        """
        Args:
            root_dir (string): Directory with the datasets
            glob (string): Glob string passed to pathlib.Path.glob
            column_name (string): Column name to use as target variable (e.g. "Classification")
            use_columns (list[string]): Column names to add as additional input
        """
        self.files = list(Path(root_dir).glob(glob))
        self.column_name = column_name
        self.max_points = max_points
        if use_columns is None:
            use_columns = []
        self.use_columns = use_columns
        super().__init__()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = str(self.files[idx])
        coords, attrs = read_las(filename, get_attributes=True)
        if coords.shape[0] >= self.max_points:
            use_idx = np.random.choice(coords.shape[0], self.max_points, replace=False)
        else:
            use_idx = np.random.choice(coords.shape[0], self.max_points, replace=True)
        if len(self.use_columns) > 0:
            x = np.empty((self.max_points, len(self.use_columns)), np.float32)
            for eix, entry in enumerate(self.use_columns):
                x[:, eix] = attrs[entry][use_idx]
        else:
            x = coords[use_idx, :]
        coords = coords - np.mean(coords, axis=0)  # centralize coordinates

        # impute target
        target = attrs[self.column_name]
        # target[np.isnan(target)] = np.nanmean(target)
        target[np.isnan(target)] = 0
        print(f"len x: {len(x)}")
        print(f"x[0]: {x[0]}")
        print(f"len y: {len(target[use_idx])}")
        print(f"len pos {len(coords[use_idx, :])}")
        sample = Data(x=torch.from_numpy(x).float(),
                      y=torch.from_numpy(np.array(target[use_idx])[:, np.newaxis]).float(), # .int(),
                      pos=torch.from_numpy(coords[use_idx, :]).float())
        if coords.shape[0] < 100:
            return None
        return sample
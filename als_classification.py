import sys
sys.path.append(r"/workspace/")

import glob
import laspy
import numpy as np

import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from tqdm import tqdm

from util.ClassificationLabels import ClassificationLabels
from util.PointCloudDataset import PointCloudDataset
from util.PointNet2 import PN2_Classification

MAX_POINTS = 1000

# -------------------------------------------------------------------------------------------------------------------------------
# !!! this is only to be used, if the LAS-file already contains all classes !!!! otherwise use ctor of ClassificationLabels
"""
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

coords, attrs = read_las("pc2011_10245101_sub.las", get_attributes=True)
classification_values = attrs["classification"]
unique_classes, class_counts = np.unique(classification_values, return_counts=True)
print(f"classification_values: {classification_values}")
print(f"Unique classes: {unique_classes}")
print(f"Class counts: {class_counts}")
total_points = len(classification_values)
sum_counts = np.sum(class_counts)
print(f"Total points: {total_points}")
print(f"Sum of class counts: {sum_counts}")
print(f"Match: {total_points == sum_counts}")
"""
# -------------------------------------------------------------------------------------------------------------------------------

def run_training_procedure():

    las_files = glob.glob("/workspace/data/train/a/*.las")

    dataset = PointCloudDataset(las_files, MAX_POINTS)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # print(dataloader)

    classification = ClassificationLabels('Vorarlberg')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_target_classes = len(classification.class_labels) # len(unique_classes)
    model = PN2_Classification(num_features=0, num_target_classes=num_target_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for param in model.parameters():
        if param.requires_grad:
            param.data.fill_(1) # Set all model weights to 1

    for epoch in range(10):  # Increase epochs as needed
        model.train()
        total_loss = 0
        for _, data in enumerate(tqdm(dataloader, desc='Training in progress')):

            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)

            # print(f"Output shape: {out.shape}")
            # print(f"Target shape: {data.y.shape}")
            # print(f"Min label: {data.y.min()}, Max label: {data.y.max()}, Unique labels: {data.y.unique()}")

            assert all(classification.is_valid_label(label.item()) for label in data.y), \
                f"Invalid label detected! >>> {data.y[torch.argmax(out)].item()} <<<"

            # predicted_labels = out.argmax(dim=1)
            # predicted_class_names = [classification.get_name_from_label(label.item()) for label in predicted_labels]
            # print(f"Predicted labels: {predicted_labels[:5]}")
            # print(f"Predicted class names: {predicted_class_names[:5]}")

            # target_class_names = [classification.get_name_from_label(label.item()) for label in data.y]
            # print(f"Target labels: {data.y[:5]}")
            # print(f"Target class names: {target_class_names[:5]}")

            # loss = F.nll_loss(out, data.y.long())
            loss = F.cross_entropy(out, data.y.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

    torch.save(model.state_dict(), str("./output/best.model"))
    torch.cuda.empty_cache()

if __name__ == "__main__":
    run_training_procedure()
import os
import sys
sys.path.append(r"/workspace/")

import glob
import laspy
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from util.ClassificationLabels import ClassificationLabels
from util.PointCloudDataset import PointCloudDataset
from util.PointNet2 import PN2_Classification

MAX_POINTS          =       -1
MAX_EPOCH           =       10
DEF_LR              =       1e-4
DEF_WEIGHT          =       1e0
DEF_FEATURES        =       0
B_USE_DEF_WEIGHT    =       True
B_USE_RAND_SEL      =       False    # [related to MAX_POINTS]
B_SAVE_DELECTED     =       True

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

def extract_subset_of_points(inFile, selected_indices):
    """
    Helper function to extract a subset of points from a LAS file based on selected indices.
    
    :param inFile: The laspy LAS file object
    :param selected_indices: The indices of the points to extract
    :return: A tuple (x, y, z, classification)
    """
    # Ensure selected indices are numpy arrays for slicing
    selected_indices = np.array(selected_indices)
    
    # Extract the relevant point dimensions
    x_subset = inFile.x[selected_indices]
    y_subset = inFile.y[selected_indices]
    z_subset = inFile.z[selected_indices]
    classification_subset = inFile.points.classification[selected_indices]
    
    return x_subset, y_subset, z_subset, classification_subset

def save_classified_las(original_las_file, predicted_labels, output_file, selected_indices=None, save_only_selected_points=False):
    '''
    Adds classification labels to a LAS file and saves a new LAS file with the labels.

    :param original_las_file: path to the input LAS file
    :param predicted_labels: a numpy array of predicted labels for each point in the point cloud
    :param output_file: the path to save the new LAS file with the classification labels
    :param selected_indices: indices of the points to update in the classification field (optional)
    '''

    # Read the original LAS file
    inFile = laspy.read(original_las_file)

    # Check if 'classification' is available in the dimensions
    classification_found = False
    for dimension in inFile.point_format.dimensions:
        if dimension.name == 'classification':
            classification_found = True
            break

    if not classification_found:
        print("Error: The LAS file format doesn't support classification.")
        return
    
    if not selected_indices:
        print("Error: The indices of points receiving the new labels are missing. Find them in PointCloudDataset return values.")
        return

    # Store the old classification values as the nex classification values before updating the labels
    new_classification_values = inFile.points.classification.copy()
    # print(f"new_classification_values: {len(new_classification_values)}")

    # Convert the tensor with new predictions to numpy and then to uint8
    predicted_labels_np = predicted_labels.cpu().to(torch.uint8).numpy()

    #--------------------------------------------------------------------------------------
    ## Dummy part --> Test arbitrary classification of classes [1] and [2]
    # num_points = len(inFile.x)
    # dummy_labels = np.zeros(num_points, dtype=np.uint8)

    # # Assign label 1 to the first half and label 2 to the second half
    # half = num_points // 2
    # dummy_labels[:half] = 1
    # dummy_labels[half:] = 2
    #--------------------------------------------------------------------------------------
    
    #--------------------------------------------------------------------------------------
    ## Selected indices part --> Apply new labels to the selected points 
    if ( False ): # --> This takes very long ...
        predicted_labels_np = predicted_labels.cpu().numpy()
        for idx in selected_indices:
            print(f"idx0: {idx}")
            print(f"idx1: {selected_indices.tolist().index(idx)}")
            new_classification_values[idx] = predicted_labels_np[selected_indices.tolist().index(idx)]
    else: # --> This should do the same, but faster ...
        selected_indices_np = selected_indices.cpu().numpy()
        new_classification_values[selected_indices_np] = predicted_labels_np[selected_indices_np]

    #--------------------------------------------------------------------------------------

    # Add the predicted labels to the classification field
    inFile.points.classification = new_classification_values # Ensure labels are in uint8 format
    # inFile.points.classification = dummy_labels # Use the dummy labels for classification

    # # if save_only_selected_points:
    # #     # Extract the subset of points
    # #     x_subset, y_subset, z_subset, classification_subset = extract_subset_of_points(inFile, selected_indices)

    # #     # Reconstruct the LAS file with the subset
    # #     new_file = laspy.create(point_format=inFile.point_format)
    # #     new_file.x = x_subset
    # #     new_file.y = y_subset
    # #     new_file.z = z_subset
    # #     new_file.points.classification = classification_subset

    # #     # Save the updated LAS file
    # #     new_file.write(output_file)
    # #     print(f"New classified LAS file saved as: {output_file}")
    if save_only_selected_points:

        selected_indices = selected_indices.cpu().numpy()

        # Extract all point attributes for the selected points
        x_subset = inFile.x[selected_indices]
        y_subset = inFile.y[selected_indices]
        z_subset = inFile.z[selected_indices]
        classification_subset = inFile.points.classification[selected_indices]
        intensity_subset = inFile.intensity[selected_indices]
        return_number_subset = inFile.return_number[selected_indices]
        number_of_returns_subset = inFile.number_of_returns[selected_indices]
        #scan_angle_rank_subset = inFile.scan_angle_rank[selected_indices]
        #user_data_subset = inFile.user_data[selected_indices]
        #point_source_ID_subset = inFile.point_source_ID[selected_indices]
        gps_time_subset = inFile.gps_time[selected_indices]

        # Create a new LAS file with the same point format as the original
        new_file = laspy.create(point_format=inFile.point_format)

        # Add the selected points with their corresponding attributes to the new file
        new_file.x = x_subset
        new_file.y = y_subset
        new_file.z = z_subset
        new_file.points.classification = classification_subset
        new_file.intensity = intensity_subset
        new_file.return_number = return_number_subset
        new_file.number_of_returns = number_of_returns_subset
        #new_file.scan_angle_rank = scan_angle_rank_subset
        #new_file.user_data = user_data_subset
        #new_file.point_source_ID = point_source_ID_subset
        new_file.gps_time = gps_time_subset

        # Save the new LAS file
        new_file.write(output_file)
        print(f"New classified LAS-subset file saved as: {output_file}")
    else:
        # Save the full LAS file (with updated classification)
        inFile.write(output_file)
        print(f"New classified LAS file saved as: {output_file}")

def run_training_procedure(save_model = False):

    las_files = glob.glob("/workspace/data/train/a/*.las")

    dataset = PointCloudDataset(las_files, MAX_POINTS, False)# B_USE_RAND_SEL)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # print(f"Dataloader finished")

    classification = ClassificationLabels('Vorarlberg')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_target_classes = len(classification.class_labels) # len(unique_classes)
    model = PN2_Classification(num_features=DEF_FEATURES, num_target_classes=num_target_classes).to(device)

    if (B_USE_DEF_WEIGHT):
        for param in model.parameters():
            if param.requires_grad:
                param.data.fill_(DEF_WEIGHT)
    # else:
        # add custom weights --> TODO
    
    optimizer = torch.optim.Adam(model.parameters(), lr=DEF_LR)
    
    # print(f"Model finished")

    for epoch in range(MAX_EPOCH):  # Increase epochs as needed
        model.train()
        total_loss = 0
        for _, data in enumerate(tqdm(dataloader, desc='>> TRAINING >>')):

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

            ## Check indices used for training (PointCloudDataset property)
            # selected_indices = data.idx
            # print(f"Selected indices: {selected_indices}")

            # loss = F.nll_loss(out, data.y.long())
            loss = F.cross_entropy(out, data.y.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")
        if (save_model):
            print("Save Model")
            # os.makedirs(f"./output/models/{str(las_files[0])}")
            torch.save(model.state_dict(), str(f"./output/models/epoch_{epoch+1}.model"))

    if (save_model):
        print("Save Model")
        torch.save(model.state_dict(), str("./output/models/best.model"))
    torch.cuda.empty_cache()

def run_inference_procedure(model_path = None, las_file_path = None):

    if not model_path:
        print(f"No model found at >>> {model_path} <<< .. exiting")
        return

    if not las_file_path:
        print("No LAS files provided .. exiting")
        return

    classification = ClassificationLabels('Vorarlberg')

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PN2_Classification(num_features=DEF_FEATURES, num_target_classes=len(classification.class_labels)).to(device)

    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError as e:
        print(f"Error: {e} \nNo model found at >>> {model_path} <<< .. exiting")
        return
    
    model.eval()  # Set the model to evaluation mode

    las = laspy.read(las_file_path)
    points = las.xyz
    points_tensor = torch.tensor(points, dtype=torch.float).to(device)
    data = Data(pos=points_tensor)
    data.batch = torch.zeros(points_tensor.size(0), dtype=torch.long).to(device)
    with torch.no_grad():  # No need to compute gradients during inference
        output = model(data)

    _, predicted_classes = output.max(dim=1)
    print(points.shape)
    print(predicted_classes.shape)

    unique_classes, class_counts = np.unique(predicted_classes.cpu().to(torch.uint8).numpy(), return_counts=True)
    print(f"unique classes: {unique_classes}, {class_counts}")
    save_classified_las(las_file_path, predicted_classes, "/data/new_file.las")

def test_model_on_training_data():
    las_file_path = "/workspace/data/train/a/pc2011_10245101_sub3.las" # "/workspace/data/train/a/pc2011_10245101_sub.las"
    las_files = glob.glob(las_file_path)
    dataset = PointCloudDataset(las_files, MAX_POINTS, B_USE_RAND_SEL)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    classification = ClassificationLabels('Vorarlberg')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PN2_Classification(num_features=DEF_FEATURES, num_target_classes=len(classification.class_labels)).to(device)

    try:
        model.load_state_dict(torch.load("./output/models/epoch_10.model"))
    except FileNotFoundError as e:
        print(f"Error: {e} \nNo model found at >>> ./output/models/epoch_10.model <<< .. exiting")
        return
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for _, data in enumerate(tqdm(dataloader, desc='>> EVALUATING >>')):
            data = data.to(device)
            predicted_classes = model(data).argmax(dim=1)
            unique_classes, class_counts = np.unique(predicted_classes.cpu().to(torch.uint8).numpy(), return_counts=True)
            print(f"unique classes: {unique_classes}, {class_counts}")
            selected_indices = data.idx
            # print(f"Selected indices: {selected_indices}")
            save_classified_las(las_file_path, predicted_classes, "/data/new_file_test_train.las", selected_indices, B_SAVE_DELECTED)
    return # predicted_classes

if __name__ == "__main__":

    modality = "train" # "test"/"train"/"test_model_on_training_data"

    if (modality == "train"):
        run_training_procedure(False)
    elif (modality == "test"):
        run_inference_procedure("./output/models/best.model", "./data/general/lsc_33412_5658_2_sn_sub.las")
    elif (modality=="test_model_on_training_data"):
        test_model_on_training_data()
    else:
        print("No modality selected ... exiting.")
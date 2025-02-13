# Copyright 2025 Cyrill Milkau

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# See LICENSE file for details.

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
from util.las_util import save_classified_las

MAX_POINTS          =       200
MAX_EPOCH           =       100
DEF_LR              =       1e-4
DEF_WEIGHT          =       1e0
DEF_FEATURES        =       0
B_USE_DEF_WEIGHT    =       True
B_USE_RAND_SEL      =       False    # [related to MAX_POINTS]
B_SAVE_DELECTED     =       True

def run_training_procedure(las_files, save_model = False):

    dataset = PointCloudDataset(las_files, MAX_POINTS, False)# B_USE_RAND_SEL)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    classification = ClassificationLabels('Vorarlberg')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_target_classes = len(classification.class_labels) # len(unique_classes)
    
    model = PN2_Classification(num_features=DEF_FEATURES, num_target_classes=num_target_classes).to(device)
    # print(f"model.parameters(): {len(list(model.parameters()))}")

    if (B_USE_DEF_WEIGHT):
        for param in model.parameters():
            if param.requires_grad:
                param.data.fill_(DEF_WEIGHT)
    # else:
        # add custom weights --> TODO
    
    optimizer = torch.optim.Adam(model.parameters(), lr=DEF_LR)
    
    for epoch in range(MAX_EPOCH):  # Increase epochs as needed
        model.train()
        total_loss = 0
        for _, data in enumerate(tqdm(dataloader, desc='>> TRAINING >>')):

            # print(f"data type: {type(data)}")
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data) # !!!

            # print(f"Output shape: {out.shape}")
            # print(f"Target shape: {data.y.shape}")
            # print(f"Min label: {data.y.min()}, Max label: {data.y.max()}, Unique labels: {data.y.unique()}")

            # # # assert all(classification.is_valid_label(label.item()) for label in data.y), \
            # # #     f"Invalid label detected! >>> {data.y[torch.argmax(out)].item()} <<<"

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

    # las_files = glob.glob("/workspace/data/general/pc2011/*.las")
    las_files = glob.glob("/workspace/data/train/a/*.las")

    modality = "inference" # "inference"/"train"/"test_model_on_training_data"

    if (modality == "train"):
        run_training_procedure(las_files, True)
    elif (modality == "inference"):
        run_inference_procedure("./output/models/epoch_100.model", "./data/general/lsc_33412_5658_2_sn.las")
    elif (modality=="test_model_on_training_data"):
        test_model_on_training_data()
    else:
        print("No modality selected ... exiting.")
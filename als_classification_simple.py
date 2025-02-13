import laspy
import torch
import math
from torch_geometric.data import Data, Dataset
import numpy as np
import os
import glob
import random 
import subprocess
from datetime import datetime

from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split

from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from util.ClassificationLabels import ClassificationLabels

N_POINTS = 1_000_0
EPOCHS = 100
TARGET_CLASSES = 14

def reset_gpu():
    try:
        subprocess.run(['nvidia-smi', '--gpu-reset'], check=True)
        print("GPU reset successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error resetting GPU: {e}")

def handle_cuda_error(e):
    if "CUDA error: device-side assert triggered" in str(e):
        print(f"CUDA error caught: {e}")
        torch.cuda.empty_cache()  # Clear the CUDA cache
        reset_gpu()  # Reset GPU to recover
    else:
        raise e  # Reraise the exception if it's not the one we want to catch

class PointCloudChunkedDataset(Dataset):
    def __init__(self, las_files, n_points=1000, label_mapping=None, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.las_files = las_files
        self.n_points = n_points
        self.n_chunks = 0
        self.point_chunks = []  # Stores (file_idx, start_idx) for all chunks
        self.label_mapping = label_mapping

        # Precompute chunks for all files
        for file_idx, las_file in enumerate(las_files):
            las = laspy.read(las_file)
            total_points = len(las.x)
            # self.n_chunks = total_points // self.n_points  # Full chunks
            self.n_chunks = 1500
            
            for chunk_idx in range (self.n_chunks):
                if (chunk_idx * self.n_points) < total_points:
                    # print(f"chunk_idx * self.n_points: {chunk_idx * self.n_points}")
                    self.point_chunks.append((file_idx, chunk_idx * self.n_points))
                else:
                    break

        self.pbar = tqdm(total=len(self), desc="Processing Chunks", position=0, leave=True)

    def len(self):
        return len(self.point_chunks)

    def get(self, idx):

        file_idx, start_idx = self.point_chunks[idx]
        las = laspy.read(self.las_files[file_idx])
        
        # Extract features
        xyz = np.column_stack((las.x, las.y, las.z))
        intensity = np.array(las.intensity, dtype=np.float32)
        return_number = np.array(las.return_number, dtype=np.float32)
        labels = np.array(las.classification, dtype=np.int64) # - 1  # Ensure zero-based labels

        features = np.column_stack((xyz, intensity, return_number))
        # print(f"features: {len(features)}")
        # indices = random.sample(range(len(features)), min(self.n_points, len(features)))

        # Extract the sequential chunk
        end_idx = start_idx + self.n_points
        # print(f"start_idx: {start_idx}")
        # print(f"start_idx: {end_idx}")
        x = torch.tensor(features[start_idx:end_idx], dtype=torch.float)
        # y = torch.tensor(labels[start_idx:end_idx], dtype=torch.long)
        # y = torch.tensor([self.label_mapping[label] for label in labels if label in self.label_mapping], dtype=torch.long)
        y = torch.tensor([self.label_mapping.get(label, -1) for label in labels[start_idx:end_idx]], dtype=torch.long)

        # print(idx, x.shape, y.shape, "\r")
        self.pbar.update(1)

        return Data(x=x, y=y)

class PointCloudDataset(Dataset):
    def __init__(self, las_files, n_points=2000, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.las_files = las_files
        self.n_points = n_points

    def len(self):
        return len(self.las_files)

    def get(self, idx):
        # Load LAS file
        las = laspy.read(self.las_files[idx])
        
        # Extract features (modify based on available attributes)
        xyz = np.vstack((las.x, las.y, las.z)).T
        intensity = np.array(las.intensity, dtype=np.float32)
        return_number = np.array(las.return_number, dtype=np.float32)
        
        # Stack features
        features = np.column_stack((xyz, intensity, return_number))
        # features = np.column_stack((xyz))
        
        # Extract labels (classification field)
        labels = np.array(las.classification, dtype=np.int64)
        
        valid_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16])
        mapped_labels = np.arange(len(valid_labels))
        label_mapping = {old: new for old, new in zip(valid_labels, mapped_labels)}

        # Randomly sample n points
        indices = random.sample(range(len(features)), min(self.n_points, len(features)))
        # x = torch.tensor(features[indices], dtype=torch.float)
        # y = torch.tensor(labels[indices], dtype=torch.long)
        x = torch.tensor(features[0:self.n_points], dtype=torch.float)
        # y = torch.tensor(labels[0:self.n_points], dtype=torch.long)
        y = torch.tensor([label_mapping[label] for label in labels if label in label_mapping], dtype=torch.long)

        print(f"x shape: {x.shape}, y shape: {y.shape}")

        return Data(x=x, y=y)

class MLPClassifier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(in_channels, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_channels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x = F.relu(self.fc1(data.x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def load_model(model_path, in_channels, out_channels, device):
    model = MLPClassifier(in_channels, out_channels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def classify_point_cloud(model, las_file, device, n_points=1000):
    las = laspy.read(las_file)
    
    # Extract features
    xyz = np.column_stack((las.x, las.y, las.z))  # Shape: (num_points, 3)
    intensity = np.array(las.intensity, dtype=np.float32)
    return_number = np.array(las.return_number, dtype=np.float32)

    features = np.column_stack((xyz, intensity, return_number))  # Shape: (num_points, 5)
    
    # Randomly sample points
    # indices = random.sample(range(len(features)), min(n_points, len(features)))
    x = torch.tensor(features[0:n_points], dtype=torch.float).to(device)  # Shape: (n_points, 5)

    # Inference
    with torch.no_grad():
        out = model(Data(x=x))
        predicted_labels = out.argmax(dim=1).cpu().numpy()  # Convert to numpy for easy use

    # labels = np.concatenate([data.y.numpy() for data in train_data])
    unique_labels, counts = np.unique(predicted_labels, return_counts=True)
    print(unique_labels, counts)

    # return indices, predicted_labels
    return predicted_labels

def save_reclassified_las(las_file, n_points, new_labels, output_file):
    las = laspy.read(las_file)
    las.classification[0:n_points] = new_labels  # Update classification
    las.write(output_file)
    print(f"Updated LAS file saved to: {output_file}")

def main():
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    las_files = glob.glob("/workspace/data/train/a2/*.las")

    valid_labels = ClassificationLabels("Vorarlberg").class_labels
    mapped_labels = np.arange(len(valid_labels))
    label_mapping = {old: new for old, new in zip(valid_labels, mapped_labels)}
    label_mapping_inv = {v: k for k, v in label_mapping.items()}
    print("Label Mapping:", label_mapping)
    print("Label Mapping inverse:", label_mapping_inv)

    # dataset = PointCloudDataset(las_files, n_points=N_POINTS)
    dataset = PointCloudChunkedDataset(las_files, n_points=N_POINTS, label_mapping=label_mapping)

    # Convert dataset to list
    data_list = [dataset[i] for i in range(len(dataset))]

    # Split
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    
    labels = np.concatenate([data.y.numpy() for data in train_data])
    unique_labels, counts = np.unique(labels, return_counts=True)
    # print(unique_labels, counts)
    
    """
    min_label = unique_labels.min()
    max_label = unique_labels.max()
    min_mapped = label_mapping_inv.get(min_label, None)
    max_mapped = label_mapping_inv.get(max_label, None)
    print(f"Min label: {min_label}, {min_mapped}, Max label: {max_label}, {max_mapped}")
    """

    # PyTorch Geometric DataLoader
    train_loader = DataLoader(train_data, batch_size=8, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Initialize model
    model = MLPClassifier(in_channels=5, out_channels=TARGET_CLASSES)  # Modify output classes based on dataset
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize a zero-filled array for all valid labels
    class_counts = np.zeros(len(valid_labels), dtype=np.int64)

    # Fill in the counts for present labels
    for label, count in zip(unique_labels, counts):
        if label in label_mapping:  # Only assign if label is in known classes
            class_counts[label_mapping[label]] = count
    # print(len(class_counts), class_counts)

    # weights = 1.0 / (class_counts + 1e-6)  # Avoid division by zero
    # weights[class_counts == 0] = 0  # Set weights to 0 for missing classes
    # weights = torch.tensor(weights, dtype=torch.float).to(device)
    # print(weights)
    # criterion = nn.CrossEntropyLoss(weight=weights)
    criterion = nn.CrossEntropyLoss()

    """
    percent = 0.1
    chunk_abs_count = len(train_data)
    chunk_eff_count = max(1, math.floor(percent * np.float64(chunk_abs_count)) )
    print(f"chunk_abs_count: {chunk_abs_count}")
    print(f"chunk_eff_count: {chunk_eff_count}")
    """

    # def train():
    #     try:
    #         model.train()
    #         total_loss = 0
    #         progress_bar = tqdm(train_loader, desc="Training", leave=True)
    #         for data in progress_bar:
    #             data = data.to(device)
    #             # print(f"data: {type(data)}")
    #             optimizer.zero_grad()
    #             out = model(data)
    #             loss = criterion(out, data.y)
    #             loss.backward()
    #             optimizer.step()
    #             total_loss += loss.item()
    #             # progress_bar.set_postfix(loss=loss.item())
    #         return total_loss / len(train_loader)
    #     except RuntimeError as e:
    #         handle_cuda_error(e)  # Handle CUDA error gracefully

    def train():
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Training", leave=False, position=1)  # Inner progress bar

        for data in progress_bar:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())  # Updates inner progress bar with loss

        progress_bar.close()  # Ensure it properly finishes before the outer loop updates
        return total_loss / len(train_loader)


    def evaluate(loader):
        model.eval()
        correct = 0
        total = 0

        progress_bar = tqdm(loader, desc="Evaluating", leave=False, position=2)  # Inner progress bar

        with torch.no_grad():
            for data in progress_bar:
                data = data.to(device)
                out = model(data)
                pred = out.argmax(dim=1)
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)

        progress_bar.close()  # Close the progress bar after evaluation
        return correct / total

    
    # Lists to track loss and accuracy
    train_losses = []
    val_accuracies = []

    with tqdm(total=EPOCHS, desc="Epochs", unit="epoch", position=0) as pbar:
        for _ in range(EPOCHS):
            train_loss = train()
            val_acc = evaluate(val_loader)
            
            train_losses.append(train_loss)
            val_accuracies.append(val_acc)
            
            pbar.set_postfix(loss=train_loss, val_acc=val_acc)
            pbar.update(1)
    
    # Plot training loss and validation accuracy
    folder_name = datetime.now().strftime("%H-%M") + f"_FILES_{len(las_files)}_POINTS_{N_POINTS}_CHUNKS_{dataset.n_chunks}_EPOCHS_{EPOCHS}"
    folder_dir = os.path.join("./output/", folder_name)
    os.makedirs(folder_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS+1), train_losses, label="Train Loss", color="blue")
    plt.plot(range(1, EPOCHS+1), val_accuracies, label="Validation Accuracy", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss / Accuracy")
    plt.yscale("log")  # Set y-axis to log scale
    plt.title("Training Loss & Validation Accuracy Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(folder_dir,"training_plot.png"))  # Save the figure
    plt.savefig("training_plot.png")  # Save the figure
    # plt.show()

    test_acc = evaluate(test_loader)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(folder_dir,"mlp_classifier.pth"))
    print("Model saved successfully!")

    """
    model = load_model("output/models/models_new/mlp_classifier.pth", in_channels=5, out_channels=TARGET_CLASSES, device=device)
    las_file = "/workspace/data/general/pc2011/pc2011_11245000_RANDOM_SUBSAMPLED_2025-02-10_23h17_29_325.las"
    new_labels = classify_point_cloud(model, las_file, device, 10000000)
    # Print results
    print(f"Classified {len(new_labels)} points from {las_file}")
    output_file = "/workspace/reclassified.las"
    save_reclassified_las(las_file, 10000000, new_labels, output_file)
    """

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(data.y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
    print("Predicted class distribution:", dict(zip(unique_preds, pred_counts)))

    unique_labels, label_counts = np.unique(all_labels, return_counts=True)
    print("True class distribution:", dict(zip(unique_labels, label_counts)))

    all_features = np.concatenate([data.x.cpu().numpy() for data in test_loader])
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(all_features)
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.5)
    plt.title("Feature Space (PCA Projection)")
    plt.savefig(os.path.join(folder_dir,"reduced_features_plot.png"))

if __name__ == "__main__":
    # try:
    #     main()
    # except RuntimeError as e:
    #     handle_cuda_error(e)  # Handle error if occurs
    main()
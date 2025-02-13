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
import time

from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split

from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from util.ClassificationLabels import ClassificationLabels

from collections import defaultdict

from torch_geometric.nn import MLP, knn_interpolate, PointNetConv, global_max_pool, fps, radius

N_POINTS = 2**13 # 1024
EPOCHS = 100
TARGET_CLASSES = 14
BATCH_SIZE = 32

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
            self.n_chunks = 15
            
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
        self.fc1 = nn.Linear(in_channels, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, out_channels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x = F.relu(self.bn1(self.fc1(data.x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

class PointNetPlusPlus(torch.nn.Module):
    def __init__(self, num_features, num_target_classes):
        super().__init__()

        # Increased network capacity
        self.sa1_module = SAModule(0.2, 2, MLP([3 + num_features, 128, 128, 256]))
        self.sa2_module = SAModule(0.25, 4, MLP([256 + 3, 256, 256, 512]))
        self.sa3_module = GlobalSAModule(MLP([512 + 3, 512, 512, 1024]))

        # Wider classification layers
        self.mlp = MLP([1024, 512, 256, num_target_classes], 
                      dropout=0.3,  # Reduced dropout
                      batch_norm=True)

    def forward(self, data):
        pos = data.x[:, :3]
        features = data.x[:, 3:]
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)

        # Set abstraction layers
        sa0_out = (features, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        
        x, _, _ = sa3_out

        # Repeat features for each point
        x = x.repeat_interleave(N_POINTS, dim=0)
        
        return F.log_softmax(self.mlp(x), dim=-1)

def load_model(model_path, in_channels, out_channels, device, model_type='mlp'):
    if model_type == 'mlp':
        model = MLPClassifier(in_channels, out_channels)
    else:  # pointnet++
        model = PointNetPlusPlus(num_features=2, num_target_classes=out_channels)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def classify_point_cloud(model, las_file, device, n_points=1000, model_type='mlp'):
    las = laspy.read(las_file)
    
    # Extract features
    xyz = np.column_stack((las.x, las.y, las.z))
    intensity = np.array(las.intensity, dtype=np.float32)
    return_number = np.array(las.return_number, dtype=np.float32)
    features = np.column_stack((xyz, intensity, return_number))
    
    # Process in batches to avoid memory issues
    predictions = []
    batch_size = 10000
    
    for i in range(0, len(features), batch_size):
        batch = features[i:min(i+batch_size, len(features))]
        x = torch.tensor(batch, dtype=torch.float).to(device)
        
        # Create Data object with batch information for PointNet++
        if model_type == 'pointnet++':
            batch_idx = torch.zeros(len(batch), dtype=torch.long, device=device)
            data = Data(x=x, batch=batch_idx)
        else:
            data = Data(x=x)
        
        # Inference
        with torch.no_grad():
            out = model(data)
            batch_pred = out.argmax(dim=1).cpu().numpy()
            predictions.append(batch_pred)
    
    predicted_labels = np.concatenate(predictions)
    
    # Print detailed statistics
    unique_labels, counts = np.unique(predicted_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        percentage = (count / len(predicted_labels)) * 100
        print(f"Class {label}: {count} points ({percentage:.2f}%)")
    
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
    # print("Label Mapping:", label_mapping)
    # print("Label Mapping inverse:", label_mapping_inv)

    # dataset = PointCloudDataset(las_files, n_points=N_POINTS)
    dataset = PointCloudChunkedDataset(las_files, n_points=N_POINTS, label_mapping=label_mapping)

    folder_name = datetime.now().strftime("%H-%M") + f"_FILES_{len(las_files)}_POINTS_{N_POINTS}_CHUNKS_{dataset.n_chunks}_EPOCHS_{EPOCHS}"
    folder_dir = os.path.join("./output/", folder_name)
    os.makedirs(folder_dir, exist_ok=True)

    # Convert dataset to list
    data_list = [dataset[i] for i in range(len(dataset))]

    # Split
    random_seed = int(time.time()) # or '42'
    # print(f"Using random seed: {random_seed}")

    # Random split
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=random_seed)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=random_seed)

    
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
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Set random seeds for PyTorch
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Choose model type ('mlp' or 'pointnet++')
    model_type = 'pointnet++'  # or 'mlp'

    # Initialize model based on type
    if model_type == 'mlp':
        model = MLPClassifier(in_channels=5, out_channels=TARGET_CLASSES)
    else:  # pointnet++
        model = PointNetPlusPlus(num_features=2, num_target_classes=TARGET_CLASSES)  # 2 features: intensity, return_number
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                   factor=0.5, patience=5, 
                                                   verbose=True)
    
    # Calculate better class weights
    total_samples = sum(counts)
    class_weights = torch.zeros(TARGET_CLASSES, device=device)
    for label, count in zip(unique_labels, counts):
        if count > 0:  # Avoid division by zero
            class_weights[label] = 1.0 / (count / total_samples)
    
    # Normalize weights and apply log scaling to reduce extreme values
    class_weights = torch.log1p(class_weights)
    class_weights = class_weights / class_weights.sum()
    print("Class weights:", class_weights)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)

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


    def evaluate(loader, iter):
        model.eval()
        correct = 0
        total = 0
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        confusion_matrix = defaultdict(lambda: defaultdict(int))

        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out = model(data)
                pred = out.argmax(dim=1)
                
                # Overall accuracy
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
                
                # Per-class accuracy
                for p, t in zip(pred.cpu().numpy(), data.y.cpu().numpy()):
                    class_correct[t] += (p == t)
                    class_total[t] += 1
                    confusion_matrix[t][p] += 1
        if ( iter == EPOCHS - 1 ):
            # Print detailed statistics
            print("\nPer-class accuracy:")
            for class_idx in sorted(class_total.keys()):
                accuracy = class_correct[class_idx] / class_total[class_idx] if class_total[class_idx] > 0 else 0
                print(f"Class {class_idx}: {accuracy:.4f} ({class_correct[class_idx]}/{class_total[class_idx]})")
            
            print("\nConfusion Matrix:")
            classes = sorted(class_total.keys())
            print("True\Pred", end="\n")
            for c in classes:
                print(f"{c}", end="\t")
            print()
            for true_class in classes:
                print(f"{true_class}", end="\t")
                for pred_class in classes:
                    print(f"{confusion_matrix[true_class][pred_class]}", end="\t")
                print()

        return correct / total

    
    # Lists to track loss and accuracy
    train_losses = []
    val_accuracies = []

    with tqdm(total=EPOCHS, desc="Epochs", unit="epoch", position=0) as pbar:
        for epoch in range(EPOCHS):
            train_loss = train()
            val_acc = evaluate(val_loader, epoch)
            
            # Update learning rate based on validation accuracy
            scheduler.step(val_acc)
            
            train_losses.append(train_loss)
            val_accuracies.append(val_acc)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': model_type
            }, os.path.join(folder_dir, f"{model_type}_EPOCH_{epoch}_classifier.pth"))

            pbar.set_postfix(loss=train_loss, val_acc=val_acc)
            pbar.update(1)
        pbar.close()
    
    # Plot training loss and validation accuracy
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

    test_acc = evaluate(test_loader, EPOCHS)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type
    }, os.path.join(folder_dir, f"{model_type}_classifier.pth"))
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
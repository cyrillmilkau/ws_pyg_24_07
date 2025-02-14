import os
import laspy
import torch
import numpy as np
import glob
import random 
import subprocess
import time
import matplotlib.pyplot as plt

from datetime import datetime
from collections import defaultdict

from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import MLP, PointNetConv, global_max_pool, fps, radius

from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from util.ClassificationLabels import ClassificationLabels

N_POINTS        = 2**15             # points per chunk
EPOCHS          = 100               # epochs of training/evaluating
BATCH_SIZE      = 32                # 
TARGET_CLASSES  = 14                # depends on training dataset(s)
NUM_FEATURES    = 0                 # xyz,intensity,return_number
IN_CHANNELS     = 3                 # 3 if xyz only; 4 if with intensity; 5 if add with return_number
OUT_CHANNELS    = TARGET_CLASSES

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

def random_rotate_points(points):
    # Random rotation around z-axis
    theta = np.random.uniform(0, 2 * np.pi)
    rotation_matrix = torch.tensor([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    xyz = points[:, :3]
    rotated_xyz = torch.matmul(xyz, rotation_matrix.T)
    points[:, :3] = rotated_xyz
    return points

def random_jitter(points, sigma=0.01, clip=0.05):
    # Add random noise to coordinates
    jittered_xyz = points[:, :3] + torch.randn_like(points[:, :3]) * sigma
    points[:, :3] = torch.clamp(jittered_xyz, -clip, clip)
    return points

class PointCloudChunkedDataset(Dataset):
    def __init__(self, las_files, n_points=1000, label_mapping=None, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.las_files = las_files
        self.n_points = n_points
        self.n_chunks = 10
        self.point_chunks = []
        self.label_mapping = label_mapping

        for file_idx, las_file in enumerate(las_files):
            las = laspy.read(las_file)
            total_points = len(las.x)
            
            # Add stride to avoid always getting the same points
            stride = total_points // (self.n_chunks * 2)
            for chunk_idx in range(self.n_chunks):
                start_idx = (chunk_idx * stride) % (total_points - n_points)
                self.point_chunks.append((file_idx, start_idx))

        self.pbar = tqdm(total=len(self), desc="Processing Chunks", position=0, leave=True)

    def len(self):
        return len(self.point_chunks)

    def get(self, idx):

        file_idx, start_idx = self.point_chunks[idx]
        las = laspy.read(self.las_files[file_idx])
        
        # Extract features --> TODO Option in constructor
        feature_list = []
        feature_list.append(np.column_stack((las.x, las.y, las.z)))             # xyz
        # feature_list.append(np.array(las.intensity, dtype=np.float32))        # intensity
        # feature_list.append(np.array(las.return_number, dtype=np.float32))    # return_number
        
        features = np.column_stack([feature for feature in feature_list])
        # print(f"features: {len(features)}")
        # indices = random.sample(range(len(features)), min(self.n_points, len(features)))
        
        # Extract labels
        labels = np.array(las.classification, dtype=np.int64)

        # Extract the sequential chunk
        x = torch.tensor(features[start_idx:start_idx + self.n_points], dtype=torch.float)
        y = torch.tensor([self.label_mapping.get(label, -1) for label in labels[start_idx:start_idx + self.n_points]], dtype=torch.long)

        # Apply augmentation during training
        if self.transform:
            x = random_rotate_points(x)
            x = random_jitter(x)

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
        feature_list = []
        feature_list.append(np.column_stack((las.x, las.y, las.z)))             # xyz
        # feature_list.append(np.array(las.intensity, dtype=np.float32))        # intensity
        # feature_list.append(np.array(las.return_number, dtype=np.float32))    # return_number

        features = np.column_stack([feature for feature in feature_list])
        
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

    def get_loss(self, pred, target, criterion):
        """Custom loss function incorporating feature similarity"""
        # Standard cross-entropy loss
        ce_loss = criterion(pred, target)
        
        # Print some statistics about predictions and targets
        if random.random() < 0.01:  # Only print 1% of the time to avoid spam
            print(f"\nLoss value: {ce_loss.item()}")
            print(f"Pred shape: {pred.shape}, Target shape: {target.shape}")
            print(f"Unique targets: {torch.unique(target).cpu().numpy()}")
            print(f"Pred max/min: {pred.max().item()}/{pred.min().item()}")
        
        return ce_loss

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

        # Reduced network capacity
        self.sa1_module = SAModule(0.5, 2, MLP([3 + num_features, 64, 64, 128]))  # More aggressive downsampling
        self.sa2_module = SAModule(0.5, 4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 256, 512]))  # Reduced feature dimension

        self.mlp = MLP([512, 256, 128, num_target_classes], 
                      dropout=0.3,
                      batch_norm=True)

    def forward(self, data):
        pos = data.x[:, :3]
        features = data.x[:, 3:]
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)

        # Process in smaller chunks if needed
        sa0_out = (features, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        
        x, _, _ = sa3_out

        # More memory-efficient repeat
        chunk_size = 1024
        repeated_features = []
        for i in range(0, N_POINTS, chunk_size):
            end = min(i + chunk_size, N_POINTS)
            chunk = x.repeat_interleave(end - i, dim=0)
            repeated_features.append(chunk)
        x = torch.cat(repeated_features, dim=0)
        
        return F.log_softmax(self.mlp(x), dim=-1)

    def get_loss(self, pred, target, criterion):
        """Custom loss function incorporating feature similarity"""
        # Standard cross-entropy loss
        ce_loss = criterion(pred, target)
        
        # Print some statistics about predictions and targets
        if random.random() < 0.01:  # Only print 1% of the time to avoid spam
            print(f"\nLoss value: {ce_loss.item()}")
            print(f"Pred shape: {pred.shape}, Target shape: {target.shape}")
            print(f"Unique targets: {torch.unique(target).cpu().numpy()}")
            print(f"Pred max/min: {pred.max().item()}/{pred.min().item()}")
        
        return ce_loss

class BalancedPointNetPlusPlus(torch.nn.Module):
    def __init__(self, num_features, num_target_classes, points_per_class=1024):
        super().__init__()
        self.points_per_class = points_per_class
        self.num_target_classes = num_target_classes
        
        # Same architecture as PointNet++
        self.sa1_module = SAModule(0.5, 2, MLP([3 + num_features, 64, 64, 128]))
        self.sa2_module = SAModule(0.5, 4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 256, 512]))

        self.mlp = MLP([512, 256, 128, num_target_classes], 
                      dropout=0.3,
                      batch_norm=True)
        
        # Class-wise feature banks to store representative features
        self.feature_banks = {i: [] for i in range(num_target_classes)}
        self.max_bank_size = 1000  # Maximum features to store per class
        
    def balance_batch(self, x, y):
        """Balance the batch by sampling equal points from each class"""
        device = x.device
        balanced_x = []
        balanced_y = []
        
        # Group points by class
        class_indices = {i: [] for i in range(self.num_target_classes)}
        for idx, label in enumerate(y):
            if label >= 0:  # Ignore invalid labels
                class_indices[label.item()].append(idx)
        
        # Filter out empty classes
        active_classes = [c for c in class_indices.keys() if len(class_indices[c]) > 0]
        
        if len(active_classes) < 2:
            # If less than 2 classes in batch, return original to avoid batch norm issues
            return x, y
        
        # Determine number of points to sample per class
        min_points = min(len(indices) for indices in class_indices.values() if len(indices) > 0)
        points_per_class = max(2, min(min_points, self.points_per_class))  # Ensure at least 2 points
        
        # Sample equal numbers of points from each class
        for class_idx in active_classes:
            indices = class_indices[class_idx]
            if len(indices) > 0:
                sampled_indices = torch.tensor(
                    random.sample(indices, points_per_class), 
                    device=device
                )
                balanced_x.append(x[sampled_indices])
                balanced_y.extend([class_idx] * points_per_class)
        
        if len(balanced_x) > 0:
            balanced_x = torch.cat(balanced_x, dim=0)
            balanced_y = torch.tensor(balanced_y, device=device)
            return balanced_x, balanced_y
        return x, y  # Return original if balancing fails

    def update_feature_banks(self, features, labels):
        """Update class-wise feature banks with new features"""
        with torch.no_grad():
            for feat, label in zip(features, labels):
                if label >= 0:  # Ignore invalid labels
                    label = label.item()
                    self.feature_banks[label].append(feat.detach().cpu())
                    # Keep bank size limited
                    if len(self.feature_banks[label]) > self.max_bank_size:
                        self.feature_banks[label].pop(0)

    def forward(self, data):
        pos = data.x[:, :3]
        features = data.x[:, 3:]
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)

        # Balance the input if in training mode
        if self.training:
            pos, data.y = self.balance_batch(pos, data.y)
            features = features[:pos.size(0)]  # Adjust features to match balanced positions
            batch = batch[:pos.size(0)]  # Adjust batch indices
        
        # Process through PointNet++ layers
        sa0_out = (features, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        
        x, _, _ = sa3_out
        
        # Update feature banks during training
        if self.training:
            self.update_feature_banks(x, data.y)
        
        # More memory-efficient repeat for prediction
        if not self.training:
            chunk_size = 1024
            repeated_features = []
            for i in range(0, N_POINTS, chunk_size):
                end = min(i + chunk_size, N_POINTS)
                chunk = x.repeat_interleave(end - i, dim=0)
                repeated_features.append(chunk)
            x = torch.cat(repeated_features, dim=0)
        
        return F.log_softmax(self.mlp(x), dim=-1)

    def get_loss(self, pred, target, criterion):
        """Custom loss function incorporating feature similarity"""
        # Standard cross-entropy loss
        ce_loss = criterion(pred, target)
        
        # Print some statistics about predictions and targets
        if random.random() < 0.01:  # Only print 1% of the time to avoid spam
            print(f"\nLoss value: {ce_loss.item()}")
            print(f"Pred shape: {pred.shape}, Target shape: {target.shape}")
            print(f"Unique targets: {torch.unique(target).cpu().numpy()}")
            print(f"Pred max/min: {pred.max().item()}/{pred.min().item()}")
        
        return ce_loss

def load_model(model_path, in_channels, out_channels, device, model_type='mlp'):
    if model_type == 'mlp':
        model = MLPClassifier(in_channels, out_channels)
    else:  # pointnet++
        model = PointNetPlusPlus(num_features=NUM_FEATURES, num_target_classes=out_channels)
    
    checkpoint = torch.load(model_path, map_location=device)

    # Extract the state_dict from the checkpoint, ignoring other keys like 'model_state_dict' and 'model_type'
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)  # Default to the checkpoint itself if 'model_state_dict' is not found

    model.load_state_dict(model_state_dict)

    # # Try loading the state_dict into the model
    # try:
    #     model.load_state_dict(model_state_dict)
    # except RuntimeError as e:
    #     print(f"Error loading state_dict: {e}")
    #     print("Here are the keys in the saved state_dict:", model_state_dict.keys())
    #     print("Here are the model's keys:", model.state_dict().keys())
    #     raise e
    
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    return model

def classify_point_cloud(model, las_file, device, model_type='mlp'):
    las = laspy.read(las_file)
    
    # Extract features --> TODO Option in constructor
    feature_list = []
    feature_list.append(np.column_stack((las.x, las.y, las.z)))             # xyz
    # feature_list.append(np.array(las.intensity, dtype=np.float32))        # intensity
    # feature_list.append(np.array(las.return_number, dtype=np.float32))    # return_number

    features = np.column_stack([feature for feature in feature_list])

    # Process in batches to avoid memory issues
    predictions = []
    batch_size = 8192 # 2048 32768 --> TODO automatic deduction
    total_points = len(features) # 10_000_000
    total_predicted = 0
    # num_batches = len(features) // batch_size + (1 if len(features) % batch_size != 0 else 0)
    num_batches = (total_points + batch_size - 1) // batch_size 

    for i in tqdm(range(0, len(features), batch_size), total=num_batches, desc="Processing batches", unit="batch"):
        batch = features[i:min(i + batch_size, len(features))]
        x = torch.tensor(batch, dtype=torch.float).to(device)
        # print(f"Batch {i // batch_size + 1}, input  shape: {x.shape}")
        
        if model_type == 'pointnet++':
            batch_idx = torch.zeros(len(batch), dtype=torch.long, device=device)
            data = Data(x=x, batch=batch_idx)
        else:
            data = Data(x=x)
        
        with torch.no_grad():
            out = model(data)
            batch_pred = out.argmax(dim=1).cpu().numpy()
            if len(batch) < len(batch_pred):
                batch_pred = batch_pred[:len(batch)]
            predictions.append(batch_pred)
            total_predicted += batch_pred.size

            # print(f"Batch {i // batch_size + 1}, output shape: {out.shape}")
    
    predicted_labels = np.concatenate(predictions)
    
    if total_predicted != total_points:
        print(f"Warning: Mismatch in the number of predicted labels! Expected {total_points}, got {total_predicted}")

    unique_labels, counts = np.unique(predicted_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        percentage = (count / len(predicted_labels)) * 100
        print(f"Class {label}: {count} points ({percentage:.2f}%)")
    
    return predicted_labels

def save_reclassified_las(las_file, n_points, new_labels, output_file):
    las = laspy.read(las_file)
    las.classification[0:n_points] = new_labels
    las.write(output_file)
    print(f"Updated LAS file saved to: {output_file}")

def main_classify(model_path, las_file_path):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, in_channels=IN_CHANNELS, out_channels=TARGET_CLASSES, device=device, model_type="pointnet++")

    new_labels = classify_point_cloud(model=model, las_file=las_file_path, device=device, model_type="pointnet++")

    save_reclassified_las(las_file_path, 10_000_000, new_labels, "/workspace/reclassified.las")
    print(f"Classified {len(new_labels)} points from {las_file_path}")

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

    folder_name = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + f"_FILES_{len(las_files)}_POINTS_{N_POINTS}_CHUNKS_{dataset.n_chunks}_EPOCHS_{EPOCHS}"
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

    # Use fewer workers for data loading
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, 
                          num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=2, pin_memory=True)

    # Set random seeds for PyTorch
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Choose model type ('mlp' or 'pointnet++')
    model_type = 'pointnet++'  # or 'mlp'

    # Initialize model based on type
    if model_type == 'mlp':
        model = MLPClassifier(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS)
    else:  # pointnet++
        model = PointNetPlusPlus(num_features=NUM_FEATURES, num_target_classes=TARGET_CLASSES)  # 2 features: intensity, return_number
        # model = BalancedPointNetPlusPlus(num_features=NUM_FEATURES, num_target_classes=TARGET_CLASSES)  # 2 features: intensity, return_number
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-5
    )
    
    # Calculate and apply balanced weights (important!)
    total_samples = sum(counts)
    class_weights = torch.zeros(TARGET_CLASSES, device=device)
    for label, count in zip(unique_labels, counts):
        if count > 0:
            class_weights[label] = 1.0  # Set equal weights for all classes
    
    # Use the weights in loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    """
    percent = 0.1
    chunk_abs_count = len(train_data)
    chunk_eff_count = max(1, math.floor(percent * np.float64(chunk_abs_count)) )
    print(f"chunk_abs_count: {chunk_abs_count}")
    print(f"chunk_eff_count: {chunk_eff_count}")
    """

    # Add memory status printing
    def print_memory_status():
        if torch.cuda.is_available():
            print(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"Memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    
    # Print memory status periodically
    print_memory_status()

    def train():
        model.train()
        total_loss = 0
        num_batches = 0
        class_predictions = defaultdict(int)
        progress_bar = tqdm(train_loader, desc="Training", leave=False, position=1)

        # Add gradient accumulation for larger effective batch size
        accumulation_steps = 4
        optimizer.zero_grad()

        for batch_idx, data in enumerate(progress_bar):
            data = data.to(device)
            
            try:
                out = model(data)
                loss = model.get_loss(out, data.y, criterion)
                scaled_loss = loss / accumulation_steps
                scaled_loss.backward()
                
                # Track the actual (unscaled) loss
                total_loss += loss.item()
                num_batches += 1
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Update progress bar with actual loss
                current_avg_loss = total_loss / num_batches
                progress_bar.set_postfix(loss=current_avg_loss)
                
            except RuntimeError as e:
                handle_cuda_error(e)
                continue

        return total_loss / num_batches  # Return average loss per batch

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
            print("True\Pred", end="\n\t")
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

    # Implement warmup learning rate schedule
    warmup_epochs = 5
    def warmup_lr_scheduler(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return scheduler.get_last_lr()[0]
    
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr_scheduler)
    
    with tqdm(total=EPOCHS, desc="Epochs", unit="epoch", position=0) as pbar:
        for epoch in range(EPOCHS):
            train_loss = train()
            val_acc = evaluate(val_loader, epoch)
            
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
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
    plt.figure(figsize=(12, 5))

    # Create two subplots
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS+1), train_losses, label="Train Loss", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS+1), val_accuracies, label="Validation Accuracy", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(folder_dir, "training_plot.png"))
    plt.savefig("training_plot.png")
    plt.close()

    # Also save the raw values for later analysis
    np.save(os.path.join(folder_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(folder_dir, "val_accuracies.npy"), np.array(val_accuracies))

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
    all_preds_inv = np.array([label_mapping_inv.get(pred, -1) for pred in all_preds])
    all_labels_inv = np.array([label_mapping_inv.get(label, -1) for label in all_labels])

    # Optional: Check if any None values are present in the result
    if np.any(all_preds_inv == -1):  # Replace -1 with the missing value or logic to handle it
        print("Warning: Some predictions could not be mapped.")

    unique_preds, pred_counts = np.unique(all_preds_inv, return_counts=True)
    print("Predicted class distribution:", dict(zip(unique_preds, pred_counts)))

    unique_labels, label_counts = np.unique(all_labels_inv, return_counts=True)
    print("True class distribution:", dict(zip(unique_labels, label_counts)))

if __name__ == "__main__":
    
    main()
    
    # # model_path = "/workspace/output/2025-41-02/14/25-10-41_FILES_4_POINTS_2048_CHUNKS_20_EPOCHS_100/pointnet++_EPOCH_99_classifier.pth"
    # model_path = "/workspace/output/2025-02-14 11:05:12_FILES_4_POINTS_2048_CHUNKS_20_EPOCHS_100/pointnet++_EPOCH_99_classifier.pth"
    # # model_path = "/workspace/output/2025-02-14 11:34:15_FILES_5_POINTS_8192_CHUNKS_20_EPOCHS_100/pointnet++_EPOCH_99_classifier.pth"
    # las_file_path = "/workspace/data/general/BA_Schneider/ALS20082012_410000-5745000_LAS12.las"
    # # las_file_path = "/workspace/data/train/a/pc2011_12245200_SUBSAMPLED_2025-02-10_23h34_52_821.las"
    # # las_file_path = "/workspace/data/train/a/pc2011_11245001_SUBSAMPLED_2025-02-10_23h18_33_167.las"
    # main_classify(model_path, las_file_path)
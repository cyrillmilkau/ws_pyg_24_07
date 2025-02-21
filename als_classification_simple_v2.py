import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import PointNetConv, global_max_pool
from torch_geometric.nn import knn_graph
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import laspy
import os
from util.ClassificationLabels import ClassificationLabels
import time

EPOCHS = 100

# Generate synthetic point cloud with 5 classes
def generate_synthetic_cloud(num_points=10000):
    # Create points in 3D space
    points = np.random.randn(num_points, 3)
    labels = np.zeros(num_points, dtype=np.int64)

    # Assign classes based on spatial regions
    for i in range(len(points)):
        x, y, z = points[i]
        # Class 0: Center region
        if x**2 + y**2 + z**2 < 1:
            labels[i] = 0
        # Class 1: Upper region
        elif z > 0.5:
            labels[i] = 1
        # Class 2: Lower region
        elif z < -0.5:
            labels[i] = 2
        # Class 3: Outer ring
        elif x**2 + y**2 > 2:
            labels[i] = 3
        # Class 4: Everything else
        else:
            labels[i] = 4

    return torch.tensor(points, dtype=torch.float), torch.tensor(labels, dtype=torch.long)

# Simple PointNet++ model
class SimplePointNet(nn.Module):
    def __init__(self, num_classes, k=8):
        super().__init__()
        # Create MLPs for the PointNetConv layers
        local_nn1 = nn.Sequential(
            nn.Linear(6, 32),  # 6 = 3 (pos_diff) + 3 (features)
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        local_nn2 = nn.Sequential(
            nn.Linear(67, 64),  # 67 = 3 (pos_diff) + 64 (features)
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        self.k = k
        self.conv1 = PointNetConv(local_nn=local_nn1, global_nn=None)
        self.conv2 = PointNetConv(local_nn=local_nn2, global_nn=None)
        self.classifier = nn.Linear(128, num_classes)  # Dynamic number of classes

    def forward(self, data):
        # Initially, both pos and x are just the 3D coordinates
        pos = data.x  # [num_points, 3] - position coordinates
        x = data.x    # [num_points, 3] - initial features are same as positions
        
        # Create edge connections using k-nearest neighbors
        edge_index = knn_graph(pos, k=self.k, batch=data.batch)
        
        # When PointNetConv processes this:
        # 1. It computes pos_diff between connected points (3D)
        # 2. Concatenates pos_diff with features x
        # So input to local_nn1 is: [pos_diff (3) + features (3) = 6]
        x = F.relu(self.conv1(x, pos, edge_index))  # x is now [num_points, 64]
        
        # For second layer:
        # Input is [pos_diff (3) + features (64) = 67]
        x = F.relu(self.conv2(x, pos, edge_index))  # x is now [num_points, 128]
        
        # Classify each point
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=-1)

class BuildingPointNet(nn.Module):
    def __init__(self, num_classes, k=16, enable_timing=False):
        super().__init__()
        
        # Local feature extraction with all features
        self.local_features = nn.Sequential(
            nn.Linear(9, 64),  # 9 = 3 (pos_diff) + 3 (coords) + 3 (normal)
            # nn.Linear(15, 64),  # 15 = 3 (pos_diff) + 3 (coords) + 3 (normal) + 6 (building features)
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128)
        )
        
        # Second layer
        self.roof_features = nn.Sequential(
            nn.Linear(131, 128),  # 131 = 3 (pos_diff) + 128 (features)
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256)
        )
        
        self.k = k
        self.conv1 = PointNetConv(local_nn=self.local_features, global_nn=None)
        self.conv2 = PointNetConv(local_nn=self.roof_features, global_nn=None)
        self.classifier = nn.Linear(256, num_classes)
        
        # Caching for both normal and building features
        self.cached_normals = {}  # Dictionary to store normals per point
        self.cached_building_features = {}  # Dictionary to store building features per point
        self.enable_timing = enable_timing
        
        # Re-add timing stats
        self.timing_stats = {
            'knn': [],
            'normals': [],
            'conv1': [],
            'conv2': [],
            'classifier': []
        }
        self.total_forward_time = 0
        self.forward_count = 0

    def compute_normal_features(self, pos, edge_index):
        """Compute surface normal features with point-wise caching"""
        if not self.enable_timing:
            return self._compute_normal_features(pos, edge_index)
        
        timing_stats = {}
        start = time.time()
        
        # Get batch information
        batch = getattr(pos, 'batch', None)
        if batch is None:
            batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
        
        # Initialize output normals
        normals = torch.zeros((len(pos), 3), device=pos.device)
        
        # Process each point
        for i in range(len(pos)):
            # Create unique key for this point based on its coordinates
            point_key = tuple(pos[i].cpu().numpy())
            
            # Check if we have cached normal for this point
            if point_key in self.cached_normals:
                normals[i] = self.cached_normals[point_key]
            else:
                # Compute normal for this point
                neighbors_idx = edge_index[1][edge_index[0] == i]
                if len(neighbors_idx) >= 2:
                    v1 = pos[neighbors_idx[0]] - pos[i]
                    v2 = pos[neighbors_idx[1]] - pos[i]
                    normal = torch.cross(v1, v2)
                    if torch.norm(normal) > 0:
                        normal = F.normalize(normal, dim=0)
                    else:
                        normal = torch.tensor([0., 0., 1.], device=pos.device)
                else:
                    normal = torch.tensor([0., 0., 1.], device=pos.device)
                
                # Cache the normal
                self.cached_normals[point_key] = normal
                normals[i] = normal
        
        timing_stats['total_time'] = time.time() - start
        print(f"\nNormal computation time: {timing_stats['total_time']*1000:.2f} ms")
        print(f"Cache size: {len(self.cached_normals)} points")
        
        return normals

    def _compute_normal_features(self, pos, edge_index):
        """Original compute_normal_features without timing"""
        j = edge_index[1]
        i = edge_index[0]
        normals = torch.zeros((len(pos), 3), device=pos.device)
        unique_centers = torch.unique(i)
        
        for center_idx in unique_centers:
            # Create unique key for this point based on its coordinates
            point_key = tuple(pos[center_idx].cpu().numpy())
            
            # Check if we have cached normal for this point
            if point_key in self.cached_normals:
                normals[center_idx] = self.cached_normals[point_key]
            else:
                # Get neighbors for this point
                mask = i == center_idx
                neighbors_idx = j[mask]
                
                if len(neighbors_idx) >= 2:
                    center = pos[center_idx]
                    neighbors = pos[neighbors_idx]
                    v1 = neighbors[0] - center
                    v2 = neighbors[1] - center
                    normal = torch.cross(v1, v2)
                    if torch.norm(normal) > 0:
                        normal = F.normalize(normal, dim=0)
                    else:
                        normal = torch.tensor([0., 0., 1.], device=pos.device)
                else:
                    normal = torch.tensor([0., 0., 1.], device=pos.device)
                
                # Cache the normal
                self.cached_normals[point_key] = normal
                normals[center_idx] = normal
        
        return normals

    def compute_building_features(self, pos, edge_index):
        """Compute features for building detection with timing and caching"""
        if not self.enable_timing:
            return self._compute_building_features(pos, edge_index)
        
        timing_stats = {}
        start_total = time.time()
        
        # Cache lookup timing
        t0 = time.time()
        cache_hits = 0
        cache_misses = 0
        
        j = edge_index[1]
        i = edge_index[0]
        features = torch.zeros((len(pos), 6), device=pos.device)
        timing_stats['initialization'] = time.time() - t0
        
        # Initialize timing counters
        timing_stats['cache_lookup'] = 0
        timing_stats['feature_computation'] = 0
        
        for center_idx in torch.unique(i):
            # Check cache
            t0 = time.time()
            point_key = tuple(pos[center_idx].cpu().numpy())
            if point_key in self.cached_building_features:
                features[center_idx] = self.cached_building_features[point_key]
                cache_hits += 1
                timing_stats['cache_lookup'] += time.time() - t0
                continue
            
            cache_misses += 1
            
            # Compute features if not in cache
            t0 = time.time()
            mask = i == center_idx
            neighbors_idx = j[mask]
            neighbors = pos[neighbors_idx]
            center = pos[center_idx]
            
            # 1. Surface roughness
            t0 = time.time()
            height_diffs = neighbors[:, 2] - center[2]
            roughness = torch.std(height_diffs)
            timing_stats['roughness'] = time.time() - t0
            
            # 3. Planarity
            t0 = time.time()
            centered = neighbors - center
            cov = torch.mm(centered.t(), centered)
            eigenvalues, _ = torch.linalg.eigh(cov)
            planarity = (eigenvalues[1] - eigenvalues[0]) / eigenvalues[2]
            timing_stats['planarity'] = time.time() - t0
            
            # 4. Verticality from normal
            t0 = time.time()
            normal = self.compute_normal_features(pos, edge_index)[center_idx]
            verticality = torch.abs(normal[2])
            timing_stats['normal'] = time.time() - t0
            
            # 5. Height features
            t0 = time.time()
            local_min_height = torch.min(neighbors[:, 2])
            height_above_ground = center[2] - local_min_height
            height_consistency = 1.0 - (torch.max(height_diffs) - torch.min(height_diffs))
            timing_stats['height'] = time.time() - t0
            
            # Combine features
            feature_vector = torch.tensor([
                roughness,
                planarity,
                verticality,
                height_above_ground,
                height_consistency,
                normal[2]
            ], device=pos.device)
            
            # Cache and store
            self.cached_building_features[point_key] = feature_vector
            features[center_idx] = feature_vector
            timing_stats['feature_computation'] += time.time() - t0
        
        total_time = time.time() - start_total
        
        # Print timing and cache statistics
        print("\nBuilding Features Timing and Cache Stats:")
        print("---------------------------------------")
        print(f"Cache hits: {cache_hits}, Cache misses: {cache_misses}")
        print(f"Cache hit rate: {cache_hits/(cache_hits + cache_misses)*100:.1f}%")
        for key, value in timing_stats.items():
            percentage = (value / total_time) * 100
            print(f"{key:20s}: {percentage:5.1f}% ({value*1000:6.2f} ms)")
        print("---------------------------------------")
        print(f"Total time: {total_time*1000:.2f} ms")
        
        return features

    def _compute_building_features(self, pos, edge_index):
        """Original building features computation with caching"""
        j = edge_index[1]
        i = edge_index[0]
        features = torch.zeros((len(pos), 6), device=pos.device)
        
        for center_idx in torch.unique(i):
            # Create unique key for this point based on its coordinates
            point_key = tuple(pos[center_idx].cpu().numpy())
            
            # Check if we have cached features for this point
            if point_key in self.cached_building_features:
                features[center_idx] = self.cached_building_features[point_key]
                continue
            
            # If not in cache, compute features
            mask = i == center_idx
            neighbors_idx = j[mask]
            neighbors = pos[neighbors_idx]
            center = pos[center_idx]
            
            # Compute features
            height_diffs = neighbors[:, 2] - center[2]
            roughness = torch.std(height_diffs)
            
            centered = neighbors - center
            cov = torch.mm(centered.t(), centered)
            eigenvalues, _ = torch.linalg.eigh(cov)
            planarity = (eigenvalues[1] - eigenvalues[0]) / eigenvalues[2]
            
            normal = self.compute_normal_features(pos, edge_index)[center_idx]
            verticality = torch.abs(normal[2])
            
            local_min_height = torch.min(neighbors[:, 2])
            height_above_ground = center[2] - local_min_height
            height_consistency = 1.0 - (torch.max(height_diffs) - torch.min(height_diffs))
            
            # Create feature vector
            feature_vector = torch.tensor([
                roughness,
                planarity,
                verticality,
                height_above_ground,
                height_consistency,
                normal[2]
            ], device=pos.device)
            
            # Cache and store features
            self.cached_building_features[point_key] = feature_vector
            features[center_idx] = feature_vector
        
        return features

    def forward(self, data):
        if not self.enable_timing:
            # Simple forward pass without timing
            pos = data.x
            x = data.x
            edge_index = knn_graph(pos, k=self.k, batch=data.batch)
            
            # Get both normal and building features
            normal_features = self.compute_normal_features(pos, edge_index)
            # building_features = self.compute_building_features(pos, edge_index)
                  
            # Concatenate all features: position + normals + building features
            x = torch.cat([x, normal_features], dim=1)
            # x = torch.cat([x, normal_features, building_features], dim=1)  # Now 12 features total
            
            x = F.relu(self.conv1(x, pos, edge_index))
            x = F.relu(self.conv2(x, pos, edge_index))
            x = self.classifier(x)
            return F.log_softmax(x, dim=-1)
        
        # Forward pass with timing
        start_total = time.time()
        
        pos = data.x
        x = data.x
        
        # kNN computation
        start = time.time()
        edge_index = knn_graph(pos, k=self.k, batch=data.batch)
        self.timing_stats['knn'].append(time.time() - start)
        
        # Normal computation
        start = time.time()
        normal_features = self.compute_normal_features(pos, edge_index)
        x = torch.cat([x, normal_features], dim=1)
        self.timing_stats['normals'].append(time.time() - start)
        
        # First convolution
        start = time.time()
        x = F.relu(self.conv1(x, pos, edge_index))
        self.timing_stats['conv1'].append(time.time() - start)
        
        # Second convolution
        start = time.time()
        x = F.relu(self.conv2(x, pos, edge_index))
        self.timing_stats['conv2'].append(time.time() - start)
        
        # Classifier
        start = time.time()
        x = self.classifier(x)
        x = F.log_softmax(x, dim=-1)
        self.timing_stats['classifier'].append(time.time() - start)
        
        # Print timing stats
        total_time = time.time() - start_total
        self.total_forward_time += total_time
        self.forward_count += 1
        
        if self.forward_count % 2 == 0:
            print("\nTiming Statistics (averaged over last 100 forward passes):")
            print("-----------------------------------------------------")
            avg_total = self.total_forward_time / self.forward_count
            for key in self.timing_stats:
                avg_time = sum(self.timing_stats[key][-100:]) / min(100, len(self.timing_stats[key]))
                percentage = (avg_time / avg_total) * 100
                print(f"{key:12s}: {percentage:5.1f}% ({avg_time*1000:6.2f} ms)")
            print("-----------------------------------------------------")
            print(f"Total time per forward pass: {avg_total*1000:.2f} ms")
            
            # Reset stats
            self.timing_stats = {key: [] for key in self.timing_stats}
            self.total_forward_time = 0
            self.forward_count = 0
        
        return x


def evaluate(model, loader):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            pred = out.max(dim=1)[1]
            predictions.extend(pred.cpu().numpy())
            labels.extend(batch.y.cpu().numpy())
    
    accuracy = accuracy_score(labels, predictions)
    return accuracy

def analyze_model_performance(model, loader, class_names=None):
    """
    Analyze and visualize model performance with class-specific metrics
    """
    # Get predictions and true labels
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            pred = out.max(dim=1)[1]
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Get unique classes in the data
    unique_classes = np.unique(np.concatenate([all_labels, all_preds]))
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(unique_classes))]
    
    # Filter class names to only include classes present in the data
    present_class_names = [class_names[i] for i in unique_classes]
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Get classification report
    report = classification_report(all_labels, all_preds, 
                                 target_names=present_class_names,
                                 labels=unique_classes)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=present_class_names, 
                yticklabels=present_class_names, 
                ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted Class')
    ax1.set_ylabel('True Class')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax1.get_yticklabels(), rotation=0)
    
    # Plot class-wise accuracy
    class_acc = cm.diagonal() / cm.sum(axis=1)
    sns.barplot(x=present_class_names, y=class_acc, ax=ax2)
    ax2.set_title('Class-wise Accuracy')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Return metrics for further analysis if needed
    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'class_accuracy': class_acc,
        'present_classes': unique_classes,
        'present_class_names': present_class_names
    }

def generate_synthetic_cloud_with_distribution(num_points=10000, distribution_type='even'):
    """
    Generate synthetic point cloud with different class distributions
    
    Args:
        num_points: total number of points to generate
        distribution_type: one of ['even', 'extreme_imbalance', 'moderate_imbalance']
    """
    if distribution_type == 'even':
        points_per_class = num_points // 5
        remainder = num_points % 5
        class_points = [points_per_class + (1 if i < remainder else 0) for i in range(5)]
    elif distribution_type == 'extreme_imbalance':
        # 95% class 0, rest share 5%
        points_class_0 = int(0.95 * num_points)
        points_other = (num_points - points_class_0) // 4
        remainder = num_points - points_class_0 - (points_other * 4)
        class_points = [points_class_0] + [points_other + (1 if i < remainder else 0) for i in range(4)]
    elif distribution_type == 'moderate_imbalance':
        # 80% class 0, 19.9% class 1, rest share 0.1%
        points_class_0 = int(0.80 * num_points)
        points_class_1 = int(0.199 * num_points)
        points_other = (num_points - points_class_0 - points_class_1) // 3
        remainder = num_points - points_class_0 - points_class_1 - (points_other * 3)
        class_points = [points_class_0, points_class_1] + [points_other + (1 if i < remainder else 0) for i in range(3)]
    
    # Generate points for each class
    all_points = []
    all_labels = []
    
    for class_idx, num_class_points in enumerate(class_points):
        points = []
        while len(points) < num_class_points:
            # Generate random points
            candidate = np.random.randn(3)
            x, y, z = candidate
            
            # Check if point belongs to current class
            if class_idx == 0 and x**2 + y**2 + z**2 < 1:  # Center region
                points.append(candidate)
            elif class_idx == 1 and z > 0.5:  # Upper region
                points.append(candidate)
            elif class_idx == 2 and z < -0.5:  # Lower region
                points.append(candidate)
            elif class_idx == 3 and x**2 + y**2 > 2:  # Outer ring
                points.append(candidate)
            elif class_idx == 4:  # Everything else
                if not (x**2 + y**2 + z**2 < 1 or z > 0.5 or z < -0.5 or x**2 + y**2 > 2):
                    points.append(candidate)
        
        all_points.extend(points)
        all_labels.extend([class_idx] * num_class_points)
    
    # Shuffle the data
    indices = np.random.permutation(len(all_points))
    all_points = np.array(all_points)[indices]
    all_labels = np.array(all_labels)[indices]
    
    return torch.tensor(all_points, dtype=torch.float), torch.tensor(all_labels, dtype=torch.long)

def save_as_las(points, labels, filename):
    """
    Save point cloud and labels as a LAS file
    
    Args:
        points: numpy array of shape (N, 3) containing point coordinates
        labels: numpy array of shape (N,) containing point classifications
        filename: output LAS file path
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Create LAS file
    las = laspy.create(file_version="1.4", point_format=6)
    
    # Set header
    las.header.scales = [0.001, 0.001, 0.001]
    las.header.offsets = [0, 0, 0]
    
    # Set point coordinates
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    
    # Set classification
    las.classification = labels
    
    # Save file
    las.write(filename)

def create_spatial_chunks(points, labels, chunk_size=100000, overlap=0.1):
    """
    Split large point cloud into overlapping spatial chunks
    
    Args:
        points: [N, 3] tensor of point coordinates
        labels: [N] tensor of point labels
        chunk_size: target number of points per spatial chunk
        overlap: fraction of overlap between adjacent chunks
    """
    # Convert to numpy for easier spatial operations
    points_np = points.numpy()
    
    # Calculate spatial extents
    x_min, y_min, z_min = points_np.min(axis=0)
    x_max, y_max, z_max = points_np.max(axis=0)
    
    # Calculate number of chunks in each dimension
    total_points = len(points)
    chunks_per_dim = int(np.ceil(np.cbrt(total_points / chunk_size)))
    
    # Calculate chunk sizes with overlap
    x_chunk = (x_max - x_min) / chunks_per_dim
    y_chunk = (y_max - y_min) / chunks_per_dim
    z_chunk = (z_max - z_min) / chunks_per_dim
    
    overlap_x = x_chunk * overlap
    overlap_y = y_chunk * overlap
    overlap_z = z_chunk * overlap
    
    batched_data = []
    
    # Create overlapping spatial chunks
    for i in range(chunks_per_dim):
        x_start = x_min + i * x_chunk - (overlap_x if i > 0 else 0)
        x_end = x_min + (i + 1) * x_chunk + (overlap_x if i < chunks_per_dim-1 else 0)
        
        for j in range(chunks_per_dim):
            y_start = y_min + j * y_chunk - (overlap_y if j > 0 else 0)
            y_end = y_min + (j + 1) * y_chunk + (overlap_y if j < chunks_per_dim-1 else 0)
            
            for k in range(chunks_per_dim):
                z_start = z_min + k * z_chunk - (overlap_z if k > 0 else 0)
                z_end = z_min + (k + 1) * z_chunk + (overlap_z if k < chunks_per_dim-1 else 0)
                
                # Get points in this chunk
                mask = ((points_np[:, 0] >= x_start) & (points_np[:, 0] < x_end) &
                       (points_np[:, 1] >= y_start) & (points_np[:, 1] < y_end) &
                       (points_np[:, 2] >= z_start) & (points_np[:, 2] < z_end))
                
                if np.any(mask):
                    chunk_points = points[mask]
                    chunk_labels = labels[mask]
                    
                    # Create Data object for this chunk
                    data = Data(x=chunk_points, y=chunk_labels)
                    data.batch = torch.zeros(len(chunk_points), dtype=torch.long)
                    batched_data.append(data)
    
    return batched_data

def run_experiment(distribution_type):
    """Run the complete experiment for a given distribution type"""
    # Generate data
    train_points, train_labels = generate_synthetic_cloud_with_distribution(num_points=8000, distribution_type=distribution_type)
    val_points, val_labels = generate_synthetic_cloud_with_distribution(num_points=1000, distribution_type=distribution_type)
    test_points, test_labels = generate_synthetic_cloud_with_distribution(num_points=1000, distribution_type=distribution_type)
    
    # Create spatial chunks
    train_chunks = create_spatial_chunks(train_points, train_labels, chunk_size=1000)
    val_chunks = create_spatial_chunks(val_points, val_labels, chunk_size=1000)
    test_chunks = create_spatial_chunks(test_points, test_labels, chunk_size=1000)
    
    # Create data loaders
    train_loader = DataLoader(train_chunks, batch_size=2) # , shuffle=True)
    val_loader = DataLoader(val_chunks, batch_size=2)
    test_loader = DataLoader(test_chunks, batch_size=2)
    
    # Setup model and training
    model = SimplePointNet(num_classes=5, k=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    # Lists to store metrics for plotting
    train_losses = []
    train_accs = []
    val_accs = []

    # Training loop
    best_val_acc = 0.0
    pbar = tqdm(range(100), desc='Training')
    for epoch in pbar:
        # Training
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        train_acc = evaluate(model, train_loader)
        val_acc = evaluate(model, val_loader)
        
        # Store metrics
        train_losses.append(loss.item())
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'train_acc': f'{train_acc:.4f}',
            'val_acc': f'{val_acc:.4f}'
        })

    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    test_acc = evaluate(model, test_loader)
    print(f'\nFinal Test Accuracy: {test_acc:.4f}')

    # Plotting
    epochs = range(len(train_losses))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot loss
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.set_title('Training Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim(0, len(epochs)-1)  # Set x limits from 0 to max epoch
    ax1.set_ylim(0, 1)  # Set y limits from 0 to 1 for loss
    
    # Plot accuracies
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(0, len(epochs)-1)  # Set x limits from 0 to max epoch
    ax2.set_ylim(0, 1)  # Set y limits from 0 to 1 for accuracy
    
    # Adjust layout and save plot
    plt.tight_layout()
    plt.savefig(f'img/training_metrics_{distribution_type}.png')
    plt.close()

    # After training and evaluating on test set, add:
    class_names = [
        'Center Region',
        'Upper Region',
        'Lower Region',
        'Outer Ring',
        'Other'
    ]
    
    analyze_model_performance(model, test_loader, class_names)
    plt.savefig(f'img/class_performance_{distribution_type}.png')
    plt.close()
    
    return test_acc, analyze_model_performance(model, test_loader, class_names)

def load_and_prepare_cloud(las_path, chunk_size=100000):
    """
    Load point cloud from LAS file and prepare it for processing
    
    Args:
        las_path: path to the LAS file
        chunk_size: number of points per spatial chunk
    Returns:
        chunks: list of Data objects containing spatial chunks
    """
    # Load LAS file
    las = laspy.read(las_path)
    
    # Extract point coordinates and classifications
    points = np.vstack((las.x, las.y, las.z)).transpose()
    labels = las.classification
    
    # Normalize coordinates to similar scale as training data
    points_mean = np.mean(points, axis=0)
    points_std = np.std(points, axis=0)
    points_normalized = (points - points_mean) / points_std
    
    # Convert to torch tensors
    points_tensor = torch.tensor(points_normalized, dtype=torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Create spatial chunks
    chunks = create_spatial_chunks(points_tensor, labels_tensor, chunk_size=chunk_size)
    
    return chunks, points_mean, points_std

def plot_inference_accuracy(class_accuracies, class_names, cloud_name):
    """Create bar plot of class-wise accuracy during inference"""
    plt.figure(figsize=(12, 6))
    
    # Create bar plot
    sns.barplot(x=class_names, y=list(class_accuracies.values()))
    
    # Customize plot
    plt.title(f'Class-wise Accuracy for {cloud_name}')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'img/inference_accuracy_{cloud_name}.png')
    plt.close()

def apply_model_to_cloud(model, las_path, output_path, chunk_size=100000, class_mapping=None):
    """
    Apply trained model to a new point cloud and save results
    
    Args:
        model: The trained model
        las_path: Path to input LAS file
        output_path: Path to save classified LAS file
        chunk_size: Size of chunks to process
        class_mapping: Dict mapping predicted classes to output classes
                      e.g. {6: 6, 5: 5, '*': 1} means:
                      - keep class 6 (buildings) as 6
                      - keep class 5 (high veg) as 5
                      - map all other classes (*) to 1 (unclassified)
    """
    timing_stats = {}
    total_start = time.time()
    
    # Get cloud name from path
    cloud_name = os.path.splitext(os.path.basename(las_path))[0]
    

    # Load class definitions
    t0 = time.time()
    las = laspy.read(las_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    # Normalize coordinates
    points_mean = np.mean(points, axis=0)
    points_std = np.std(points, axis=0)
    points_normalized = (points - points_mean) / points_std
    points_tensor = torch.tensor(points_normalized, dtype=torch.float)
    timing_stats['load_data'] = time.time() - t0
    
    # Create chunks without computing features
    t0 = time.time()
    num_points = len(points)
    chunks = []
    for i in range(0, num_points, chunk_size):
        end_idx = min(i + chunk_size, num_points)
        chunk_points = points_tensor[i:end_idx]
        chunk_data = Data(x=chunk_points, pos=chunk_points)
        chunks.append(chunk_data)
    timing_stats['create_chunks'] = time.time() - t0
    
    # Create loader with larger batch size since we're just inferencing
    loader = DataLoader(chunks, batch_size=16)  # Increased batch size
    
    # Make predictions
    t0 = time.time()
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing point cloud"):
            # Forward pass without computing building features
            pos = batch.x
            x = batch.x
            edge_index = knn_graph(pos, k=model.k, batch=batch.batch)
            normal_features = model.compute_normal_features(pos, edge_index)
            x = torch.cat([x, normal_features], dim=1)
            
            x = F.relu(model.conv1(x, pos, edge_index))
            x = F.relu(model.conv2(x, pos, edge_index))
            x = model.classifier(x)
            pred = F.log_softmax(x, dim=-1).max(dim=1)[1]
            predictions.extend(pred.cpu().numpy())
    
    timing_stats['inference'] = time.time() - t0
    
    # Remap classes if mapping is provided
    if class_mapping is not None:
        default_class = class_mapping.get('*', 1)  # Default to unclassified (1)
        remapped_predictions = []
        for pred in predictions:
            remapped_predictions.append(class_mapping.get(int(pred), default_class))
        predictions = remapped_predictions
    
    # Save results
    t0 = time.time()
    las.classification = np.array(predictions)
    las.write(output_path)
    timing_stats['save_results'] = time.time() - t0
    
    # Print timing statistics
    total_time = time.time() - total_start
    print("\nApplication Timing Statistics:")
    print("-----------------------------")
    for key, value in timing_stats.items():
        percentage = (value / total_time) * 100
        print(f"{key:20s}: {percentage:5.1f}% ({value:6.2f} s)")
    print("-----------------------------")
    print(f"Total time: {total_time:.2f} s")
    print(f"Points processed: {len(predictions)}")
    
    return predictions

def run_real(las_path, chunk_size=1000, mod='all', lr=0.01, k=16, overlap=0.1):
    """Run training on real point cloud data"""
    # Load class definitions for the region
    class_def = ClassificationLabels('Vorarlberg')
    
    # Load the point cloud
    print(f"Loading point cloud from: {las_path}")
    las = laspy.read(las_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    labels = las.classification
    
    print(f"Total points: {len(points)}")
    print("Found classes:", [f"{label} ({class_def.get_name_from_label(label)})" 
                           for label in np.unique(labels)])
    
    # Normalize coordinates
    points_mean = np.mean(points, axis=0)
    points_std = np.std(points, axis=0)
    points_normalized = (points - points_mean) / points_std
    
    # Convert to torch tensors
    points_tensor = torch.tensor(points_normalized, dtype=torch.float)
    
    # Get number of unique classes and create mapping
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    print(f"Number of classes: {num_classes}")
    
    # Create class mapping (in case classes are not 0-based consecutive integers)
    class_mapping = {c: i for i, c in enumerate(unique_classes)}
    reverse_mapping = {i: c for c, i in class_mapping.items()}
    print("Class mapping:", {f"{k} ({class_def.get_name_from_label(k)})": v 
                           for k, v in class_mapping.items()})
    
    # Map labels to 0-based consecutive integers
    labels_mapped = np.array([class_mapping[l] for l in labels])
    labels_tensor = torch.tensor(labels_mapped, dtype=torch.long)
    
    # Split data into train, val, test
    num_points = len(points)
    indices = np.random.permutation(num_points)
    train_idx = indices[:int(0.7*num_points)]
    val_idx = indices[int(0.7*num_points):int(0.85*num_points)]
    test_idx = indices[int(0.85*num_points):]
    
    print("\nData split:")
    print(f"Training: {len(train_idx)} points")
    print(f"Validation: {len(val_idx)} points")
    print(f"Test: {len(test_idx)} points")
    
    # Create chunks for each set
    train_chunks = create_spatial_chunks(points_tensor[train_idx], labels_tensor[train_idx], chunk_size=chunk_size, overlap=overlap)
    val_chunks = create_spatial_chunks(points_tensor[val_idx], labels_tensor[val_idx], chunk_size=chunk_size, overlap=overlap)
    test_chunks = create_spatial_chunks(points_tensor[test_idx], labels_tensor[test_idx], chunk_size=chunk_size, overlap=overlap)
    
    print(f"\nCreated chunks of size {chunk_size}:")
    print(f"Training chunks: {len(train_chunks)}")
    print(f"Validation chunks: {len(val_chunks)}")
    print(f"Test chunks: {len(test_chunks)}")
    
    # Create data loaders
    train_loader = DataLoader(train_chunks, batch_size=2)
    val_loader = DataLoader(val_chunks, batch_size=2)
    test_loader = DataLoader(test_chunks, batch_size=2)
    
    # Setup model with correct number of classes
    if mod == 'all':
        model = SimplePointNet(num_classes=num_classes, k=k)
    elif mod == 'building':
        model = BuildingPointNet(num_classes=num_classes, k=k)
    else:
        model = SimplePointNet(num_classes=num_classes, k=k)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    
    # Lists to store metrics for plotting
    train_losses = []
    train_accs = []
    val_accs = []
    
    # Training loop
    best_val_acc = 0.0
    pbar = tqdm(range(100), desc='Training on real point cloud')
    for epoch in pbar:
        # Training
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        train_acc = evaluate(model, train_loader)
        val_acc = evaluate(model, val_loader)
        
        # Store metrics
        train_losses.append(loss.item())
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'train_acc': f'{train_acc:.4f}',
            'val_acc': f'{val_acc:.4f}'
        })
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    test_acc = evaluate(model, test_loader)
    print(f'\nFinal Test Accuracy: {test_acc:.4f}')
    
    # Plot training metrics
    epochs = range(len(train_losses))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot loss
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.set_title('Training Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim(0, len(epochs)-1)
    ax1.set_ylim(0, 1)
    
    # Plot accuracies
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(0, len(epochs)-1)
    ax2.set_ylim(0, 1)
    
    # Adjust layout and save plot
    plt.tight_layout()
    plt.savefig('img/training_metrics_real.png')
    plt.close()
    
    # Analyze and plot class performance
    class_names = [class_def.get_name_from_label(reverse_mapping[i]) 
                  for i in range(len(unique_classes))]
    metrics = analyze_model_performance(model, test_loader, class_names)
    plt.savefig('img/class_performance_real.png')
    plt.close()
    
    # Save the trained model
    torch.save(model.state_dict(), f'trained_model_real_mod_{mod}_feat_all_epochs_{EPOCHS}_lr_{lr}_k_{k}_ov_{overlap}.pt')
    
    print(f"\nModel trained and saved as: trained_model_real.pt")
    
    return test_acc, metrics, model

def main():
    results = {}
    if( False ):  # This will be properly recognized by IDEs
        # First run the training as before
        distributions = ['even', 'extreme_imbalance', 'moderate_imbalance']
        
        for dist_type in distributions:
            print(f"\nRunning experiment with {dist_type} distribution:")
            test_acc, metrics = run_experiment(dist_type)
            results[dist_type] = {
                'test_accuracy': test_acc,
                'metrics': metrics
            }

        # Print comparative results
        print("\nComparative Results:")
        for dist_type, result in results.items():
            print(f"\n{dist_type.replace('_', ' ').title()}:")
            print(f"Test Accuracy: {result['test_accuracy']:.4f}")

    if( False ):
        input_las_path = "/workspace/data/general/pc2011/sub/pc2011_10245101_SUBSAMPLED_100_000.las"
        test_acc, metrics, model = run_real(input_las_path, chunk_size=1000, mod='building', lr=0.001, k=4, overlap=0.1)
        results['real'] = {
                    'test_accuracy': test_acc,
                    'metrics': metrics
                }
        for dist_type, result in results.items():
            print(f"\n{dist_type.replace('_', ' ').title()}:")
            print(f"Test Accuracy: {result['test_accuracy']:.4f}")

    if ( False ):
        # Load the saved model
        # model = SimplePointNet(num_classes=12)  # use same number of classes as training
        model = BuildingPointNet(num_classes=12, k=4)
        model.load_state_dict(torch.load('trained_model_real_mod_building_epochs_100_lr_0.001_k_16_ov_0.4.pt'))

        # Apply to new point cloud
        new_cloud_path = "input_cloud.las"
        output_path = "output_cloud.las"
 
        predictions = apply_model_to_cloud(
            model=model,
            las_path=new_cloud_path,
            output_path=output_path,
            chunk_size=1000
        )

    if ( False ):

        model = BuildingPointNet(num_classes=12, k=4)
        model.load_state_dict(torch.load('trained_model_real_mod_building_feat_all_epochs_100_lr_0.001_k_4_ov_0.1.pt'))

        # Define which classes to keep (example: keep only buildings and high vegetation)
        class_mapping = {
            3: 3,    # Keep low vegetation as class 3
            4: 4,    # Keep medium vegetation as class 4
            5: 5,    # Keep high vegetation as class 5
            6: 6,    # Keep buildings as class 6
            '*': 1   # Map everything else to unclassified (1)
        }

        ncp = [
            # "ALS20082012_410000-5747000_LAS12_section2.las",
            # "ALS20082012_410000-5747000_LAS12_section.las",
        ]

        ocp = [
            # "ALS20082012_410000-5747000_LAS12_section2_classified.las" # "_classified_buildings_veg.las",
            # "ALS20082012_410000-5747000_LAS12_section_classified.las",
        ]

        for i in range(len(ncp)):
            predictions = apply_model_to_cloud(
                model=model,
                las_path=ncp[i],
                output_path=ocp[i],
                chunk_size=1000,
                class_mapping=class_mapping
            )

    if ( True ): 
        model = BuildingPointNet(num_classes=12, k=4)
        model.load_state_dict(torch.load('trained_model_real_mod_building_feat_all_epochs_100_lr_0.001_k_4_ov_0.1.pt'))

        # Base filenames
        base_files = [
            "ALS20082012_410000-5747000_LAS12",
            "ALS20082012_410000-5745000_LAS12",
            "ALS20082012_411000-5745000_LAS12",
        ]

        # Class mapping for the second pass
        class_mapping = {
            3: 3,    # Keep low vegetation as class 3
            4: 4,    # Keep medium vegetation as class 4
            5: 5,    # Keep high vegetation as class 5
            6: 6,    # Keep buildings as class 6
            '*': 1   # Map everything else to unclassified (1)
        }

        # Process each file twice - once without mapping and once with mapping
        for base_file in base_files:
            input_path = f"{base_file}.las"
            
            # First pass - without mapping
            output_path = f"{base_file}_classified.las"
            apply_model_to_cloud(
                model=model,
                las_path=input_path,
                output_path=output_path,
                chunk_size=1000,
                class_mapping=None
            )
            
            # Second pass - with mapping
            output_path_mapped = f"{base_file}_mapped_classified.las"
            apply_model_to_cloud(
                model=model,
                las_path=input_path,
                output_path=output_path_mapped,
                chunk_size=1000,
                class_mapping=class_mapping
            )

    if ( False ):
        model = BuildingPointNet(num_classes=12, k=4)
        model.load_state_dict(torch.load('trained_model_real_mod_building_feat_all_epochs_100_lr_0.001_k_4_ov_0.1.pt'))

        ncp = [
            "ALS20082012_410000-5747000_LAS12_section2.las",
            # "ALS20082012_410000-5747000_LAS12_section.las",
            # "ALS20082012_410000-5745000_LAS12.las",
            # "ALS20082012_411000-5745000_LAS12.las",
        ]

        ocp = [
            "ALS20082012_410000-5747000_LAS12_section2_classified.las" # "_classified_buildings_veg.las",
            # "ALS20082012_410000-5747000_LAS12_section_classified.las",
            # "ALS20082012_410000-5745000_LAS12_classified.las",
            #"ALS20082012_411000-5745000_LAS12_classified.las",
        ]


if __name__ == "__main__":
    main()
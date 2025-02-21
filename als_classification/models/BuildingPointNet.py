import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PointNetConv, knn_graph
import time
import numpy as np

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
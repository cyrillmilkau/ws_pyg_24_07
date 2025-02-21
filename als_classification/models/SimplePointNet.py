import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PointNetConv, global_max_pool
from torch_geometric.nn import knn_graph

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
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, data):
        pos = data.x
        x = data.x
        
        edge_index = knn_graph(pos, k=self.k, batch=data.batch)
        
        x = F.relu(self.conv1(x, pos, edge_index))
        x = F.relu(self.conv2(x, pos, edge_index))
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=-1) 
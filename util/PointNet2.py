# Copyright (c) 2025 Cyrill Milkau
# Licensed under the Apache License, Version 2.0
# See LICENSE file for details.

import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP, PointNetConv, global_max_pool, fps, radius, knn_interpolate

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

class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip

class PN2_Classification(torch.nn.Module):
    def __init__(self, num_features, num_target_classes):
        super().__init__()
        print(f"num_target_classes: {num_target_classes}")
        print(f"num_features: {num_features}")
        # Input channels account for both pos and node features.
        self.sa1_module = SAModule(0.2, 2, MLP([3 + num_features, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 8, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + num_features, 128, 128, 128]))

        # self.mlp = MLP([1024, 512, 256, 128, num_target_classes], dropout=0.5, batch_norm=True)
        self.mlp = MLP([128, 128, 128, num_target_classes], dropout=0.5, batch_norm=True)

    def forward(self, data):
        # print(f"data.x: {data.x.shape}")
        # print(f"data.pos: {data.pos.shape}")
        # with open("tensor_shapes.txt", "a") as f:
        #     f.write(f"data.x: {data.x.shape}\n")
        #     f.write(f"data.pos: {data.pos.shape}\n\n")
        #     f.write(f"--------\n data.x: {data.x}\n\n")

        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        # x, _, _ = sa3_out

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        return F.log_softmax(self.mlp(x), dim=-1)
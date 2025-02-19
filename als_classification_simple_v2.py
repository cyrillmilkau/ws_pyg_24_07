import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import PointNetConv, global_max_pool

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
    def __init__(self):
        super().__init__()
        self.conv1 = PointNetConv(3, 64)
        self.conv2 = PointNetConv(64, 128)
        self.classifier = nn.Linear(128, 5)  # 5 classes

    def forward(self, data):
        x, pos = data.x, data.x
        # Apply PointNet++ layers
        x = F.relu(self.conv1(x, pos))
        x = F.relu(self.conv2(x, pos))
        # Classify each point
        x = self.classifier(x)
        return F.log_softmax(x, dim=-1)

def main():
    # Generate data
    points, labels = generate_synthetic_cloud()
    data = Data(x=points, y=labels)
    loader = DataLoader([data], batch_size=1)

    # Setup model and training
    model = SimplePointNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    # Training loop
    for epoch in range(50):
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

if __name__ == "__main__":
    main()
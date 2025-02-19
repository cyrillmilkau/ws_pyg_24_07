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
        
        self.conv1 = PointNetConv(local_nn=local_nn1, global_nn=None)
        self.conv2 = PointNetConv(local_nn=local_nn2, global_nn=None)
        self.classifier = nn.Linear(128, 5)  # 5 classes

    def forward(self, data):
        # Initially, both pos and x are just the 3D coordinates
        pos = data.x  # [num_points, 3] - position coordinates
        x = data.x    # [num_points, 3] - initial features are same as positions
        
        # Create edge connections using k-nearest neighbors
        edge_index = knn_graph(pos, k=8, batch=data.batch)
        
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
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    
    # Plot class-wise accuracy
    class_acc = cm.diagonal() / cm.sum(axis=1)
    sns.barplot(x=present_class_names, y=class_acc, ax=ax2)
    ax2.set_title('Class-wise Accuracy')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Adjust layout and save plot
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

def run_experiment(distribution_type):
    """Run the complete experiment for a given distribution type"""
    # Generate data
    train_points, train_labels = generate_synthetic_cloud_with_distribution(num_points=8000, distribution_type=distribution_type)
    val_points, val_labels = generate_synthetic_cloud_with_distribution(num_points=1000, distribution_type=distribution_type)
    test_points, test_labels = generate_synthetic_cloud_with_distribution(num_points=1000, distribution_type=distribution_type)
    
    # Save point clouds as LAS files
    save_as_las(train_points.numpy(), train_labels.numpy(), 
                f'data/{distribution_type}/train.las')
    save_as_las(val_points.numpy(), val_labels.numpy(), 
                f'data/{distribution_type}/val.las')
    save_as_las(test_points.numpy(), test_labels.numpy(), 
                f'data/{distribution_type}/test.las')
    
    # Create data objects
    train_data = Data(x=train_points, y=train_labels)
    val_data = Data(x=val_points, y=val_labels)
    test_data = Data(x=test_points, y=test_labels)
    
    # Add batch information
    train_data.batch = torch.zeros(train_points.size(0), dtype=torch.long)
    val_data.batch = torch.zeros(val_points.size(0), dtype=torch.long)
    test_data.batch = torch.zeros(test_points.size(0), dtype=torch.long)
    
    # Create data loaders
    train_loader = DataLoader([train_data], batch_size=1)
    val_loader = DataLoader([val_data], batch_size=1)
    test_loader = DataLoader([test_data], batch_size=1)

    # Setup model and training
    model = SimplePointNet()
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
    ax1.set_ylim(0, 10)  # Set y limits from 0 to 10 for loss
    
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

def main():
    # Run experiments for each distribution type
    distributions = ['even', 'extreme_imbalance', 'moderate_imbalance']
    results = {}
    
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

if __name__ == "__main__":
    main()
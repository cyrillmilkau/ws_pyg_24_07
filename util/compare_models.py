import torch
import os
import csv
from itertools import combinations

# Directory containing model files
model_dir = "/workspace/output/00-29_FILES_16_POINTS_32768_CHUNKS_20_EPOCHS_1000/"  # Adjust as needed
model_paths = sorted(os.listdir(model_dir))

# Load models into a dictionary
models = {}
for path in model_paths:
    model_data = torch.load(os.path.join(model_dir, path), map_location="cpu")
    models[path] = model_data["model_state_dict"]  # Extract only the state_dict

# Extract all unique parameter names
param_names = set()
for model_state in models.values():
    param_names.update(model_state.keys())  # Extract layer names
param_names = sorted(param_names)  # Sort for consistency

# Write to CSV
csv_file = os.path.join("/workspace/", "model_parameters.csv")
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Model"] + param_names)  # Column headers
    
    for model_name, model_state in models.items():
        row = [model_name]
        for param in param_names:
            if param in model_state:
                # Get the first entry of the parameter
                first_entry = model_state[param].data.view(-1)[0].item()
                row.append(first_entry)  # Store first entry of parameter
            else:
                row.append("N/A")  # If a model lacks a parameter
        writer.writerow(row)

print(f"Saved model parameters to {csv_file}")
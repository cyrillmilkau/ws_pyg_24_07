# ALS Classification Scripts Overview

This repository contains three iterations of ALS (Airborne Laser Scanning) point cloud classification scripts, showing the evolution from a complex initial approach to a streamlined working solution.

## Script Versions

### 1. als_classification.py
- Initial complex implementation
- Contains extensive preprocessing and data handling
- Multiple classification approaches implemented
- Not functional due to complexity and integration issues
- Serves as a reference for comprehensive point cloud processing concepts

### 2. als_classification_simple.py
- First simplified version
- Stripped down to essential classification functionality
- Working implementation but produces basic results
- Serves as proof-of-concept
- Limited practical application due to simplified approach

### 3. als_classification_simple_v2.py
- Current working implementation
- Optimized and refactored code
- Produces reliable classification results
- Features:
  - Efficient file handling
  - Two-pass classification (raw and mapped)
  - Configurable class mapping
  - Chunk-based processing for large point clouds
  - Memory-efficient implementation

## Usage

### 0. Clone Repository

```
git clone https://kis5.geoinformation.htw-dresden.de/gitlab/blika389/als_classification.git
```

### 1. Download Docker

https://www.docker.com/products/docker-desktop/

### 2. Run PyG Docker Container

https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pyg e.g. 

```
docker run --gpus all -it -v D:/src/als_classification:/workspace nvcr.io/nvidia/pyg:24.07-py3
```

### 3. Install package and run

```
pip install -e .
```
or
```
pip install -e .[dev]
```

then run via

```
python -m als_classification.als_classification
```

### IV (Optional:) Attach via VS Code

https://code.visualstudio.com/docs/devcontainers/attach-container

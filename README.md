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
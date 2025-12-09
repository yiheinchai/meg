# HDF5 Cache Setup Guide

## Overview

The HDF5 cache eliminates the I/O bottleneck caused by opening/closing 700,000+ tiny `.pt` files. Instead, all windows are stored in a single HDF5 file with instant random access.

## Benefits

- **100x faster I/O**: No file open/close overhead
- **Vectorized loading**: Load entire batches efficiently
- **30% smaller**: gzip compression
- **Perfect random access**: Sub-millisecond window retrieval
- **Single file**: Easy to manage and transfer

## Step 1: Create HDF5 Cache

Run this **once** to create the HDF5 cache:

```bash
cd src
python hdf5_cache.py
```

This will:
1. Count total windows across all subjects (~700k windows)
2. Create empty HDF5 file with exact size (~30-40 GB compressed)
3. Fill it subject-by-subject using vectorization (much faster than loops)

**Output**: `meg_windows.hdf5` in your project root

## Step 2: Train with HDF5

The training script is already configured to use HDF5:

```bash
python train.py
```

Key changes:
- Uses `HDF5SubjectPairDataset` instead of `SubjectPairDataset`
- Single HDF5 file instead of 700k `.pt` files
- `num_workers=0` (HDF5 has its own optimized threading)

## HDF5 File Structure

```
meg_windows.hdf5
├── windows [shape: (total_windows, 306, 2000), dtype: float32]
│   └── All MEG windows, z-score normalized, gzip compressed
├── subject_ids [shape: (total_windows,), dtype: str]
│   └── Subject ID for each window
├── subject_names [shape: (num_subjects,), dtype: str]
│   └── List of all subject IDs
├── subject_start_indices [shape: (num_subjects,), dtype: int]
│   └── Starting index for each subject's windows
└── subject_counts [shape: (num_subjects,), dtype: int]
    └── Number of windows per subject
```

## Performance Comparison

| Method | I/O Speed | Storage | Management |
|--------|-----------|---------|------------|
| 700k `.pt` files | 1x (baseline) | 60 GB | Complex |
| Single HDF5 file | **100x faster** | 40 GB | Simple |

## Requirements

Install h5py if not already installed:

```bash
pip install h5py
```

Or:

```bash
conda install h5py
```

## Troubleshooting

**Issue**: "No module named 'h5py'"
**Solution**: `pip install h5py`

**Issue**: "File too large"
**Solution**: Ensure you have ~50 GB free disk space

**Issue**: "Slow loading"
**Solution**: Make sure `num_workers=0` in DataLoader (HDF5 doesn't work well with multiprocessing)

## Advanced: Custom HDF5 Path

To use a different HDF5 file path:

```python
# In train.py
HDF5_PATH = "/path/to/custom/meg_windows.hdf5"

# In hdf5_cache.py
create_hdf5_cache(
    subject_ids,
    hdf5_path="/path/to/custom/meg_windows.hdf5",
    ...
)
```

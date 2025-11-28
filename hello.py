#|default_exp train
#|export

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import mne
import warnings
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
from tqdm.notebook import tqdm
from IPython.display import clear_output, display
import umap

warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Define dataset paths
base_path = Path('D:/CamCAN/cc700/meg/pipeline/release005/BIDSsep/rest')

# List files in the base directory
files = list(base_path.iterdir())
print("Files in the directory:")
for file in files:
    print(file.name)


rest_path = base_path / 'rest'
noise_path = base_path / 'noise'

print("Dataset paths configured successfully!")


#|export

# Load real MEG data from CamCAN dataset
def load_meg_data(subject_id, base_path=base_path):
    """
    Load and preprocess MEG data for a given subject.
    
    Parameters:
    -----------
    subject_id : str
        Subject ID (e.g., 'CC110033')
    base_path : Path
        Base path to the dataset
        
    Returns:
    --------
    data : ndarray
        Preprocessed MEG data of shape (n_channels, n_samples)
    """
    # Construct file path
    meg_file = base_path / f'{subject_id}' / 'meg' / f'{subject_id}_task-rest_meg.fif'
    
    if not meg_file.exists():
        raise FileNotFoundError(f"MEG file not found: {meg_file}")
    
    # Load MEG data using MNE
    raw = mne.io.read_raw_fif(meg_file, preload=True, verbose=False)
    
    # Keep only magnetometers for simplicity (102 channels)
    raw.pick_types(meg=True)
    
    # # Apply basic preprocessing
    # raw.filter(l_freq=1, h_freq=45, verbose=False)  # Band-pass filter
    # raw.notch_filter([50, 100], verbose=False)  # Remove line noise
    
    # Get data as numpy array
    data = raw.get_data()  # Shape: (n_channels, n_samples)
    
    return torch.from_numpy(data)

participants_df = torch
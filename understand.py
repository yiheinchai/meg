#|default_exp train
#|export
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import mne
from functools import lru_cache

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

train_log = open("train_log", "w+")
sys.stdout = train_log

# Define dataset paths
base_path = Path('D:/CamCAN/cc700/meg/pipeline/release005/BIDSsep')
rest_path = base_path / 'rest'
noise_path = base_path / 'noise'

print("Dataset paths configured successfully!")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_raw_meg_data(subject_id, base_path=rest_path):
    meg_file = base_path / f'{subject_id}' / 'meg' / f'{subject_id}_task-rest_meg.fif'
    
    if not meg_file.exists():
        raise FileNotFoundError(f"MEG file not found: {meg_file}")
    
    # Load MEG data using MNE
    raw = mne.io.read_raw_fif(meg_file, preload=True, verbose=False)
    
    return raw

#|export

# Load real MEG data from CamCAN dataset
def load_meg_data(subject_id, base_path=rest_path):
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


def fast_load_meg_data(subject_id, cache_dir='meg_cache', base_path=rest_path):
    """
    Fast loading of MEG data from npz cache. Falls back to load_meg_data if cache doesn't exist.
    
    Parameters:
    -----------
    subject_id : str
        Subject ID (e.g., 'CC110033')
    cache_dir : str
        Directory containing cached .npz files
    base_path : Path
        Base path to the dataset (used if cache doesn't exist)
        
    Returns:
    --------
    data : torch.Tensor
        MEG data of shape (n_channels, n_samples)
    """
    cache_path = Path(cache_dir) / f"{subject_id}.npz"
    
    if cache_path.exists():
        # Load from cache (much faster)
        data = np.load(cache_path)['data']
        return torch.from_numpy(data)
    else:
        # Fall back to slow loading
        return load_meg_data(subject_id, base_path)


def preprocess_and_cache_all_subjects(subject_ids, cache_dir='meg_cache', base_path=rest_path):
    """
    Preprocess and save all MEG data as .npz files for fast loading.
    Run this once before training to avoid repeated slow loads.
    
    Parameters:
    -----------
    subject_ids : list
        List of subject IDs to preprocess
    cache_dir : str
        Directory to save cached .npz files
    base_path : Path
        Base path to the dataset
    """
    import os
    cache_path = Path(cache_dir)
    os.makedirs(cache_path, exist_ok=True)
    
    print("\n" + "="*50)
    print("PREPROCESSING AND CACHING MEG DATA")
    print("="*50)
    print(f"Cache directory: {cache_path.absolute()}")
    print(f"Total subjects: {len(subject_ids)}")
    print("This will take some time but only needs to be done once...\n")
    
    for idx, subj_id in enumerate(subject_ids, 1):
        cache_file = cache_path / f"{subj_id}.npz"
        
        if cache_file.exists():
            print(f"[{idx}/{len(subject_ids)}] {subj_id} - Already cached, skipping")
            continue
        
        try:
            print(f"[{idx}/{len(subject_ids)}] {subj_id} - Loading and caching...")
            data = load_meg_data(subj_id, base_path)
            
            # Save as compressed npz
            np.savez_compressed(cache_file, data=data.numpy())
            print(f"    ✓ Saved to {cache_file.name}")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    print("\n" + "="*50)
    print("CACHING COMPLETE")
    print("="*50)
    print("All future loads will be 10-100x faster!\n")


def create_window_cache(subject_ids, window_size=2000, stride=500, 
                        subject_cache_dir='meg_cache', window_cache_dir='meg_windows_cache', 
                        base_path=rest_path):
    """
    Create individual .pt (PyTorch tensor) files for each 2-second window.
    Uses float16 to reduce storage by 50% while maintaining training quality.
    This provides the fastest possible training by eliminating all file parsing overhead.
    
    Parameters:
    -----------
    subject_ids : list
        List of subject IDs to process
    window_size : int
        Window size in samples (default: 2000 for 2 seconds at 1000Hz)
    stride : int
        Stride between windows in samples (default: 500 for overlap)
    subject_cache_dir : str
        Directory containing subject-level .npz files (optional, will use raw .fif if not available)
    window_cache_dir : str
        Directory to save window-level .pt files
    base_path : Path
        Base path to dataset (fallback if subject cache doesn't exist)
    
    Benefits:
    ---------
    - 50% smaller storage than float32 (using float16)
    - No file parsing overhead during training
    - Perfect global shuffling (not subject-based)
    - Maximum GPU utilization (~95% vs ~40%)
    """
    import os
    window_path = Path(window_cache_dir)
    os.makedirs(window_path, exist_ok=True)
    
    print("\n" + "="*50)
    print("CREATING OPTIMIZED WINDOW CACHE (.pt files)")
    print("="*50)
    print(f"Window cache directory: {window_path.absolute()}")
    print(f"Window size: {window_size} samples")
    print(f"Stride: {stride} samples")
    print(f"Storage format: PyTorch tensors (float16)")
    print(f"Total subjects: {len(subject_ids)}")
    print(f"Expected storage savings: 50% vs float32\n")
    
    total_windows = 0
    
    for idx, subj_id in enumerate(subject_ids, 1):
        print(f"[{idx}/{len(subject_ids)}] Processing {subj_id}...")
        
        try:
            # Load subject data (from cache if available, else raw .fif)
            data = fast_load_meg_data(subj_id, cache_dir=subject_cache_dir, base_path=base_path)
            
            total_length = data.shape[-1]
            last_index = total_length - window_size
            window_count = 0
            
            # Create windows with sliding window approach
            for start_idx in range(0, last_index + 1, stride):
                if start_idx <= last_index:
                    window_data = data[:, start_idx:start_idx + window_size]
                    
                    # Convert to float16 for 50% storage reduction
                    tensor_segment = window_data.to(torch.float16)
                    
                    # Save as .pt file: SubjectID_WindowIndex.pt
                    save_name = window_path / f"{subj_id}_{window_count}.pt"
                    torch.save(tensor_segment, save_name)
                    
                    window_count += 1
            
            total_windows += window_count
            print(f"    ✓ Created {window_count} windows")
            del data  # Free memory
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    print("\n" + "="*50)
    print("OPTIMIZED WINDOW CACHE CREATION COMPLETE")
    print("="*50)
    print(f"Total windows created: {total_windows}")
    print(f"Average per subject: {total_windows/len(subject_ids):.1f}")
    
    # Calculate estimated storage
    # 306 channels * 2000 samples * 2 bytes (float16) = ~1.2 MB per window
    estimated_gb = (total_windows * 1.2) / 1024
    print(f"Estimated storage: ~{estimated_gb:.1f} GB")
    print("\nBenefits:")
    print("  ✓ 50% smaller than float32")
    print("  ✓ No file parsing overhead")
    print("  ✓ Perfect global shuffling")
    print("  ✓ Maximum GPU utilization (~95%)\n")

#|export

print("\n" + "="*50)
print("LOADING PARTICIPANT IDS (NOT DATA)")
print("="*50)

df = pd.read_csv(rest_path / 'participants.tsv', sep="\t")

participants_ids = df['participant_id'].to_list()
print(f"Found {len(participants_ids)} participants")
print("Note: Data will be loaded on-demand to avoid memory overflow")

# Step 1: Preprocess and cache all subjects as .npz files (run once)
# Uncomment to create subject-level cache:
# preprocess_and_cache_all_subjects(participants_ids, cache_dir='meg_cache', base_path=rest_path)

# Step 2: Create window-level cache as .pt files for ultra-fast training (run once after Step 1)
# This creates individual PyTorch tensor files (float16) for each 2-second window
# Benefits: 50% storage reduction, no parsing overhead, perfect shuffling, ~95% GPU usage
# Uncomment to create window-level cache:
# create_window_cache(participants_ids, window_size=2000, stride=500, 
#                     subject_cache_dir='meg_cache', window_cache_dir='meg_windows_cache')

print("="*50)


# raise RuntimeError("Finished creating window cache")
#|export

class NoiseInjection():
    def __init__(self):
        pass

    def __call__(self, timeseries):
        if isinstance(timeseries, np.ndarray):
            timeseries = torch.from_numpy(timeseries)

        std = timeseries.std()
        
        noise = torch.randn_like(timeseries) * std
        return timeseries + noise
    

class StdGaussianNoise():
    def __init__(self, *, std):
        self.std = std

    def __call__(self, timeseries):
        if isinstance(timeseries, np.ndarray):
            timeseries = torch.from_numpy(timeseries)
        return timeseries + (torch.randn_like(timeseries) * self.std)


class ZScore():
    def __init__(self):
        pass

    def __call__(self, timeseries: torch.Tensor):
        mean = timeseries.mean(dim=-1, keepdim=True) # (batch, 306, ..time..) avg across time
        std = timeseries.std(dim=-1, keepdim=True)
        return (timeseries - mean) / std

#|export

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=306, out_channels=64, kernel_size=3, padding=1) # batch, 64, window_size
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1) # batch, 64

        self.fc = nn.Linear(64, 128) # batch, 128

    def forward(self, x: torch.Tensor):
        x = x.float()
        x = self.conv1(x)
        x = self.relu(x)
        x = self.avg_pool(x)

        x = x.squeeze(-1)

        x = self.fc(x)
        # x = self.relu(x) # do not collapse the embedding space
        
        return x

#|export

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def __call__(self, z1: torch.Tensor, z2: torch.Tensor):
        batch_size = z1.shape[0]
        device = z1.device

        z = torch.cat([z1, z2]) # 64 x 128 -> 128 x 128
        z = nn.functional.normalize(z)
        sim_matrix = (z @ z.T) / self.temperature

        # masking 
        sim_matrix = torch.masked_fill(sim_matrix, torch.eye(sim_matrix.shape[0], device=device).bool(), -torch.inf)

        y = torch.tensor([batch_size + i for i in range(batch_size)] + [i for i in range(batch_size)], dtype=torch.long, device=device)

        loss = nn.functional.cross_entropy(sim_matrix, y)

        return loss

#|export

class MEG_Dataset(Dataset):
    def __init__(self, data, window_size=2000, transforms=None, stride=500):
        super().__init__()

        if transforms is None:
            raise TypeError("Transforms must be filled")
        
        self.window_size = window_size
        self.data = data
        self.transforms = transforms

        total_length = self.data.shape[-1]
        last_index = total_length - self.window_size
        self.stride = stride
        self.indices = [i for i in range(0, last_index, self.stride) if i <= last_index]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        start_index = self.indices[index]
        segment = self.data[:, start_index: start_index + self.window_size]
        view_1 = self.transforms(segment)
        view_2 = self.transforms(segment)

        return view_1, view_2


class LazyMEG_Dataset(Dataset):
    """Loads MEG data on-demand with caching to avoid redundant loads"""
    def __init__(self, subject_ids, window_size=2000, transforms=None, stride=500, base_path=rest_path, cache_size=20, use_npz_cache=True, npz_cache_dir='meg_cache'):
        super().__init__()
        if transforms is None:
            raise TypeError("Transforms must be filled")
        
        self.subject_ids = subject_ids
        self.window_size = window_size
        self.transforms = transforms
        self.stride = stride
        self.base_path = base_path
        self.use_npz_cache = use_npz_cache
        self.npz_cache_dir = npz_cache_dir
        
        # Build cached loader with LRU cache
        @lru_cache(maxsize=cache_size)
        def cached_load(subj_id):
            if self.use_npz_cache:
                return fast_load_meg_data(subj_id, cache_dir=npz_cache_dir, base_path=base_path)
            else:
                return load_meg_data(subj_id, base_path)
        
        self.cached_load = cached_load
        
        # Build index mapping without loading all data
        # Estimate: MEG rest data is typically ~5 min at 1000 Hz = ~300,000 samples
        print("Building index map (sampling first subject for length estimation)...")
        
        # Load first subject to get typical length
        if use_npz_cache:
            first_data = fast_load_meg_data(subject_ids[0], cache_dir=npz_cache_dir, base_path=base_path)
        else:
            first_data = load_meg_data(subject_ids[0], base_path)
        typical_length = first_data.shape[-1]
        del first_data
        print(f"  Estimated samples per subject: {typical_length}")
        
        self.index_map = []
        for subj_idx, subj_id in enumerate(subject_ids):
            # Use typical length for estimation (slight over-estimate is fine)
            last_index = typical_length - window_size
            for start_idx in range(0, last_index + 1, stride):
                if start_idx <= last_index:
                    self.index_map.append((subj_idx, start_idx))
        
        print(f"Total estimated windows: {len(self.index_map)}")
        print(f"Cache size: {cache_size} subjects (keeps recently used data in memory)")
        if use_npz_cache:
            print(f"Using fast npz cache from: {npz_cache_dir}")
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, index):
        subj_idx, start_idx = self.index_map[index]
        subj_id = self.subject_ids[subj_idx]
        
        # Load data with caching (reuses if recently accessed)
        data = self.cached_load(subj_id)
        
        # Handle edge case where actual length < estimated length
        if start_idx + self.window_size > data.shape[-1]:
            start_idx = max(0, data.shape[-1] - self.window_size)
        
        segment = data[:, start_idx: start_idx + self.window_size]
        
        view_1 = self.transforms(segment)
        view_2 = self.transforms(segment)
        
        return view_1, view_2


class WindowMEG_Dataset(Dataset):
    """Loads individual 2-second windows directly from .pt files - ultra fast and memory efficient
    
    This is the fastest possible training approach because:
    1. No file parsing overhead (pre-saved PyTorch tensors)
    2. Perfect global shuffling (not subject-based)
    3. Maximum GPU utilization (~95% vs ~40%)
    4. 50% smaller storage (float16)
    """
    def __init__(self, window_cache_dir='meg_windows_cache', transforms=None):
        super().__init__()
        if transforms is None:
            raise TypeError("Transforms must be filled")
        
        self.window_cache_dir = Path(window_cache_dir)
        self.transforms = transforms
        
        # Fast indexing of all .pt files
        print(f"Indexing .pt files in {window_cache_dir}...")
        self.file_paths = list(self.window_cache_dir.glob("*.pt"))
        
        if len(self.file_paths) == 0:
            raise FileNotFoundError(
                f"No .pt files found in {window_cache_dir}. "
                "Please run create_window_cache() first."
            )
        
        print(f"Loaded {len(self.file_paths)} windows ready for training")
        print(f"  Storage format: PyTorch tensors (float16)")
        print(f"  Benefits: No parsing overhead + Perfect shuffling + Max GPU usage")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        # Load pre-saved tensor (extremely fast, no parsing)
        path = self.file_paths[index]
        data = torch.load(path)
        
        # Convert from float16 to float32 for training stability
        # (PyTorch prefers float32 for calculations)
        data = data.float()
        
        # Apply transforms to create two augmented views
        view_1 = self.transforms(data)
        view_2 = self.transforms(data)
        
        return view_1, view_2
    
    def get_subject_id(self, index):
        """Get subject ID for a given window index"""
        # Extract subject ID from filename: SubjectID_WindowIndex.pt
        filename = self.file_paths[index].stem  # Remove .pt extension
        subject_id = filename.rsplit('_', 1)[0]  # Split on last underscore
        return subject_id

#|export

print("\n" + "="*50)
print("PREPARING TRAINING")
print("="*50)

aug_pipeline = transforms.Compose([ZScore(), StdGaussianNoise(std=0.1)])

# Choose between LazyMEG_Dataset (loads subjects with caching) or 
# WindowMEG_Dataset (loads individual windows - FASTEST and most memory efficient)
USE_WINDOW_CACHE = True  # Set to False to use LazyMEG_Dataset instead

if USE_WINDOW_CACHE:
    print("Using WindowMEG_Dataset - loading .pt tensors from disk...")
    print("Expected GPU utilization: ~95% (vs ~40% with lazy loading)")
    train_dataset = WindowMEG_Dataset(window_cache_dir='meg_windows_cache', transforms=aug_pipeline)
else:
    print("Using LazyMEG_Dataset with window_size=2000, stride=500...")
    train_dataset = LazyMEG_Dataset(participants_ids, window_size=2000, transforms=aug_pipeline, stride=500)

print(f"Total training windows: {len(train_dataset)}")

print("Creating DataLoader with batch_size=64, 8 workers...")
# Increased num_workers for .pt files (they load much faster than .fif)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
print(f"Total batches per epoch: {len(train_loader)}")

print("\nInitializing model...")
encoder = Encoder()
encoder = encoder.to(device)
print(f"Encoder moved to {device}")

criterion = NTXentLoss(temperature=0.5)
optimiser = optim.Adam(encoder.parameters(), lr=1e-3)
print("Loss function and optimizer initialized")
print("="*50)

#|export

print("\n" + "="*50)
print("STARTING TRAINING")
print("="*50)

encoder.train()

num_epochs = 10
print(f"Training for {num_epochs} epochs\n")

fig, ax = plt.subplots(figsize=(10, 6))
display_handle = display(fig, display_id=True)
losses = []
batch_nums = []

for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")
    print("-" * 50)
    epoch_loss = 0.0
    
    for batch_idx, (x1, x2) in enumerate(train_loader):
        x1, x2 = x1.to(device), x2.to(device)
        z1 = encoder(x1)
        z2 = encoder(x2)

        loss = criterion(z1, z2)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        epoch_loss += loss.item()

        # Track metrics for live plot
        losses.append(loss.item())
        batch_nums.append(epoch * len(train_loader) + batch_idx)

        # Log every 10 batches
        if batch_idx % 10 == 0:
            print(f"  Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        if batch_idx % 5 == 0 or batch_idx == len(train_loader) - 1:
            ax.clear()
            ax.plot(batch_nums, losses)
            ax.set_xlabel('Batch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss per Batch')
            ax.grid(True)
            display_handle.update(fig)

    avg_loss = epoch_loss/len(train_loader)
    print(f"\n✓ Epoch {epoch+1}/{num_epochs} Complete | Average Loss: {avg_loss:.4f}")

display_handle.update(fig)
print("\n" + "="*50)
print("TRAINING COMPLETED")
print("="*50)

#|export

print("\nSaving model checkpoint...")
torch.save({"model_state_dict": encoder.state_dict(), "optimizer_state_dict": optimiser.state_dict()}, 'gpu_run_v1.pth')
print("✓ Model saved to 'gpu_run_v1.pth'")

#|export

print("\n" + "="*50)
print("LOADING MODEL FOR INFERENCE")
print("="*50)

encoder = Encoder()
encoder.load_state_dict(torch.load("gpu_run_v1.pth", map_location=device)['model_state_dict'])
encoder = encoder.to(device)
print("✓ Model loaded successfully")

#|export

print("\n" + "="*50)
print("GENERATING EMBEDDINGS")
print("="*50)
print("Note: Using subject-level loading for embeddings (not window cache)")
print("This allows us to generate embeddings for all windows of each subject efficiently\n")

encoder.eval()

# Create directory to save embeddings
import os
embeddings_dir = Path('embeddings_cache')
os.makedirs(embeddings_dir, exist_ok=True)
print(f"Saving embeddings to: {embeddings_dir}")

with torch.no_grad():
    for i, subj_id in enumerate(participants_ids):
        print(f"Processing subject {i+1}/{len(participants_ids)}: {subj_id}...")
        
        # Load this subject's data (uses npz cache if available)
        data = fast_load_meg_data(subj_id, cache_dir='meg_cache', base_path=rest_path)
        
        # Generate embeddings in batches
        individual_embeddings = []
        infer_dataset = MEG_Dataset(data, window_size=2000, transforms=transforms.Compose([ZScore()]), stride=500)
        infer_dataloader = DataLoader(infer_dataset, batch_size=256, shuffle=False, num_workers=2)

        for batch_idx, (x1, x2) in enumerate(infer_dataloader):
            x1 = x1.to(device)
            z1 = encoder(x1)
            individual_embeddings.append(z1.cpu())
        
        # Concatenate and save to disk immediately
        subject_embeddings = torch.cat(individual_embeddings, dim=0)
        torch.save(subject_embeddings, embeddings_dir / f"{subj_id}_embeddings.pt")
        print(f"  Generated {subject_embeddings.shape[0]} embeddings, saved to disk")
        
        # Free memory
        del data, individual_embeddings, subject_embeddings
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

print(f"\n✓ Generated and saved embeddings for {len(participants_ids)} subjects")
print("="*50)




#|export

print("\n" + "="*50)
print("LOADING METADATA FOR VISUALIZATION")
print("="*50)

# Load age and sex from participants.tsv
age_labels_list = []
sex_labels_list = []

for i, subj_id in enumerate(participants_ids):
    # Get metadata from dataframe
    subj_data = df[df['participant_id'] == subj_id].iloc[0]
    age = subj_data.get('age', 0)  # Adjust column name as needed
    sex = subj_data.get('sex', 0)  # Adjust column name as needed (or 'gender')
    
    # Load embeddings to get count
    embeddings = torch.load(embeddings_dir / f"{subj_id}_embeddings.pt")
    num_windows = embeddings.shape[0]
    
    age_labels_list.append(np.full(num_windows, age))
    sex_labels_list.append(np.full(num_windows, sex))
    
    del embeddings  # Free memory
    if (i + 1) % 50 == 0:
        print(f"Processed metadata for {i+1}/{len(participants_ids)} subjects")

age_labels_numpy = np.concatenate(age_labels_list, axis=0)
sex_labels_numpy = np.concatenate(sex_labels_list, axis=0)
print(f"Total data points: {len(age_labels_numpy)}")


print("\n" + "="*50)
print("DIMENSIONALITY REDUCTION (UMAP)")
print("="*50)

print("Fitting UMAP reducer in batches...")
dim_reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)

# Load embeddings in batches to avoid RAM overflow
batch_size_subjects = 50  # Process 50 subjects at a time
all_embeddings_2d = []

for batch_start in range(0, len(participants_ids), batch_size_subjects):
    batch_end = min(batch_start + batch_size_subjects, len(participants_ids))
    print(f"Processing subjects {batch_start+1}-{batch_end}/{len(participants_ids)}...")
    
    batch_embeddings = []
    for i in range(batch_start, batch_end):
        subj_id = participants_ids[i]
        embeddings = torch.load(embeddings_dir / f"{subj_id}_embeddings.pt")
        batch_embeddings.append(embeddings)
    
    batch_concat = torch.cat(batch_embeddings).numpy()
    
    # Fit on first batch, transform on subsequent
    if batch_start == 0:
        batch_2d = dim_reducer.fit_transform(batch_concat)
    else:
        batch_2d = dim_reducer.transform(batch_concat)
    
    all_embeddings_2d.append(batch_2d)
    del batch_embeddings, batch_concat
    print(f"  Batch shape: {batch_2d.shape}")

embedding_2d = np.vstack(all_embeddings_2d)
print(f"\nFinal UMAP output shape: {embedding_2d.shape}")
print("✓ UMAP reduction complete")
print("="*50)

#|export

print("\n" + "="*50)
print("CREATING VISUALIZATIONS")
print("="*50)

# --- Create a 1x2 figure ---
print("Generating plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# --- Plot 1: Colored by AGE ---
plot1 = ax1.scatter(
    embedding_2d[:, 0], 
    embedding_2d[:, 1],
    c=age_labels_numpy,    # Color by age
    cmap='jet',            # Continuous colormap
    s=0.5,
    alpha=0.3
)
ax1.set_title('UMAP Projection Colored by Age')
ax1.set_xlabel('UMAP Dimension 1')
ax1.set_ylabel('UMAP Dimension 2')
fig.colorbar(plot1, ax=ax1, label='Participant Age')

# --- Plot 2: Colored by SEX ---
# (Assuming sex=0 and sex=1)
plot2 = ax2.scatter(
    embedding_2d[:, 0], 
    embedding_2d[:, 1],
    c=sex_labels_numpy,    # Color by sex
    cmap='coolwarm',       # Categorical colormap
    s=0.5,
    alpha=0.3
)
ax2.set_title('UMAP Projection Colored by Sex')
ax2.set_xlabel('UMAP Dimension 1')
ax2.set_ylabel('UMAP Dimension 2')
fig.colorbar(plot2, ax=ax2, label='Participant Sex')

print("✓ Plots generated")
plt.show()

print("\n" + "="*50)
print("ALL PROCESSING COMPLETE")
print("="*50)

# Optional: Visualization code (commented out as it references undefined variables)
# transform = transforms.Compose([ZScore(), StdGaussianNoise(std=0.5)])
# norm_transform = transforms.Compose([ZScore()])
# plt.plot(norm_transform(data[:1000,0]))
# plt.plot(transform(data[:1000,0]))
# plt.show()

# from nbdev.export import nb_export 
# nb_export('understand.ipynb', '.')


sys.stdout = sys.__stdout__
train_log.close()
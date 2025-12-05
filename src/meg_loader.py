import mne
import numpy as np
import torch
from pathlib import Path

from constants import REST_PATH


# Load real MEG data from CamCAN dataset
def load_meg_data(subject_id, base_path):
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
    meg_file = base_path / f"{subject_id}" / "meg" / f"{subject_id}_task-rest_meg.fif"

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


def fast_load_meg_data(subject_id, cache_dir="meg_cache", base_path=REST_PATH):
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
        data = np.load(cache_path)["data"]
        return torch.from_numpy(data)
    else:
        # Fall back to slow loading
        return load_meg_data(subject_id, base_path)

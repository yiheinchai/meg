import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataset import Dataset
from pathlib import Path
from constants import WINDOW_CACHE_PATH


class NoiseInjection:
    def __init__(self):
        pass

    def __call__(self, timeseries):
        if isinstance(timeseries, np.ndarray):
            timeseries = torch.from_numpy(timeseries)

        std = timeseries.std()

        noise = torch.randn_like(timeseries) * std
        return timeseries + noise


class StdGaussianNoise:
    def __init__(self, *, std):
        self.std = std

    def __call__(self, timeseries):
        if isinstance(timeseries, np.ndarray):
            timeseries = torch.from_numpy(timeseries)
        return timeseries + (torch.randn_like(timeseries) * self.std)


class ZScore:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, timeseries: torch.Tensor):
        mean = timeseries.mean(
            dim=-1, keepdim=True
        )  # (batch, 306, ..time..) avg across time
        std = timeseries.std(dim=-1, keepdim=True)
        return (timeseries - mean) / (std + self.eps)


# |export


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=306, out_channels=64, kernel_size=3, padding=1
        )  # batch, 64, window_size
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)  # batch, 64

        self.fc = nn.Linear(64, 128)  # batch, 128

    def forward(self, x: torch.Tensor):
        x = x.float()
        x = self.conv1(x)
        x = self.relu(x)
        x = self.avg_pool(x)

        x = x.squeeze(-1)

        x = self.fc(x)

        return x


# |export


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def __call__(self, z1: torch.Tensor, z2: torch.Tensor):
        batch_size = z1.shape[0]
        device = z1.device

        z = torch.cat([z1, z2])  # 64 x 128 -> 128 x 128
        z = nn.functional.normalize(z)
        sim_matrix = (z @ z.T) / self.temperature

        # masking
        sim_matrix = torch.masked_fill(
            sim_matrix, torch.eye(sim_matrix.shape[0], device=device).bool(), -torch.inf
        )

        y = torch.tensor(
            [batch_size + i for i in range(batch_size)]
            + [i for i in range(batch_size)],
            dtype=torch.long,
            device=device,
        )

        loss = nn.functional.cross_entropy(sim_matrix, y)

        return loss


# |export


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
        segment = self.data[:, start_index : start_index + self.window_size]
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

    def __init__(self, window_cache_dir=WINDOW_CACHE_PATH, transforms=None, max_files=None):
        super().__init__()
        if transforms is None:
            raise TypeError("Transforms must be filled")

        self.window_cache_dir = Path(window_cache_dir)
        self.transforms = transforms

        # Fast indexing of all .pt files from subject subfolders
        print(f"Indexing .pt files in {window_cache_dir}...")
        self.file_paths = list(self.window_cache_dir.glob("*/*.pt"))

        if max_files is not None:
            self.file_paths = self.file_paths[:max_files]

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
        # Extract subject ID from folder name: SubjectID/WindowIndex.pt
        path = self.file_paths[index]
        subject_id = path.parent.name  # Get parent folder name
        return subject_id

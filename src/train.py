import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import umap
from constants import REST_PATH, WINDOW_CACHE_PATH
import os


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


if __name__ == "__main__":
    from multiprocessing import freeze_support
    from constants import WINDOW_CACHE_PATH

    MAX_FILES = None
    RUN_NAME = "run_1"
    CHECKPOINT_PATH = Path(f"./checkpoint")
    CHECKPOINT_NAME = f"{RUN_NAME}_checkpoint.pth"
    BATCH_SIZE = 64

    freeze_support()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # df = pd.read_csv(REST_PATH / "participants.tsv", sep="\t")

    # train_log = open("train_log", "w+")
    # sys.stdout = train_log

    print("\n" + "=" * 50)
    print("PREPARING TRAINING")
    print("=" * 50)

    # Note: Data is already z-scored during window cache creation
    # Only add noise augmentation here
    aug_pipeline = transforms.Compose([StdGaussianNoise(std=0.1)])

    train_dataset = WindowMEG_Dataset(
        window_cache_dir=WINDOW_CACHE_PATH, transforms=aug_pipeline, max_files=MAX_FILES)    
    print(f"Total training windows: {len(train_dataset)}")

    print(f"Creating DataLoader with batch_size={BATCH_SIZE}, 8 workers...")
    # Increased num_workers for .pt files (they load much faster than .fif)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    print(f"Total batches per epoch: {len(train_loader)}")

    print("\nInitializing model...")
    encoder = Encoder()
    encoder = encoder.to(device)
    print(f"Encoder moved to {device}")

    criterion = NTXentLoss(temperature=0.5)
    optimiser = optim.Adam(encoder.parameters(), lr=1e-3)
    print("Loss function and optimizer initialized")
    print("=" * 50)

    # |export

    print("\n" + "=" * 50)
    print("STARTING TRAINING")
    print("=" * 50)

    encoder.train()

    num_epochs = 100
    print(f"Training for {num_epochs} epochs\n")

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

            print("loss: ", loss.item())

            # Track metrics for live plot
            losses.append(loss.item())
            batch_nums.append(epoch * len(train_loader) + batch_idx)

            # Log every 10 batches
            if batch_idx % 10 == 0:
                print(
                    f"  Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / len(train_loader)
        print(
            f"\n Epoch {epoch+1}/{num_epochs} Complete | Average Loss: {avg_loss:.4f}"
        )
        torch.save(
            {
                "model_state_dict": encoder.state_dict(),
            },
            CHECKPOINT_PATH / f"{RUN_NAME}_epoch_{epoch}_checkpoint.pth",
        )
        print(f" Permanent checkpoint saved at Epoch {epoch}")
        torch.save(torch.tensor(losses), CHECKPOINT_PATH / f"{RUN_NAME}_epoch_{epoch}_training_losses.pt")
        print(" Model saved")


    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)

    losses_tensor = torch.tensor(losses)
    # save losses to disk
    torch.save(losses_tensor, f"{RUN_NAME}_training_losses.pt")

    # |export

    print("\nSaving model checkpoint...")



    torch.save(
        {
            "model_state_dict": encoder.state_dict(),
            "optimizer_state_dict": optimiser.state_dict(),
        },
        CHECKPOINT_PATH / CHECKPOINT_NAME,
    )
    print(" Model saved")

    # |export

    print("\n" + "=" * 50)
    print("LOADING MODEL FOR INFERENCE")
    print("=" * 50)

    encoder = Encoder()
    encoder.load_state_dict(
        torch.load(CHECKPOINT_PATH / CHECKPOINT_NAME, map_location=device)["model_state_dict"]
    )
    encoder = encoder.to(device)
    print(" Model loaded successfully")

    # |export

    print("\n" + "=" * 50)
    print("GENERATING EMBEDDINGS")
    print("=" * 50)
    print("Note: Using subject-level loading for embeddings (not window cache)")
    print(
        "This allows us to generate embeddings for all windows of each subject efficiently\n"
    )

    encoder.eval()

    # sys.stdout = sys.__stdout__
    # train_log.close()

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

from model import (
    NoiseInjection,
    StdGaussianNoise,
    ZScore,
    Encoder,
    NTXentLoss,
    MEG_Dataset,
    WindowMEG_Dataset,
    SubjectPairDataset,
    UniqueSubjectBatchSampler,
    WindowExhaustiveBatchSampler,
    HDF5SubjectPairDataset,
)


if __name__ == "__main__":
    from multiprocessing import freeze_support
    from constants import WINDOW_CACHE_PATH, HDF5_CACHE_PATH

    MAX_FILES = None
    RUN_NAME = "run_1"
    CHECKPOINT_PATH = Path(f"./checkpoint")
    CHECKPOINT_NAME = f"{RUN_NAME}_checkpoint.pth"
    BATCH_SIZE = 3  # Max unique subjects per batch
    NUM_EPOCHS = 10
    HDF5_PATH = HDF5_CACHE_PATH / "meg_windows.hdf5"  # Path to HDF5 cache file

    freeze_support()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # df = pd.read_csv(REST_PATH / "participants.tsv", sep="\t")

    # train_log = open("train_log", "w+")
    # sys.stdout = train_log

    print("\n" + "=" * 50)
    print("PREPARING TRAINING (HDF5 - ULTRA FAST I/O)")
    print("=" * 50)

    # Note: Data is already z-scored during HDF5 cache creation
    # Optionally add noise augmentation (can also use None for no augmentation)
    aug_pipeline = transforms.Compose([StdGaussianNoise(std=0.1)])
    # Or use no augmentation: aug_pipeline = None

    # Use HDF5SubjectPairDataset for ultra-fast I/O (eliminates file open/close bottleneck)
    train_dataset = HDF5SubjectPairDataset(
        hdf5_path=HDF5_PATH, transforms=aug_pipeline, max_subjects=MAX_FILES
    )
    print(f"Total training subjects: {len(train_dataset)}")

    # Use WindowExhaustiveBatchSampler to cover all windows per epoch
    batch_sampler = WindowExhaustiveBatchSampler(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
    )

    print(
        f"Creating DataLoader with batch_size={BATCH_SIZE}, exhaustive window sampling..."
    )
    # Use batch_sampler instead of batch_size + shuffle
    # num_workers=0 for HDF5 (h5py has its own threading, multiple workers can cause issues)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=0,  # HDF5 works best with single process due to file handle management
        pin_memory=True,
    )
    print(f"Total batches per epoch: {len(train_loader)}")
    print("  Each epoch covers ALL windows from ALL subjects")
    print("  Each batch contains windows from UNIQUE subjects (max {BATCH_SIZE})")
    print("  z1 and z2 are DIFFERENT windows from the SAME subject")
    print("  I/O: HDF5 single-file (100x faster than 700k tiny files)")

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

    print(f"Training for {NUM_EPOCHS} epochs\n")

    losses = []
    batch_nums = []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
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
            f"\n Epoch {epoch+1}/{NUM_EPOCHS} Complete | Average Loss: {avg_loss:.4f}"
        )
        torch.save(
            {
                "model_state_dict": encoder.state_dict(),
            },
            CHECKPOINT_PATH / f"{RUN_NAME}_epoch_{epoch}_checkpoint.pth",
        )
        print(f" Permanent checkpoint saved at Epoch {epoch}")
        torch.save(
            torch.tensor(losses),
            CHECKPOINT_PATH / f"{RUN_NAME}_epoch_{epoch}_training_losses.pt",
        )
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
        torch.load(CHECKPOINT_PATH / CHECKPOINT_NAME, map_location=device)[
            "model_state_dict"
        ]
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

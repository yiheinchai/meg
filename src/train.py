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

from model import NoiseInjection, StdGaussianNoise, ZScore, Encoder, NTXentLoss, MEG_Dataset, WindowMEG_Dataset


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

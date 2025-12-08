import os
from constants import REST_PATH, WINDOW_CACHE_PATH, CHECKPOINT_PATH, CACHE_PATH
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from model import Encoder

CHECKPOINT_NAME = "run_1_epoch_9_checkpoint.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder()
encoder.load_state_dict(
    torch.load(CHECKPOINT_PATH / CHECKPOINT_NAME, map_location=device, weights_only=True)["model_state_dict"]
)
encoder = encoder.to(device)


def get_participant_embeddings(subject_id, window_indices):
    windows = []
    for idx in window_indices:
        try:
            window = torch.load(WINDOW_CACHE_PATH / subject_id / f"{idx}.pt", weights_only=False)
            windows.append(window)
        except FileNotFoundError:
            pass  # Skip missing windows
    if not windows:
        return torch.empty(0, 128)  # Return empty tensor if no windows
    windows = torch.stack(windows)  # Shape: (batch, 306, 2000)
    with torch.no_grad():
        embeddings = encoder(windows.to(device)).cpu()
    return embeddings  # Shape: (batch, 128)

if __name__ == "__main__":
    print("Loading participants data...")
    participants = pd.read_csv(os.path.join(REST_PATH, 'participants.tsv'), sep='\t')
    print(f"Found {len(participants)} participants")
    embeddings = []
    labels = []
    total_participants = len(participants)
    for idx, row in enumerate(participants.iterrows()):
        _, row = row  # iterrows returns (index, row)
        subject_id = row['participant_id']
        print(f"Processing participant {idx+1}/{total_participants}: {subject_id}")
        window_indices = np.linspace(100, 900, 50, dtype=int)
        embs = get_participant_embeddings(subject_id, window_indices)
        windows_collected = embs.shape[0]
        for emb in embs:
            embeddings.append(emb.numpy())
            labels.append(subject_id)
        print(f"  Collected {windows_collected} windows for {subject_id}")
    
    print(f"Total embeddings collected: {len(embeddings)}")
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    print("Applying UMAP...")
    # Apply UMAP
    reducer = umap.UMAP(random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    print("UMAP applied successfully")
    
    print("Creating attribute dictionary...")
    attr_dict = {row['participant_id']: {col: row[col] for col in ['p2_age', 'sex', 'handedness', 'arm', 'years_edu']} for _, row in participants.iterrows()}
    
    print("Saving embeddings to cache...")
    # Save embeddings to cache
    os.makedirs(CACHE_PATH / 'embeddings', exist_ok=True)
    torch.save({
        'embeddings': torch.tensor(embeddings),
        'labels': labels,
        'umap_2d': torch.tensor(embedding_2d),
        'attr_dict': attr_dict
    }, CACHE_PATH / 'embeddings' / 'embeddings_data.pt')
    print("Embeddings saved to embeddings/embeddings_data.pt")
    
    # Create attribute dict
    

    continuous_attrs = ['p2_age', 'handedness', 'years_edu']
    categorical_attrs = ['sex', 'arm']
    
    for attr in continuous_attrs + categorical_attrs:
        print(f"Generating plot for {attr}...")
        plt.figure(figsize=(10, 8))
        values = [attr_dict[label][attr] for label in labels]
        if attr in continuous_attrs:
            # Filter out NaN
            valid_mask = [not pd.isna(v) for v in values]
            valid_2d = embedding_2d[valid_mask]
            valid_values = [v for v, m in zip(values, valid_mask) if m]
            scatter = plt.scatter(valid_2d[:, 0], valid_2d[:, 1], c=valid_values, cmap='viridis', alpha=0.5)
            plt.colorbar(scatter, label=attr)
        else:
            unique_vals = list(set(v for v in values if not pd.isna(v)))
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_vals)))
            val_to_color = dict(zip(unique_vals, colors))
            for val in unique_vals:
                mask = [v == val and not pd.isna(v) for v in values]
                plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], color=val_to_color[val], label=str(val), alpha=0.5)
            plt.legend(title=attr)
        plt.title(f'UMAP of MEG Embeddings colored by {attr}')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.savefig(f'./figures/umap_{attr}.png')
        plt.close()
        print(f"Plot saved: figures/umap_{attr}.png")
    
    print("All plots generated. Script completed.")
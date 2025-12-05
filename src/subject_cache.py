from pathlib import Path

import numpy as np

from constants import REST_PATH
from meg_loader import load_meg_data
import os


def preprocess_and_cache_all_subjects(
    subject_ids, cache_dir="meg_cache", base_path=REST_PATH
):
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

    print("\n" + "=" * 50)
    print("PREPROCESSING AND CACHING MEG DATA")
    print("=" * 50)
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

    print("\n" + "=" * 50)
    print("CACHING COMPLETE")
    print("=" * 50)
    print("All future loads will be 10-100x faster!\n")


if __name__ == "__main__":
    from subject_cache import preprocess_and_cache_all_subjects
    from constants import MEG_CACHE_PATH, REST_PATH

    dirs = os.listdir(REST_PATH)
    subject_ids = [d for d in dirs if d.startswith("sub-")]
    print(subject_ids)

    preprocess_and_cache_all_subjects(
        subject_ids, cache_dir=MEG_CACHE_PATH, base_path=REST_PATH
    )

import pickle
import bz2
import os
import numpy as np

def save_stats(data, filename="landscape_stats.pbz2"):
    """Compress and save data to a bz2-compressed pickle file."""
    print(f"Compressing and saving to {filename}...")
    with bz2.BZ2File(filename, "w") as f:
        pickle.dump(data, f)
    print(f"Done. File size: {os.path.getsize(filename) / 1024**2:.2f} MB")

def load_stats(filename="landscape_stats.pbz2"):
    """Load data from a bz2-compressed pickle file."""
    print(f"Loading {filename}...")
    with bz2.BZ2File(filename, "rb") as f:
        return pickle.load(f)

def ensure_dir(path):
    """Ensure that a directory exists."""
    os.makedirs(path, exist_ok=True)

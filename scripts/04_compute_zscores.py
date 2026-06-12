"""
Step 4: Z-Score Computation
Compute pixel-level Z-scores for every pixel against a collection of temporally evolving developmental landscapes.

Usage:
python scripts/04_compute_zscores.py --pixel_input all_pixels.parquet --landscape_input landscapes_collapsed.pbz2
"""

import argparse
import pandas as pd
import numpy as np
import progressbar
from geo_trajectories.utils import load_stats

def main():
    parser = argparse.ArgumentParser(description="Compute pixel-level Z-scores across all treatment landscapes.")
    parser.add_argument("--pixel_input", type=str, default="all_pixels_consolidated.parquet", help="Input pixel data.")
    parser.add_argument("--landscape_input", type=str, default="landscapes_collapsed.pbz2", help="Input collapsed landscapes.")
    parser.add_argument("--output", type=str, default="all_pixels_with_zscores.parquet", help="Output parquet file.")
    args = parser.parse_args()

    print("Loading data...")
    df = pd.read_parquet(args.pixel_input)
    landscapes = load_stats(args.landscape_input)

    # We need to map each pixel to its corresponding bin in the landscape
    # The edges are the same for all landscapes in the collapsed set
    first_key = list(landscapes.keys())[0]
    d_edges, e_edges = landscapes[first_key]['edges']

    # Digitize pixel coordinates
    df['d_bin'] = np.digitize(df['D'], d_edges) - 1
    df['e_bin'] = np.digitize(df['E'], e_edges) - 1

    # Clip to valid range
    df['d_bin'] = df['d_bin'].clip(0, len(d_edges) - 2)
    df['e_bin'] = df['e_bin'].clip(0, len(e_edges) - 2)

    print("Computing Z-scores for each landscape...")
    for key, stats in progressbar.progressbar(landscapes.items()):
        treatment, day = key
        col_name = f"Z_score_{treatment}_{day}"
        
        # Extract mean and std for the specific bins
        means = stats['mean'][df['d_bin'], df['e_bin']]
        stds = stats['std'][df['d_bin'], df['e_bin']]
        
        # Calculate Z-score
        z = (df['F_bin'] - means) / stds
        df[col_name] = z

    # Remove temporary bin columns
    df = df.drop(columns=['d_bin', 'e_bin'])
    
    print(f"Saving to {args.output}...")
    df.to_parquet(args.output, index=False)
    print("Done.")

if __name__ == "__main__":
    main()

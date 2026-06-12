"""
Step 2: Frequency Landscape Generation
Generate frequency landscapes (2D histograms) for each condition and replicate.

Usage:
python scripts/02_generate_landscapes.py --input all_pixels.parquet --output_prefix landscapes
"""

import argparse
import pandas as pd
import numpy as np
import progressbar
from geo_trajectories.landscapes import compute_2d_stats, collapse_replicates
from geo_trajectories.utils import save_stats

def main():
    parser = argparse.ArgumentParser(description="Generate 2D frequency landscapes from pixel data.")
    parser.add_argument("--input", type=str, default="all_pixels_consolidated.parquet", help="Input pixel data.")
    parser.add_argument("--output_prefix", type=str, default="landscapes", help="Prefix for output stats files.")
    parser.add_argument("--bins", type=int, default=100, help="Number of bins for 2D histograms.")
    args = parser.parse_args()

    print("Loading data...")
    alldata = pd.read_parquet(args.input)

    d_range = [np.percentile(alldata.D.values, 1), np.percentile(alldata.D.values, 99)]
    e_range = [np.percentile(alldata.E.values, 1), np.percentile(alldata.E.values, 99)]

    results = {}
    print("Computing landscapes for all wells...")
    groups = alldata.groupby(['treatment', 'day', 'well'])
    for name, group in progressbar.progressbar(groups):
        results[name] = compute_2d_stats(group, d_range, e_range, bins=args.bins)

    save_stats(results, f"{args.output_prefix}_replicates.pbz2")

    print("Collapsing replicates...")
    collapsed = collapse_replicates(results)
    save_stats(collapsed, f"{args.output_prefix}_collapsed.pbz2")

if __name__ == "__main__":
    main()

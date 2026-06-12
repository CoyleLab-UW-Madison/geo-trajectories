"""
Step 3: Trajectory Analysis
Compute pairwise distances between landscapes and project them into a 2D/3D state-space using MDS.

Usage:
python scripts/03_analyze_trajectories.py --input landscapes_collapsed.pbz2 --method welch
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from geo_trajectories.utils import load_stats
from geo_trajectories.trajectories import compute_pairwise_distances, fit_mds, plot_waddington_trajectories

def main():
    parser = argparse.ArgumentParser(description="Analyze developmental trajectories using MDS.")
    parser.add_argument("--input", type=str, default="landscapes_collapsed.pbz2", help="Input collapsed landscapes.")
    parser.add_argument("--method", type=str, default="welch", help="Distance metric.")
    parser.add_argument("--output_csv", type=str, default="mds_trajectories.csv", help="Output MDS coordinates.")
    args = parser.parse_args()

    print("Loading landscapes...")
    landscapes = load_stats(args.input)

    print("Computing pairwise distances...")
    dm = compute_pairwise_distances(landscapes, list(landscapes.keys()), method=args.method)

    print("Fitting MDS...")
    mds_df = fit_mds(dm)
    mds_df.to_csv(args.output_csv)
    print(f"MDS coordinates saved to {args.output_csv}")

    print("Generating Waddington plot...")
    fig, ax = plot_waddington_trajectories(mds_df)
    plt.savefig("waddington_trajectories.png")
    print("Plot saved to waddington_trajectories.png")
    plt.show()

if __name__ == "__main__":
    main()

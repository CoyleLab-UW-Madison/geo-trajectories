"""
Step 5: Pseudotime Calculation
Calculate a pseudotime estimate for every pixels based on similarity to frequency landscapes from different timepoints.

Usage:
python scripts/05_compute_pseudotime.py --input all_pixels_with_zscores.parquet --output all_pixels_with_pseudotime.parquet
"""

import argparse
import pandas as pd
import progressbar
from geo_trajectories.pseudotime import calculate_pseudotime

def main():
    parser = argparse.ArgumentParser(description="Calculate pseudotime from pixel-level Z-scores.")
    parser.add_argument("--input", type=str, default="all_pixels_with_zscores.parquet", help="Input data with Z-scores.")
    parser.add_argument("--output", type=str, default="all_pixels_with_pseudotime.parquet", help="Output parquet file.")
    parser.add_argument("--tau", type=float, default=1.5, help="Developmental prior width.")
    args = parser.parse_args()

    print("Loading data...")
    df = pd.read_parquet(args.input)

    # We typically calculate pseudotime per treatment
    results = []
    print("Calculating pseudotime for each treatment...")
    for treatment, group in df.groupby('treatment'):
        print(f"Processing {treatment}...")
        # Filter Z-score columns relevant to this treatment
        z_cols = [c for c in group.columns if f'Z_score_{treatment}_' in c]
        
        # Sub-select only necessary columns for calculation to save memory
        calc_df = group[['day'] + z_cols].copy()
        
        # Calculate pseudotime
        # Rename columns to match expected format 'Z_score_day'
        calc_df.columns = ['day'] + [c.split('_')[-1] for c in z_cols]
        # Wait, calculate_pseudotime expects 'Z_score_X' where X is the day
        # Let's fix the column names in the function call
        
        group['pseudotime'] = calculate_pseudotime(group, tau=args.tau)
        results.append(group)

    print("Consolidating...")
    final_df = pd.concat(results, ignore_index=True)
    final_df.to_parquet(args.output, index=False)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()

"""
Step 1: Pixel Extraction
Process raw .nd2 or .tif files to extract pixel-level GEO features (D levels, E levels, associated metrics (F_bin, amplitude, etc.)).

Usage:
python scripts/01_process_pixels.py --base_dir ./data --output all_pixels.parquet
"""

import os
import argparse
import pandas as pd
from pathlib import Path
import progressbar
from geo_trajectories.core import process_nd2_file, consolidate_data

def main():
    parser = argparse.ArgumentParser(description="Process raw .nd2 files into pixel-level features.")
    parser.add_argument("--base_dir", type=str, default=".", help="Base directory containing day folders.")
    parser.add_argument("--output", type=str, default="all_pixels_consolidated.parquet", help="Output parquet file.")
    parser.add_argument("--c_val", type=int, default=35, help="FFT filtering threshold.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    all_data_frames = []

    day_folders = sorted([d for d in base_dir.iterdir() if d.is_dir() and "day" in d.name])

    for day_path in progressbar.progressbar(day_folders):
        condition_folders = [c for c in day_path.iterdir() if c.is_dir()]
        
        for cond_path in progressbar.progressbar(condition_folders):
            files = list(cond_path.glob("*.nd2"))
            
            for file_path in files:
                print(f"Processing: {file_path.name}")
                df = process_nd2_file(file_path, c_val=args.c_val)
                df['condition'] = cond_path.name
                df['day_label'] = day_path.name
                df['filename'] = file_path.name
                all_data_frames.append(df)

    if all_data_frames:
        print("Consolidating all data...")
        final_df = consolidate_data(all_data_frames)
        final_df.to_parquet(args.output, index=False)
        print(f"Successfully saved to {args.output}")
    else:
        print("No data found to consolidate.")

if __name__ == "__main__":
    main()

"""
Step 6: Visualization & Cell Stats
Project pseudotime back onto images and aggregate results for individual cell contours or masks.

Usage:
python scripts/06_visualize_results.py --input all_pixels_with_pseudotime.parquet --label_dir ./masks
"""

import argparse
import pandas as pd
import numpy as np
import tifffile
import os
import progressbar
from geo_trajectories.pseudotime import map_feature_to_canvas, flood_labels_with_stat

def main():
    parser = argparse.ArgumentParser(description="Visualize pseudotime and generate cell-level statistics.")
    parser.add_argument("--input", type=str, default="all_pixels_with_pseudotime.parquet", help="Input data with pseudotime.")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Output directory for images.")
    parser.add_argument("--label_dir", type=str, default=".", help="Directory containing mask/label images.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading data...")
    df = pd.read_parquet(args.input)

    all_stats = []

    # Process each image file
    unique_files = df['filename'].unique()
    print(f"Processing {len(unique_files)} images...")
    
    for filename in progressbar.progressbar(unique_files):
        img_df = df[df['filename'] == filename]
        
        # 1. Map pseudotime to pixel canvas
        pt_canvas = map_feature_to_canvas(img_df, 'pseudotime')
        
        # Save pseudotime image
        out_name = filename.replace('.nd2', '_pseudotime.tif')
        tifffile.imwrite(os.path.join(args.output_dir, out_name), pt_canvas)
        
        # 2. If label image exists, compute stats
        # This part depends on the availability of mask files
        # Assuming mask files are named similarly in a 'masks' folder
        mask_path = os.path.join(args.label_dir, filename.replace('.nd2', '_mask.tif'))
        if os.path.exists(mask_path):
            labels = tifffile.imread(mask_path)
            flooded, l_ids, means = flood_labels_with_stat(labels, pt_canvas)
            
            # Save flooded image
            tifffile.imwrite(os.path.join(args.output_dir, filename.replace('.nd2', '_flooded.tif')), flooded)
            
            # Collect stats
            for i, label_id in enumerate(l_ids):
                all_stats.append({
                    'filename': filename,
                    'treatment': img_df['treatment'].iloc[0],
                    'day': img_df['day'].iloc[0],
                    'well': img_df['well'].iloc[0],
                    'cell_id': label_id,
                    'mean_pseudotime': means[i]
                })

    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_df.to_csv(os.path.join(args.output_dir, "cell_pseudotime_stats.csv"), index=False)
        print("Stats saved to cell_pseudotime_stats.csv")

if __name__ == "__main__":
    main()

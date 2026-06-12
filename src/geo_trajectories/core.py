import os
import pandas as pd
import numpy as np
import torch
import tifffile
import cellstream
import gc
from pathlib import Path
import progressbar

def process_nd2_file(file_path, c_val=35, write_registered=False):
    """
    Process a single .nd2 file to extract pixel-level features.
    """
    well_label = file_path.stem 
    
    # Load and register image
    img = cellstream.image.load_image(str(file_path))
    img = cellstream.experimental.register_and_transform_image_timeseries(img)
    
    if write_registered:
        new_name = str(file_path).replace(".nd2", "_registered.tif")
        tifffile.imwrite(new_name, img.numpy(), imagej=True, metadata={'axes': 'TCYX'})
    
    # Generate FFT features
    results = cellstream.fft.generate_fft_features(img, device='cuda', maxbin=20, batch_size=250)
    
    # FFT Filtering Logic: 
    ## select within the relevant FFT bin range (2-15 for our images)
    ## filter for high-quality pixels (statistically significant signals in both the E and D channel)

    compz = torch.prod(results['z_score'], dim=1, keepdims=True)
    clamps = torch.where(compz[2:15] > c_val, 1, 0)
    testpix = torch.where(clamps.sum(axis=0) == 1, 1, 0)
    
    fft_peak_bin = torch.argmax(clamps, dim=0)
    
    # Feature extraction
    mean_D = img[:, 1, :, :].mean(dim=0, keepdim=True)
    mean_E = img[:, 0, :, :].mean(dim=0, keepdim=True)
    D_sd = img[:, 1, :, :].std(dim=0, keepdim=True)
    E_sd = img[:, 0, :, :].std(dim=0, keepdim=True)
    
    amp, _ = results['full_amplitude'][2:15].max(dim=0, keepdim=True)
    normamp, _ = results['normalized_amplitude'][2:15].max(dim=0, keepdim=True)
    zs, _ = results['z_score'][2:15].max(dim=0, keepdim=True)
    
    # Pixel Coordinates
    mask_indices = torch.where(testpix == 1)
    y_coords = mask_indices[1].cpu().numpy()
    x_coords = mask_indices[2].cpu().numpy()
    
    #oragnize data-frame for high quality pixels and their metrics
    data_img = {
        'y': y_coords,
        'x': x_coords,
        'D': mean_D[mask_indices].cpu().numpy(),
        'E': mean_E[mask_indices].cpu().numpy(),
        'D_sd': D_sd[mask_indices].cpu().numpy(),
        'E_sd': E_sd[mask_indices].cpu().numpy(),
        'F_bin': fft_peak_bin[mask_indices].cpu().numpy() + 2,
        'D_amp': amp[:, 1, :, :][mask_indices].cpu().numpy(),
        'E_amp': amp[:, 0, :, :][mask_indices].cpu().numpy(),
        'D_norm_amp': normamp[:, 1, :, :][mask_indices].cpu().numpy(),
        'E_norm_amp': normamp[:, 0, :, :][mask_indices].cpu().numpy(),
        'D_z': zs[:, 1, :, :][mask_indices].cpu().numpy(),
        'E_z': zs[:, 0, :, :][mask_indices].cpu().numpy(),
    }
    
    df = pd.DataFrame(data_img)
    
    # Cleanup
    del results, img
    gc.collect()
    torch.cuda.empty_cache()
    
    return df

def consolidate_data(all_data_frames):
    """
    Consolidate multiple dataframes and parse condition strings.
    """
    if not all_data_frames:
        return None
        
    final_df = pd.concat(all_data_frames, ignore_index=True)
    
    # Parse condition: assuming format treatment_day_X_well_Y
    split_data = final_df['condition'].str.split('_', expand=True)
    
    # Mapping might need adjustment based on actual naming conventions
    final_df['treatment'] = split_data[0]
    final_df['day'] = split_data[2].astype(float)
    final_df['well'] = split_data[4].astype(int)
    
    return final_df

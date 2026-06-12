import numpy as np
import pandas as pd
import torch
from skimage.util import map_array
from scipy.ndimage import labeled_comprehension

def calculate_pseudotime(df, tau=1.5):
    """
    Vectorized calculation of pseudotime based on Z-scores and optional developmental priors.
        We selected tau=1.5 for our stem-cell analysis as this seemed to be the natural timescale
        across which differences between portraits began to diverge significantly.
    """
    z_cols = [c for c in df.columns if 'Z_score_' in c]
    days = np.array([float(c.split('_')[-1]) for c in z_cols])
    actual_days = df['day'].values
    
    zs = df[z_cols].values.astype(float)
    likelihood = np.exp(-0.5 * np.square(zs))
    
    diffs = (days - actual_days[:, np.newaxis]) / tau
    prior = np.exp(-0.5 * np.square(diffs))
    
    weights = likelihood * prior
    sum_weights = np.nansum(weights, axis=1)
    weighted_days_sum = np.nansum(weights * days, axis=1)
    
    results = np.divide(weighted_days_sum, sum_weights, 
                        out=actual_days.copy(), 
                        where=sum_weights != 0)
    
    return results

def map_feature_to_canvas(df, feature, m=1608, n=1608):
    """
    Map a feature from a dataframe onto a 2D image canvas using pixel coordinates.
    """
    x = torch.from_numpy(df['x'].values).long()
    y = torch.from_numpy(df['y'].values).long()
    values = torch.from_numpy(df[feature].values).float()

    canvas = torch.zeros((m, n), dtype=torch.float32)
    canvas[y, x] = values
    return canvas.numpy()

def flood_labels_with_stat(label_image, pixel_image, stat_func=np.mean):
    """
    Aggregate pixel values within labeled regions to obtain whole-cell metric.
    """
    labels = np.unique(label_image)
    labels = labels[labels != 0]

    def masked_stat(vals):
        clean_vals = vals[vals > 0]
        return stat_func(clean_vals) if clean_vals.size > 0 else 0

    stats = labeled_comprehension(pixel_image, label_image, labels, masked_stat, float, 0)
    return map_array(label_image, labels, stats), labels, stats

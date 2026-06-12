import numpy as np
import pandas as pd
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from .landscapes import compute_t_and_p

def compute_pairwise_distances(df, keys, method='welch'):
    """
    Compute a distance matrix between conditions using a collection of different metrics.
    """
    n = len(keys)
    dist_matrix = pd.DataFrame(index=keys, columns=keys, dtype=float)
    
    for i, key1 in enumerate(keys):
        for j, key2 in enumerate(keys):
            if method == 'welch':
                T, p = compute_t_and_p(df, key1, key2)
                dist = np.nansum(np.abs(T))
            elif method == 'difference_normalized':
                num = np.abs(df[key1]['mean'] - df[key2]['mean'])
                num = np.where(np.isfinite(num), num, np.nan)
                Nbin = np.sum(~(np.isnan(num)))
                dist = np.nansum(num) / Nbin
            else:
                raise ValueError(f"Unknown method: {method}")
                
            dist_matrix.loc[key1, key2] = dist
            
    return dist_matrix

def fit_mds(dist_matrix, n_components=2):
    """
    Fit MDS to a distance matrix.
    """
    sorted_df = dist_matrix.sort_index(axis=0).sort_index(axis=1)
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(sorted_df.values)
    
    cols = [f'Dim{i+1}' for i in range(n_components)]
    mds_df = pd.DataFrame(coords, index=sorted_df.index, columns=cols)
    return mds_df

def plot_waddington_trajectories(mds_df, colors=None):
    """
    Plot 3D Waddington-like trajectories. 
    In these plots, two axes are MDS derived and the third is experimental time.
    """
    plot_df = mds_df.reset_index()
    # Assuming index was (Treatment, Time)
    plot_df.columns = ['Treatment', 'Time'] + list(mds_df.columns)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((2, 2, 1))
    
    if colors is None:
        colors = {'control': '#1f77b4', 'ecto': '#ff7f0e', 'endo': '#2ca02c', 'meso': '#d62728'}
    
    for treatment in plot_df['Treatment'].unique():
        subset = plot_df[plot_df['Treatment'] == treatment].sort_values('Time')
        
        ax.plot(subset['Dim1'], subset['Dim2'], subset['Time'], 
                label=treatment, color=colors.get(treatment, None), linewidth=3, alpha=0.8)
        
        ax.scatter(subset['Dim1'], subset['Dim2'], subset['Time'], 
                   color=colors.get(treatment, None), s=40, edgecolors='w')
    
    ax.set_xlabel('MDS Dim 1')
    ax.set_ylabel('MDS Dim 2')
    ax.set_zlabel('Time')
    ax.view_init(elev=25, azim=45)
    
    plt.legend(title="Treatment")
    return fig, ax

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from scipy import stats

def compute_2d_stats(sub_df, d_range, e_range, bins=100, min_count=10):
    """
    Computes mean, std, and count histograms for column F based on (E, D).
    """
    d_vals = sub_df['D']
    e_vals = sub_df['E']
    f_vals = sub_df['F_bin']
    
    stat_args = {
        'x': d_vals,
        'y': e_vals,
        'values': f_vals,
        'bins': bins,
        'range': [d_range, e_range]
    }
    
    count_stat, d_edges, e_edges, _ = binned_statistic_2d(**stat_args, statistic='count')
    mean_stat, _, _, _ = binned_statistic_2d(**stat_args, statistic='mean')
    std_stat, _, _, _ = binned_statistic_2d(**stat_args, statistic='std')
    
    # Mask bins with low counts
    mask = count_stat < min_count
    mean_stat[mask] = np.nan
    std_stat[mask] = np.nan
    count_stat[mask] = np.nan
    
    return {
        'mean': mean_stat,
        'std': std_stat,
        'count': count_stat,
        'edges': (d_edges, e_edges)
    }

def collapse_replicates(results_dict):
    """
    Pool replicates by calculating weighted means and pooled standard deviations.
    """
    collapsed = {}
    conditions = set((k[0], k[1]) for k in results_dict.keys())
    
    for treat, day in conditions:
        well_keys = [k for k in results_dict.keys() if k[0] == treat and k[1] == day]
        
        all_means = np.stack([results_dict[k]['mean'] for k in well_keys])
        all_counts = np.stack([results_dict[k]['count'] for k in well_keys])
        all_stds = np.stack([results_dict[k]['std'] for k in well_keys])
        
        total_counts = np.nansum(all_counts, axis=0)
        weighted_mean = np.nansum(all_means * all_counts, axis=0) / total_counts
        pooled_std = np.sqrt(np.nansum((all_stds**2) * all_counts, axis=0) / total_counts)
        
        collapsed[(treat, day)] = {
            'mean': weighted_mean,
            'std': pooled_std,
            'count': total_counts,
            'edges': results_dict[well_keys[0]]['edges']
        }
    return collapsed

def compute_t_and_p(df, key1, key2):
    """
    Compute Welch's T-statistic and p-values between two histograms.
    """
    num = df[key1]['mean'] - df[key2]['mean']
    s1 = df[key1]['std']**2
    s2 = df[key2]['std']**2
    n1 = df[key1]['count']
    n2 = df[key2]['count']

    denom = np.sqrt(s1/n1 + s2/n2)
    T = num / denom

    df_welch = (
        (s1/n1 + s2/n2)**2 /
        (
            (s1**2 / (n1**2 * (n1 - 1))) +
            (s2**2 / (n2**2 * (n2 - 1)))
        )
    )

    p = 2 * (1 - stats.t.cdf(np.abs(T), df_welch))
    return T, p

def plot_2d_landscape(stats_dict, stat_name='mean', cmap='turbo', title="2D Landscape", vmin=0, vmax=10):
    """
    Plot the requested statistic onto a 2D grid.
    """
    data = stats_dict[stat_name]
    d_edges, e_edges = stats_dict['edges']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    current_cmap = plt.get_cmap(cmap).copy()
    current_cmap.set_bad(color='lightgrey')
    
    mesh = ax.pcolormesh(d_edges, e_edges, data.T, cmap=current_cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(mesh, ax=ax, label=stat_name)
    ax.set_xlabel('D')
    ax.set_ylabel('E')
    ax.set_title(title)
    
    return fig, ax

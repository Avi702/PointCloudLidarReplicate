import numpy as np

import matplotlib.pyplot as plt

import le_tools
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.interpolate import griddata



hhdcs_dir = '/Users/avnee/PointCloudLidarReplicate/hhd/hhdc_casals/'
hhdcs_files = le_tools.get_files(hhdcs_dir, concat_dir=True)
# Select one random file
random_hhdc = '/Users/avnee/PointCloudLidarReplicate/hhd/hhdc_casals/363148.0_4304467.0_hhdc_casals.npz'

hhdc_hr = np.load(random_hhdc)['arr_0']
hhdc_lr = np.load(random_hhdc.replace('hhdc_casals', 'hhdc_1x1'))['arr_0']
hhdc_hsr = hhdc_hr
def apply_kernel(dtm, center, k_size):
    """
    Extract a subsection of the dtm centered at the given point 
    by the given kernel size
    """
    x_0 = center[0] - k_size // 2
    y_0 = center[1] - k_size // 2

    x_0 = 0 if x_0 < 0 else x_0
    y_0 = 0 if y_0 < 0 else y_0

    return dtm[x_0: center[0] + k_size // 2 + 1, y_0: center[1] + k_size // 2 + 1]

def adaptive_dtm_filter(dtm):
    """
    Find noisy pixels in the dtm based on abnormal height values
    and replace them with the mean of the pixels in a window 
    that are below the some percentile
    """
    k_size = 7
    dtm = dtm.copy()

    for i in range(0, dtm.shape[0]):
        for j in range(0, dtm.shape[1]): 
            subsection = apply_kernel(dtm, (i, j), k_size).flatten()

            if subsection.shape[0] > 0:
                le_per = np.percentile(subsection, [70])

                if dtm[i,j] > le_per[0]:
                    dtm[i,j] = subsection[np.where(subsection <= le_per[0])].mean()

    return dtm

def filter_percentiles(view, percentiles):
    """
    Filter the view based on the given percentiles
    """
    lims = np.percentile(view, percentiles)
    filtered_view = np.where(view < lims[0], lims[0], view)
    filtered_view = np.where(filtered_view > lims[1], lims[1], filtered_view)

    return filtered_view

def get_views(hhdc):
    hhdc_dem = le_tools.get_dem(hhdc)
    hhdc_dem = filter_percentiles(hhdc_dem, [2, 98])

    hhdc_dtm = le_tools.get_dtm(hhdc)
    hhdc_dtm_filter1 = filter_percentiles(hhdc_dtm, [5, 97])
    hhdc_dtm_filter2 = adaptive_dtm_filter(hhdc_dtm_filter1.copy())
    hhdc_dtm_filter3 = filter_percentiles(hhdc_dtm_filter2, [5, 99])

    hhdc_chm = hhdc_dem - hhdc_dtm_filter3
    hhdc_chm = filter_percentiles(hhdc_chm, [2, 98])

    hhdc_chm_unfiltered = hhdc_dem - hhdc_dtm_filter1
    hhdc_chm_unfiltered = filter_percentiles(hhdc_chm_unfiltered, [2, 98])

    return hhdc_dem, (hhdc_dtm_filter1, hhdc_dtm_filter2, hhdc_dtm_filter3), (hhdc_chm, hhdc_chm_unfiltered)
def create_2dplot_cbd(csv_path):
    print(f"Loading data from: {csv_path}")
    

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    df_clean = df.dropna(subset=['X', 'Y', 'CBD'])
    
    x_coords = tuple(df_clean['X'])
    y_coords = tuple(df_clean['Y'])
    raw_cbd = tuple(df_clean['CBD'])
    
    print(f"Processing {len(raw_cbd)} points...")
    
 
    # Convert to numpy array
    cbd_array = np.array(raw_cbd)
    
    # Define a continuous colormap from Green -> Yellow -> Red
    # Values < 0.01 will be Green
    # Values > 0.10 will be Red
    
    colors = ["green", "yellow", "red"]
    cmap = mcolors.LinearSegmentedColormap.from_list("fire_risk", colors)
    
    norm = plt.Normalize(vmin=0.01, vmax=0.10)
    
    print(f"CBD Range: {np.min(cbd_array):.4f} to {np.max(cbd_array):.4f}")
        
    # 4. Graph the results
    # Center the coordinates
    x_center = np.mean(x_coords)
    y_center = np.mean(y_coords)
    x_plot = np.array(x_coords) - x_center
    y_plot = np.array(y_coords) - y_center
    
    # Sort points so high values are drawn on top
    sort_indices = np.argsort(cbd_array)
    x_sorted = x_plot[sort_indices]
    y_sorted = y_plot[sort_indices]
    c_sorted = cbd_array[sort_indices]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(x_sorted, y_sorted, c=c_sorted, s=2, cmap=cmap, norm=norm, alpha=1.0)
    
    # Add a continuous colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Canopy Bulk Density (kg/m³)')
    
    # Add text annotations for the risk zones on the colorbar
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='< 0.01: Gap / Air (Fire cannot climb)'),
        Patch(facecolor='yellow', edgecolor='black', label='0.01 - 0.05: Sparse Canopy'),
        Patch(facecolor='orange', edgecolor='black', label='0.05 - 0.10: Moderate Canopy'),
        Patch(facecolor='red', edgecolor='black', label='> 0.10: Dense Canopy (High Risk)')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', title="Fire Risk Thresholds", fontsize=10)
    
    ax.set_title(f"2D Fuel Density Map (Continuous)\n(n={len(x_sorted):,})")
    ax.set_xlabel('X (meters from center)')
    ax.set_ylabel('Y (meters from center)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()


import os

def create_cropped_comparison(csv_path, tensor_chm, center_coordinates, crop_size):
    center_x, center_y = center_coordinates
    size_x, size_y = crop_size
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    min_x, max_x = center_x - size_x/2, center_x + size_x/2
    min_y, max_y = center_y - size_y/2, center_y + size_y/2
    
    mask = (df['X'] >= min_x) & (df['X'] <= max_x) & (df['Y'] >= min_y) & (df['Y'] <= max_y)
    df_crop = df[mask].copy()
    
    if df_crop.empty:
        print(f"No points found in bounds: X[{min_x}, {max_x}], Y[{min_y}, {max_y}]")
        return

    # Center relative to the crop center for plotting
    x_coords = df_crop['X'] - center_x
    y_coords = df_crop['Y'] - center_y
    values = df_crop['CBD'].values
    
    resolution = 1.0 

    grid_x, grid_y = np.mgrid[
        -size_x/2 : size_x/2 : complex(0, size_x/resolution),
        -size_y/2 : size_y/2 : complex(0, size_y/resolution)
    ]
    

    points = np.column_stack((x_coords, y_coords))
    
    grid_cbd = griddata(points, values, (grid_x, grid_y), method='linear')
    
    mask_nan = np.isnan(grid_cbd)
    if np.any(mask_nan):
        grid_cbd[mask_nan] = griddata(points, values, (grid_x[mask_nan], grid_y[mask_nan]), method='nearest')

    fig, axes = plt.subplots(1, 2, figsize=(12, 12))
    
    extent = [-size_x/2, size_x/2, -size_y/2, size_y/2]
    
    im0 = axes[0].imshow(tensor_chm.T, origin='lower', extent=extent, cmap='viridis')
    axes[0].set_title('Tensor HS-CHM')
    plt.colorbar(im0, ax=axes[0])
    
    colors = ["green", "yellow", "red"]
    cmap = mcolors.LinearSegmentedColormap.from_list("fire_risk", colors)
    norm = plt.Normalize(vmin=0.01, vmax=0.10)
    
    im1 = axes[1].imshow(grid_cbd.T, origin='lower', extent=extent, cmap=cmap, norm=norm, interpolation='bilinear')
    
    axes[1].set_xlim(-size_x/2, size_x/2)
    axes[1].set_ylim(-size_y/2, size_y/2)
    axes[1].set_title(f'CBD')
    axes[1].set_xlabel('X (meters from center)')
    axes[1].set_ylabel('Y (meters from center)')
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    
    plt.colorbar(im1, ax=axes[1], label='Canopy Bulk Density (kg/m³)')
    
    plt.tight_layout()
    plt.show()
hhdc_hr_dem, hhdc_hr_dtm, hhdc_hr_chm = get_views(hhdc_hr)

filename_base = '363148.0_4304467.0_hhdc_casals.npz'
# Extract coords from filename
parts = filename_base.split('_')
cx = float(parts[0])
cy = float(parts[1])


sx = hhdc_hr.shape[0]
sy = hhdc_hr.shape[1]

pointcloud_csv = "/Users/avnee/PointCloudLidarReplicate/results/NEON_4_profile.csv"

create_cropped_comparison(pointcloud_csv, hhdc_hr_chm[1], (cx, cy), (sx, sy))


"""
Usage: Python or python3 2dplot.py <file1.csv> [file2.csv ...]
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
import sys
from pathlib import Path
from scipy.interpolate import griddata

def filter_percentiles(view, percentiles):
    """
    Clamps values between the specified percentiles to remove extreme outliers.
    """
    # Calculate the percentile limits (ignoring NaNs if any exist)
    lims = np.nanpercentile(view, percentiles)
    
    # Clamp values
    filtered_view = np.where(view < lims[0], lims[0], view)
    filtered_view = np.where(filtered_view > lims[1], lims[1], filtered_view)
    return filtered_view

def create_2dplot_cbd(csv_path):
    print(f"Loading data from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    # 1. Identify Height Column
    height_col = 'H' 
    if height_col not in df.columns and 'Height' in df.columns:
        height_col = 'Height'
        
    required_cols = ['X', 'Y', 'CBD', height_col]
    
    # 2. Filter Clean Data
    df_clean = df.dropna(subset=required_cols)
    
    if df_clean.empty:
        print("  Error: No valid data points found.")
        return

    points = df_clean[['X', 'Y']].values
    values_cbd = df_clean['CBD'].values
    values_height = df_clean[height_col].values
    
    print(f"Processing {len(values_cbd)} points...")

    # 3. Define Grid (Resolution = 1.0m)
    resolution = 1.0
    x_min, x_max = np.min(points[:,0]), np.max(points[:,0])
    y_min, y_max = np.min(points[:,1]), np.max(points[:,1])
    
    grid_x, grid_y = np.mgrid[
        x_min:x_max:resolution, 
        y_min:y_max:resolution
    ]
    
    print("  Interpolating grids...")
    

    grid_cbd = griddata(points, values_cbd, (grid_x, grid_y), method='linear')
    grid_height = griddata(points, values_height, (grid_x, grid_y), method='linear')

    mask_nan_cbd = np.isnan(grid_cbd)
    if np.any(mask_nan_cbd):
        grid_cbd[mask_nan_cbd] = griddata(points, values_cbd, (grid_x[mask_nan_cbd], grid_y[mask_nan_cbd]), method='nearest')
        
    mask_nan_h = np.isnan(grid_height)
    if np.any(mask_nan_h):
        grid_height[mask_nan_h] = griddata(points, values_height, (grid_x[mask_nan_h], grid_y[mask_nan_h]), method='nearest')

    # 5. Apply Height Threshold
    height_threshold = 1.25 
    print(f"  Filtering low vegetation (Height <= {height_threshold}m)...")
    grid_cbd[grid_height <= height_threshold] = 0

    # 6. Apply Percentile Filter
    print("  Applying percentile filter [2, 98]...")
    grid_cbd = filter_percentiles(grid_cbd, [2, 98])


    grid_cbd_t = grid_cbd.T
    
    # Define Plot Extents
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    plot_extent = [x_min - x_center, x_max - x_center, y_min - y_center, y_max - y_center]
    
    # 7. Visualization
    colors = ["darkblue","darkgreen", "green", "yellow", "orange","red","darkred","brown"]
    cmap = mcolors.LinearSegmentedColormap.from_list("fire_risk", colors)

    

    norm = plt.Normalize(vmin=0.0, vmax=0.10)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    img = ax.imshow(grid_cbd_t, extent=plot_extent, origin='lower', cmap=cmap, norm=norm)
    
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label('Canopy Bulk Density (kg/m³)')
    

    legend_elements = [
        Patch(facecolor='darkblue', edgecolor='black', label='~0.00: No Canopy / Gap'),
        Patch(facecolor='green', edgecolor='black', label='< 0.04: Low Density'),
        Patch(facecolor='yellow', edgecolor='black', label='0.04 - 0.06: Moderate'),
        Patch(facecolor='red', edgecolor='black', label='> 0.08: High Density')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', title="Density Levels")
    
    ax.set_title(f"2D Fuel Density Map\n(Interpolation: Linear+Nearest | Filter: Percentile [2,98])")
    ax.set_xlabel('X (meters from center)')
    ax.set_ylabel('Y (meters from center)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    results_dir = Path('results')
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    stem = Path(csv_path).stem
    output_file = plots_dir / f"{stem}_2d_plot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_file}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        print("No files provided. Searching for *_profile.csv in results/...")
        results_dir = Path('results')
        if results_dir.exists():
            files = list(results_dir.glob('*_profile.csv'))
            if not files:
                print("No profile CSV files found in results/.")
        else:
            print("Results directory not found.")
            files = []

    for csv_file in files:
        create_2dplot_cbd(str(csv_file))
"""
Usage: python fill_gaps.py global_cbd_metrics.csv
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.interpolate import griddata
import sys
from pathlib import Path

def fill_and_plot(csv_path):
    print(f"Loading data from: {csv_path}")
    
    # 1. Load Data
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    # Check for Height (to apply the filter BEFORE filling)
    height_col = None
    if 'H' in df.columns: height_col = 'H'
    elif 'Height' in df.columns: height_col = 'Height'

    # 2. Snap Coordinates to 10m Grid
    df['X'] = np.round(df['X'] / 10.0) * 10.0
    df['Y'] = np.round(df['Y'] / 10.0) * 10.0
    
    # Handle Duplicates (Average them)
    df = df.groupby(['X', 'Y'], as_index=False).mean()

    # 3. Apply Height Filter (Low Veg = 0)
    # We do this BEFORE filling so we don't accidentally fill a gap with "tree" data
    # if it's supposed to be short vegetation.
    if height_col:
        print(f"  Filtering: Setting CBD to 0 where Height <= 1.25m")
        df.loc[df[height_col] <= 1.25, 'CBD'] = 0

    # --- 4. THE FILLING LOGIC ---
    print("  Interpolating missing pixels (Linear method)...")
    
    # Define the full grid we WANT to have
    grid_x = np.arange(df['X'].min(), df['X'].max() + 10, 10)
    grid_y = np.arange(df['Y'].min(), df['Y'].max() + 10, 10)
    grid_x_mesh, grid_y_mesh = np.meshgrid(grid_x, grid_y)
    
    # The points we HAVE
    points = df[['X', 'Y']].values
    values = df['CBD'].values
    
    # Interpolate:
    # 1. 'linear': Accurate, fills inside the convex hull (triangle between points).
    filled_cbd = griddata(points, values, (grid_x_mesh, grid_y_mesh), method='linear')
    
    # 2. 'nearest': Fills the outer edges where linear might fail (extrapolation).
    # We use this ONLY to patch NaNs that 'linear' missed.
    nearest_cbd = griddata(points, values, (grid_x_mesh, grid_y_mesh), method='nearest')
    
    # Combine: Use Linear where possible, fallback to Nearest for edges
    mask_nan = np.isnan(filled_cbd)
    filled_cbd[mask_nan] = nearest_cbd[mask_nan]
    
    # --- 5. Save the Corrected Data ---
    # We flatten the grid back into a list to save as CSV
    filled_df = pd.DataFrame({
        'X': grid_x_mesh.flatten(),
        'Y': grid_y_mesh.flatten(),
        'CBD': filled_cbd.flatten()
    })
    
    # Remove crazy outliers if interpolation went wild (unlikely but safe)
    filled_df['CBD'] = filled_df['CBD'].clip(lower=0.0)
    

    # --- 6. Visualization (Raster Plot) ---
    print("  Generating plot...")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Colors
    colors = ["darkblue","darkgreen", "green", "yellow", "orange","red","darkred","brown"]
    cmap = mcolors.LinearSegmentedColormap.from_list("fire_risk", colors)
    
    # Plot Extents
    extent = [
        grid_x.min() - 5, 
        grid_x.max() + 5, 
        grid_y.min() - 5, 
        grid_y.max() + 5
    ]
    
    img = ax.imshow(filled_cbd, extent=extent, origin='lower', 
                    cmap=cmap, vmin=0.0, vmax=0.10, interpolation='none')
    
    cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Canopy Bulk Density (kg/m³)')
    
    ax.set_title(f"Gap-Filled Fuel Map\n(Interpolated Missing Pixels)")
    ax.set_xlabel('UTM X')
    ax.set_ylabel('UTM Y')
    ax.set_aspect('equal')
    
    legend_elements = [
        Patch(facecolor='darkblue', edgecolor='black', label='~0.00: Gaps / Short Veg'),
        Patch(facecolor='green', edgecolor='black', label='< 0.04: Low Density'),
        Patch(facecolor='yellow', edgecolor='black', label='0.04 - 0.06: Moderate'),
        Patch(facecolor='red', edgecolor='black', label='> 0.08: High Density'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    output_plot = Path('results/plots') / f"{Path(csv_path).stem}_filled.png"
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"  Saved plot to: {output_plot}\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = list(Path('.').glob('global_cbd_metrics.csv'))
        
    for f in files:
        fill_and_plot(str(f))
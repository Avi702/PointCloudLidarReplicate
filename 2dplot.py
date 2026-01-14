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

def rasterize_points(x, y, values, resolution=1.0, method='linear'):
    """
    Rasterize point data into a continuous grid using interpolation.
    
    Args:
        x, y: Coordinates
        values: Values at coordinates
        resolution: Grid cell size
        method: Interpolation method ('linear', 'nearest', 'cubic')
    
    Returns:
        grid: Interpolated mesh
        extent: [xmin, xmax, ymin, ymax]
    """
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    # Define grid coordinates
    grid_x, grid_y = np.mgrid[
        x_min:x_max:resolution, 
        y_min:y_max:resolution
    ]
    
    # Interpolate unstructured data to grid
    grid = griddata(
        (x, y), 
        values, 
        (grid_x, grid_y), 
        method=method,
        fill_value=np.nan
    )
    
    # Transpose to match imshow expectation (rows=y, cols=x)
    # Origin will be handled by imshow='lower'
    return grid.T, [x_min, x_max, y_min, y_max]

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
    
    # Rasterize using interpolation for continuous heatmap
    # Using 'linear' ensures continuity between points
    grid_cbd, extent = rasterize_points(df_clean['X'], df_clean['Y'], cbd_array, resolution=1.0, method='linear')
    
    # Define a continuous colormap from Green -> Yellow -> Red
    # Values < 0.01 will be Green
    # Values > 0.10 will be Red
    
    # Create a custom continuous colormap
    colors = ["green", "yellow", "red"]
    # Define the positions for these colors (0.0 to 1.0 range)
    cmap = mcolors.LinearSegmentedColormap.from_list("fire_risk", colors)
    cmap.set_bad('white', 0) # Set NaN values to transparent/white
    
    norm = plt.Normalize(vmin=0.01, vmax=0.10)
    
    print(f"CBD Range: {np.nanmin(grid_cbd):.4f} to {np.nanmax(grid_cbd):.4f}")
        
    # 4. Graph the results
    # Use the raster extent for axes
    x_min, x_max, y_min, y_max = extent
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    # Center the extent for plotting
    plot_extent = [x_min - x_center, x_max - x_center, y_min - y_center, y_max - y_center]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use imshow for rasterized plotting with bilinear interpolation for smoothness
    img = ax.imshow(grid_cbd, extent=plot_extent, origin='lower', cmap=cmap, norm=norm, interpolation='bilinear')
    
    # Add a continuous colorbar
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label('Canopy Bulk Density (kg/m³)')
    
    # Add text annotations for the risk zones on the colorbar
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='< 0.01: Gap / Air (Fire cannot climb)'),
        Patch(facecolor='yellow', edgecolor='black', label='0.01 - 0.05: Sparse Canopy'),
        Patch(facecolor='orange', edgecolor='black', label='0.05 - 0.10: Moderate Canopy'),
        Patch(facecolor='red', edgecolor='black', label='> 0.10: Dense Canopy (High Risk)')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', title="Fire Risk Thresholds", fontsize=10)
    
    # Calculate non-nan pixels for count
    n_pixels = np.count_nonzero(~np.isnan(grid_cbd))
    ax.set_title(f"2D Fuel Density Map (Rasterized 1m grid)\n(covered area approx. {n_pixels} m²)")
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
    print(f"Saved plot to: {output_file}")


def create_2d_plot_cfl(csv_path):
    print(f"Loading data from: {csv_path}")
    

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    df_clean = df.dropna(subset=['X', 'Y', 'CFL'])
    
    x_coords = tuple(df_clean['X'])
    y_coords = tuple(df_clean['Y'])
    raw_cfl = tuple(df_clean['CFL'])
    
    print(f"Processing {len(raw_cfl)} points...")
    
 
    # Convert to numpy array
    cfl_array = np.array(raw_cfl)
    
    # Rasterize the data (resolution=1.0, method='linear')
    grid_cfl, extent = rasterize_points(df_clean['X'], df_clean['Y'], cfl_array, resolution=1.0, method='linear')
    
    # Define a continuous colormap from Blue -> Cyan -> Yellow -> Red
    colors = ["blue", "cyan", "yellow", "red"]
    cmap = mcolors.LinearSegmentedColormap.from_list("cfl_risk", colors)
    cmap.set_bad('white', 0)
    
    norm = plt.Normalize(vmin=np.nanmin(grid_cfl), vmax=np.nanmax(grid_cfl))
    
    print(f"CFL Range: {np.nanmin(grid_cfl):.4f} to {np.nanmax(grid_cfl):.4f}")
        
    # 4. Graph the results
    # Use the raster extent for axes
    x_min, x_max, y_min, y_max = extent
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    plot_extent = [x_min - x_center, x_max - x_center, y_min - y_center, y_max - y_center]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use imshow for rasterized plotting
    img = ax.imshow(grid_cfl, extent=plot_extent, origin='lower', cmap=cmap, norm=norm, interpolation='bilinear')
    
    # Add a continuous colorbar
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label('Canopy Fuel Load (kg/m²)')
    
    n_pixels = np.count_nonzero(~np.isnan(grid_cfl))
    ax.set_title(f"2D Canopy Fuel Load Map (Rasterized 1m grid)\n(covered area approx. {n_pixels} m²)")
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
    print(f"Saved plot to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        laz_files = sys.argv[1:]
    else:
        print("No files provided. Searching for *_profile.csv in results/...")
        results_dir = Path('results')
        if results_dir.exists():
            laz_files = list(results_dir.glob('*_profile.csv'))
            if not laz_files:
                print("No profile CSV files found in results/.")
        else:
            print("Results directory not found.")
            laz_files = []

    for csv_file in laz_files:
        create_2dplot_cbd(str(csv_file))
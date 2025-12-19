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
    
    # Create a custom continuous colormap
    colors = ["green", "yellow", "red"]
    # Define the positions for these colors (0.0 to 1.0 range)
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
    

    results_dir = Path('results')
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    stem = Path(csv_path).stem
    output_file = plots_dir / f"{stem}_2d_plot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_file}")
"""
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
    
    # Define a continuous colormap from Blue -> Cyan -> Yellow -> Red
    colors = ["blue", "cyan", "yellow", "red"]
    cmap = mcolors.LinearSegmentedColormap.from_list("cfl_risk", colors)
    
    norm = plt.Normalize(vmin=np.min(cfl_array), vmax=np.max(cfl_array))
    
    print(f"CFL Range: {np.min(cfl_array):.4f} to {np.max(cfl_array):.4f}")
        
    # 4. Graph the results
    # Center the coordinates
    x_center = np.mean(x_coords)
    y_center = np.mean(y_coords)
    x_plot = np.array(x_coords) - x_center
    y_plot = np.array(y_coords) - y_center
    
    # Sort points so high values are drawn on top
    sort_indices = np.argsort(cfl_array)
    x_sorted = x_plot[sort_indices]
    y_sorted = y_plot[sort_indices]
    c_sorted = cfl_array[sort_indices]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(x_sorted, y_sorted, c=c_sorted, s=2, cmap=cmap, norm=norm, alpha=1.0)
    
    # Add a continuous colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Canopy Fuel Load (kg/m²)')
    
    ax.set_title(f"2D Canopy Fuel Load Map (Continuous)\n(n={len(x_sorted):,})")
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

"""

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
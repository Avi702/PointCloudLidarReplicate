"""
Standalone plotting script for LiDAR fuel metrics
Reads CSV files produced by pipeline.py and creates visualizations
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats
import laspy

# Fix for OverflowError in Agg backend with large datasets
plt.rcParams['agg.path.chunksize'] = 10000

def plot_bd_profile(profile_csv, output_file, cbh=None, threshold_abs=0.02, threshold_rel_pct=10):
    """
    Plot Bulk Density (CBD) vertical profile with key thresholds
    
    Parameters:
    - profile_csv: Path to the profile CSV file
    - output_file: Path to save the plot
    - cbh: Canopy Base Height (will be read from metrics if not provided)
    - threshold_abs: Absolute CBD threshold (default 0.02 kg/m³)
    - threshold_rel_pct: Relative threshold as percentage of max CBD (default 10%)
    """
    print(f"Creating BD profile plot from {profile_csv}...")
    
    # Read the profile data
    profile_df = pd.read_csv(profile_csv)
    
    # FIX: If the CSV contains point-level data (many rows), aggregate it back to a profile
    # We want unique Height (H) vs CBD pairs, sorted by Height
    if 'H' in profile_df.columns and 'CBD' in profile_df.columns:
        # Drop duplicates to get one row per height bin
        profile_df = profile_df[['H', 'CBD']].drop_duplicates()
        # Sort by Height to ensure the line is drawn sequentially
        profile_df = profile_df.sort_values('H')
    
    # Calculate thresholds
    max_cbd = profile_df['CBD'].max()
    threshold_rel = (threshold_rel_pct / 100) * max_cbd
    
    # Try to get CBH from the metrics file if not provided
    if cbh is None:
        metrics_file = Path(profile_csv).parent / 'all_metrics.csv'
        if metrics_file.exists():
            metrics_df = pd.read_csv(metrics_file)
            filename = Path(profile_csv).stem.replace('_profile', '')
            file_metrics = metrics_df[metrics_df['filename'] == filename + '.laz']
            if not file_metrics.empty:
                cbh = file_metrics['CBH'].values[0]
    
    # Create a new figure
    plt.figure(figsize=(12, 8))
    
    # Plot CBD profile (main line)
    plt.plot(profile_df['CBD'], profile_df['H'], 'b-', linewidth=2.5, label='CBD Profile', zorder=5)
    
    # Add vertical dashed line at threshold (absolute)
    plt.axvline(x=threshold_abs, color='red', linestyle='--', linewidth=2, 
                label=f'Absolute Threshold ({threshold_abs} kg/m³)', zorder=3)
    
    # Add vertical dashed line at relative threshold (10% of max)
    plt.axvline(x=threshold_rel, color='orange', linestyle='--', linewidth=2, 
                label=f'Relative Threshold ({threshold_rel_pct}% of max = {threshold_rel:.3f} kg/m³)', zorder=3)
    
    # Add horizontal line at CBH if available (even if it's 0)
    if cbh is not None and cbh >= 0:
        plt.axhline(y=cbh, color='green', linestyle='-', linewidth=2, 
                    label=f'Canopy Base Height (CBH = {cbh:.2f} m)', zorder=4)
    
    # Formatting
    plt.xlabel('Canopy Bulk Density (kg/m³)', fontsize=13, fontweight='bold')
    plt.ylabel('Height Above Ground (m)', fontsize=13, fontweight='bold')
    plt.title('Bulk Density Vertical Profile with Key Thresholds', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Set axis limits with some padding
    plt.xlim(0, max(profile_df['CBD'].max() * 1.1, threshold_abs * 1.5))
    plt.ylim(0, profile_df['H'].max() * 1.05)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    print(f"  - Max CBD: {max_cbd:.4f} kg/m³")
    print(f"  - Absolute threshold: {threshold_abs} kg/m³")
    print(f"  - Relative threshold ({threshold_rel_pct}%): {threshold_rel:.4f} kg/m³")
    if cbh is not None:
        print(f"  - CBH: {cbh:.2f} m\n")
    else:
        print(f"  - CBH: Not available\n")
    plt.close()

    


def create_3d_scatter_plot(las):
    x, y, z = las.X, las.Y, las.Z

    if len(x) > 500000:
        print(f"Subsampling to 500,000 points for faster rendering...")
        indices = np.random.choice(len(x), 250000, replace=False)
        x, y, z = x[indices], y[indices], z[indices]

        # Create 3D scatter plot with height-based color coding
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(projection='3d')

        # Color by height (z values) using a colormap
        # Create scatter plot with color based on height
        scatter = ax.scatter(x, y, z, 
                            c=z,  # Color by actual z values
                            cmap='viridis',  # Color scheme: viridis, plasma, inferno, turbo, jet
                            s=1,  # Point size
                            alpha=0.6,  # Transparency
                            vmin=z.min(),  # Min value for color scale
                            vmax=z.max())  # Max value for color scale

        # Add a colorbar to show height scale
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.1)
        cbar.set_label('Height (m)', rotation=270, labelpad=20, fontsize=12)

        # Labels and title
        ax.set_xlabel('X Coordinate (m)', fontsize=11)
        ax.set_ylabel('Y Coordinate (m)', fontsize=11)
        ax.set_zlabel('Z Coordinate (Height, m)', fontsize=11)
        ax.set_title('3D Point Cloud', fontsize=14, fontweight='bold')

        print(f"Height range: {z.min():.2f} m to {z.max():.2f} m")
        print("Displaying plot...")
        plt.show()




if __name__ == '__main__':
    results_dir = Path('results')
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if we got a specific profile CSV as argument
    if len(sys.argv) > 1:
        profile_csv = sys.argv[1]
        stem = Path(profile_csv).stem
        
        # Create BD profile plot
        output_file = plots_dir / f"{stem}_bd_plot.png"
        plot_bd_profile(profile_csv, output_file)
        
    else:
        print("No specific file provided. Plotting all available data...\n")
        
        # Plot individual profiles
        for profile_csv in results_dir.glob('*_profile.csv'):
            stem = profile_csv.stem
            
            # BD profile
            output_file = plots_dir / f"{stem}_bd_plot.png"
            plot_bd_profile(profile_csv, output_file)
            
        metrics_csv = results_dir / 'all_metrics.csv'
    x = input("Give file path to .laz to 3d render:")
    try:
        las = laspy.read(x)
        create_3d_scatter_plot(las)
    except Exception as e:
        print(f"Error reading {x}. Please check the file path.")
        sys.exit(1)


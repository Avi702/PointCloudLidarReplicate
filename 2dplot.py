import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_2dplot_cbd(csv_path):
    print(f"Loading data from: {csv_path}")
    

    df = pd.read_csv(csv_path)
    
    df_clean = df.dropna(subset=['X', 'Y', 'CBD'])
    
    x_coords = tuple(df_clean['X'])
    y_coords = tuple(df_clean['Y'])
    raw_cbd = tuple(df_clean['CBD'])
    
    print(f"Processing {len(raw_cbd)} points...")
    
 
    # Convert to numpy array
    cbd_array = np.array(raw_cbd)
    
    # Multiply by 100 to handle small numbers
    cbd_scaled = cbd_array * 100.0
    
    # Min-Max Normalization (0 to 1)
    # Formula: (value - min) / (max - min)
    min_val = np.min(cbd_scaled)
    max_val = np.max(cbd_scaled)
    
    print(f"Scaled CBD Range: {min_val:.4f} to {max_val:.4f}")
    
    if max_val > min_val:
        cbd_normalized = (cbd_scaled - min_val) / (max_val - min_val)
    else:
        # If all values are the same (e.g., all 0), set to 0
        cbd_normalized = np.zeros_like(cbd_scaled)
        
    # 4. Graph the results
    # Center the coordinates
    x_center = np.mean(x_coords)
    y_center = np.mean(y_coords)
    x_plot = np.array(x_coords) - x_center
    y_plot = np.array(y_coords) - y_center
    
    # Sort points so high values (Red) are drawn on top of low values (Green)
    sort_indices = np.argsort(cbd_normalized)
    x_sorted = x_plot[sort_indices]
    y_sorted = y_plot[sort_indices]
    c_sorted = cbd_normalized[sort_indices]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(x_sorted, y_sorted, c=c_sorted, s=2, cmap='RdYlGn_r', alpha=1.0)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Normalized Canopy Bulk Density (0-1)')
    
    ax.set_title(f"2D Fuel Density Map\n(n={len(x_sorted):,})")
    ax.set_xlabel('X (meters from center)')
    ax.set_ylabel('Y (meters from center)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

input_csv = "/Users/avnee/LiDAR/LidarReplicate/results/NEON_4_combined.csv"
create_2dplot_cbd(input_csv)
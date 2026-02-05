import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import os
import glob
import re 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


HHDC_DIR = "/Users/avnee/PointCloudLidarReplicate/HHDC"
CSV_DIR = "/Users/avnee/PointCloudLidarReplicate/results" 
OUTPUT_DIR = "/Users/avnee/PointCloudLidarReplicate/cbd_arrays"
IMAGES_DIR = "/Users/avnee/PointCloudLidarReplicate/cbd_images" 

def load_all_csvs(csv_folder):
    """
    Loads all relevant CSV files into memory once.
    """
    csv_files = glob.glob(os.path.join(csv_folder, "*_profile.csv"))
    data_frames = {}
    
    print(f"Loading {len(csv_files)} CSV files into memory... (This may take a moment)")
    
    for f in csv_files:
        basename = os.path.basename(f)
        site_key = basename.split('_profile')[0] 
        
        try:
            print(f"  Reading {basename}...")
            # Optimization: only load needed columns
            df = pd.read_csv(f, usecols=['X', 'Y', 'CBD', 'H'])         
            df.columns = df.columns.str.strip()
            
            data_frames[site_key] = df
        except Exception as e:
            print(f"  Error loading {basename}: {e}")
            
    return data_frames

def generate_cbd_grid(df, target_shape, center_coordinates, crop_size):
    """
    Generates a filtered CBD grid from a loaded DataFrame.
    """
    center_x, center_y = center_coordinates
    size_x, size_y = crop_size
    ny, nx = target_shape

    # Crop 
    min_x, max_x = center_x - size_x/2, center_x + size_x/2
    min_y, max_y = center_y - size_y/2, center_y + size_y/2
    
    mask = (df['X'] >= min_x) & (df['X'] <= max_x) & (df['Y'] >= min_y) & (df['Y'] <= max_y)
    df_crop = df[mask].copy()
    
    if df_crop.empty:
        return np.zeros(target_shape)

    x_coords = df_crop['X'] - center_x
    y_coords = df_crop['Y'] - center_y
    values_cbd = df_crop['CBD'].values
    
    if 'H' in df_crop.columns:
        values_height = df_crop['H'].values
    else:
        values_height = np.zeros_like(values_cbd)

    points = np.column_stack((x_coords, y_coords))

    if points.shape[0] < 4:
        return np.zeros(target_shape)
    
    # Create Target Grid
    grid_x_1d = np.linspace(-size_x/2, size_x/2, nx)
    grid_y_1d = np.linspace(-size_y/2, size_y/2, ny)
    grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)

    # Interpolate
    grid_cbd = griddata(points, values_cbd, (grid_x, grid_y), method='linear', fill_value=0)
    grid_height = griddata(points, values_height, (grid_x, grid_y), method='linear', fill_value=0)

    # Fill NaNs
    mask_nan_cbd = np.isnan(grid_cbd)
    if np.any(mask_nan_cbd):
        try:
             grid_cbd[mask_nan_cbd] = griddata(points, values_cbd, (grid_x[mask_nan_cbd], grid_y[mask_nan_cbd]), method='nearest')
        except: pass
        
    mask_nan_h = np.isnan(grid_height)
    if np.any(mask_nan_h):
        try:
            grid_height[mask_nan_h] = griddata(points, values_height, (grid_x[mask_nan_h], grid_y[mask_nan_h]), method='nearest')
        except: pass

    grid_cbd = np.nan_to_num(grid_cbd, 0)
    grid_height = np.nan_to_num(grid_height, 0)

    # Height Filter
    height_threshold = 1
    grid_cbd[grid_height <= height_threshold] = 0

    # Percentile Filter
    if np.any(grid_cbd > 0):
        lims = np.percentile(grid_cbd[grid_cbd > 0], [2, 98])
        grid_cbd = np.clip(grid_cbd, lims[0], lims[1])
    grid_cbd = np.flipud(grid_cbd) 

    return grid_cbd

def save_cbd_visualization(grid_cbd, save_path):
    """
    Saves the CBD grid as a clean PNG image without whitespace, axes, or colorbars.
    """
    colors = ["darkblue","darkgreen", "green", "yellow", "orange","red","darkred","brown"]
    cmap = mcolors.LinearSegmentedColormap.from_list("fire_risk", colors)
    
    # plt.imsave writes the array directly to an image file.
    # origin='lower' ensures correct orientation.
    # vmin/vmax ensures consistent color scaling across all images.
    plt.imsave(save_path, grid_cbd, cmap=cmap, origin='lower', vmin=0.0, vmax=0.10)

def extract_metadata_from_filename(filename):
    """
    Parses filenames to extract coordinates and site info.
    """
    try:
        parts = filename.split('_')
        center_x = float(parts[0])
        center_y = float(parts[1])

        site_name = None
        for part in parts:
            if part.startswith("NEON"):
                site_name = part
                idx = parts.index(part)
                if idx + 1 < len(parts) and parts[idx+1].isdigit():
                     site_name = f"{part}_{parts[idx+1]}"
                break
        
        return center_x, center_y, site_name
        
    except Exception as e:
        return None, None, None

def find_matching_df(center_x, center_y, site_name_hint, loaded_dfs):
    if site_name_hint and site_name_hint in loaded_dfs:
        return loaded_dfs[site_name_hint]

    for df in loaded_dfs.values():
        x_min, x_max = df['X'].min(), df['X'].max()
        y_min, y_max = df['Y'].min(), df['Y'].max()
        
        if (x_min <= center_x <= x_max) and (y_min <= center_y <= y_max):
            return df
            
    return None

def process_all_files():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    loaded_dfs = load_all_csvs(CSV_DIR)
    if not loaded_dfs:
        print("No CSV files loaded! Check your CSV_DIR path.")
        return

    files = sorted([f for f in os.listdir(HHDC_DIR) if f.endswith('.npz')])
    print(f"Found {len(files)} NPZ files to process.")
    
    count = 0
    skipped = 0
    
    for f in files:
        full_path = os.path.join(HHDC_DIR, f)
        
        try:
            tensor_data = np.load(full_path)
            arr = tensor_data[tensor_data.files[0]] 
            target_shape = arr.shape[-2:] 
            nx = target_shape[1]
            ny = target_shape[0]
            crop_size = (float(nx), float(ny)) 
        except Exception as e:
            print(f"Skipping {f}, bad tensor load: {e}")
            skipped += 1
            continue
            
        center_x, center_y, site_hint = extract_metadata_from_filename(f)
        if center_x is None:
            skipped += 1
            continue
            
        df = find_matching_df(center_x, center_y, site_hint, loaded_dfs)
        
        if df is None:
            skipped += 1
            continue
            
        grid = generate_cbd_grid(df, target_shape, (center_x, center_y), crop_size)
        
        npy_name = os.path.join(OUTPUT_DIR, f.replace(".npz", ".npy"))
        np.save(npy_name, grid)

        png_name = os.path.join(IMAGES_DIR, f.replace(".npz", ".png"))
        save_cbd_visualization(grid, png_name)
        
        count += 1
        if count % 20 == 0:
            print(f"Processed {count} files...")

    print(f"Done. Processed: {count}, Skipped: {skipped}")

if __name__ == "__main__":
    process_all_files()
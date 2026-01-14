from pathlib import Path
import numpy as np
import pandas as pd
import sys
import CSF
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

if len(sys.argv) < 2:
    print("Usage: python tensoranalysis.py <directory_path>")
    sys.exit(1)

directory_path = Path(sys.argv[1])
bin_size = 0.5  
tensor_size = 20  
clump = 0.77  
FMA = 0.2  

cos_theta = 0.5 

def normalize_height_with_tin(file):
    """
    Normalize the voxel tensor using Triangular Irregular Network (TIN) interpolation.
    This creates a ground surface model from ground points and normalizes all heights relative to it.
    
    Uses Cloth Simulation Filter (CSF) to classify ground points.
    
    Returns:
        normalized_tensor: Tensor with heights normalized to TIN ground surface
    """
    data = np.load(file)
    tensor = data[data.files[0]] if len(data.files) > 0 else data['arr_0']
    
    # Step 1: Extract all points from tensor for CSF
    points = []
    
    for x in range(tensor_size):
        for y in range(tensor_size):
            column = tensor[x, y, :]
            indices = np.nonzero(column)[0]
            for z_idx in indices:
                # Use voxel center as point
                points.append([x * bin_size, y * bin_size, z_idx * bin_size])
    
    if not points:
        return np.zeros_like(tensor, dtype=np.float32)

    points = np.array(points)
    
    # Step 2: Run CSF to find ground points
    csf = CSF.CSF()
    csf.setPointCloud(points)
    csf.params.bSloopSmooth = True
    csf.params.cloth_resolution = 1.0
    csf.params.class_threshold = 0.5
    
    ground_indices = CSF.VecInt()
    non_ground_indices = CSF.VecInt()
    csf.do_filtering(ground_indices, non_ground_indices)
    
    ground_points = points[ground_indices]
    
    # If no ground points found (rare), fallback to lowest points
    if len(ground_points) < 3:
        ground_points = []
        for x in range(tensor_size):
            for y in range(tensor_size):
                column = tensor[x, y, :]
                indices = np.nonzero(column)[0]
                if len(indices) > 0:
                    ground_points.append([x * bin_size, y * bin_size, indices[0] * bin_size])
        ground_points = np.array(ground_points)

    # Step 3: Create TIN using Delaunay triangulation
    xy_coords = ground_points[:, :2]  # X, Y coordinates
    z_values = ground_points[:, 2]     # Z heights
    
    # Create interpolator using TIN
    try:
        tin_interpolator = LinearNDInterpolator(xy_coords, z_values, fill_value=np.mean(z_values))
        
        # Step 4: Create ground surface for entire grid
        x_grid, y_grid = np.meshgrid(range(tensor_size), range(tensor_size), indexing='ij')
        grid_points = np.column_stack([x_grid.ravel() * bin_size, y_grid.ravel() * bin_size])
        ground_surface_flat = tin_interpolator(grid_points)
        ground_surface = ground_surface_flat.reshape(tensor_size, tensor_size)
        
    except Exception as e:
        # Fallback if TIN fails (e.g., collinear points, not enough points)
        # Use mean ground height for the whole tile
        mean_ground = np.mean(z_values)
        ground_surface = np.full((tensor_size, tensor_size), mean_ground)
    
    # Step 5: Normalize tensor relative to TIN surface
    normalized_tensor = np.zeros_like(tensor, dtype=np.float32)
    
    for x in range(tensor_size):
        for y in range(tensor_size):
            column = tensor[x, y, :]
            ground_height = ground_surface[x, y]
            ground_index = int(ground_height / bin_size)
            
            # For each voxel, calculate its height above the TIN surface
            for z_index in range(len(column)):
                if column[z_index] > 0:
                    absolute_height = z_index * bin_size
                    normalized_height = absolute_height - ground_height
                    
                    if normalized_height >= 0:
                        # Place in normalized position
                        normalized_z_index = int(normalized_height / bin_size)
                        if 0 <= normalized_z_index < len(column):
                            normalized_tensor[x, y, normalized_z_index] += column[z_index]
    
    return normalized_tensor


def compute_NRD(normalized_tensor, bin_size):
    """
    Compute Normalized Return Density (NRD) and Gap Fraction (Gf) at all height levels.
    
    Args:
        normalized_tensor: Voxel tensor normalized to ground (z=0 at ground)
        bin_size: Height of each bin in meters
    Returns:
        HAG: List of heights above ground (meters)
        NRD: Normalized Return Density at each height
        Gf: Gap fraction (1 - NRD) at each height
    """
    x_size, y_size, z_size = normalized_tensor.shape
    
    # Find maximum height with data
    max_z_index = 0
    for x in range(x_size):
        for y in range(y_size):
            column = normalized_tensor[x, y, :]
            indices = np.nonzero(column)[0]
            if len(indices) > 0:
                max_z_index = max(max_z_index, indices[-1])
    
    HAG = []
    NRD = []
    Gf = []
    
    # Calculate metrics for each height bin from ground to max height
    for z_index in range(max_z_index + 1):
        height = z_index * bin_size
        bin_start_index = z_index
        bin_end_index = z_index + 1
        
        total_hits = 0
        bin_hits = 0
        
        for x in range(x_size):
            for y in range(y_size):
                # Total hits from ground to top of current bin
                total_hits += np.sum(normalized_tensor[x, y, :bin_end_index])
                # Hits within this specific bin
                bin_hits += np.sum(normalized_tensor[x, y, bin_start_index:bin_end_index])
        
        if total_hits > 0:
            nrd = bin_hits / total_hits
            gf = 1 - nrd
        else:
            nrd = 0
            gf = 1
        
        HAG.append(height)
        NRD.append(nrd)
        Gf.append(gf)
    
    return HAG, NRD, Gf

def compute_PAD(Gf, clump, bin_size, cos_theta):
    """
    Compute Plant Area Density from Gap Fraction.
    PAD = (-ln(Gf) * cos(theta)) / (0.5 * omega * d)
    where omega is the clumping factor and d is the bin depth.
    """
    PAD = []
    for gf in Gf:
        if gf <= 0 or gf >= 1:
            PAD.append(0)
        else:
            pad_value = (-np.log(gf) * cos_theta) / (0.5 * clump * bin_size)
            PAD.append(pad_value)
    return PAD

def compute_BD(PAD, FMA):
    """
    Compute Bulk Density from Plant Area Density.
    BD = PAD * FMA
    """
    BD = []
    for height in PAD:
        BD.append(height * FMA)
    return BD



results = []
for file_path in directory_path.iterdir():
    if file_path.suffix == '.npz':
        filename = file_path.stem
        parts = filename.split('_')
        x = float(parts[0])
        y = float(parts[1])
        
        # Normalize the tensor using TIN interpolation
        norm_tensor = normalize_height_with_tin(file_path)
        
        # Compute metrics at all height levels
        HAG, NRD, Gf = compute_NRD(norm_tensor, bin_size)
        PAD = compute_PAD(Gf, clump, bin_size, cos_theta)
        BD = compute_BD(PAD, FMA)
        
        # Create a row for each height bin
        for i in range(len(HAG)):
            results.append({
                'X': x,
                'Y': y,
                'HAG': HAG[i],
                'NRD': NRD[i],
                'PAD': PAD[i],
                'CBD': BD[i],
            })

df = pd.DataFrame(results)


results_dir = Path("/Users/avnee/LiDAR/PointCloudLidarReplicate/results_tensor")
results_dir.mkdir(exist_ok=True)

output_file = results_dir / f"{directory_path.name}_metrics.csv"
df.to_csv(output_file, index=False)
print(f"Metrics saved to {output_file}")
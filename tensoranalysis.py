from pathlib import Path
import numpy as np
import pandas as pd
import sys

if len(sys.argv) < 2:
    print("Usage: python tensoranalysis.py <directory_path>")
    sys.exit(1)

directory_path = Path(sys.argv[1])
bin_size = 0.5  
tensor_size = 20
clump = 0.77  
FMA = 0.2  
HAG_sample = 1.5  

def normalize_height_to_ground(file):
    """
    Normalize the voxel tensor so that the lowest point in each column represents ground (z=0).
    This shifts all voxels in each column down so the first hit becomes the ground reference.
    
    Returns:
        normalized_tensor: Tensor with heights normalized to ground
        ground_height: The actual height of the ground in the original coordinate system (in bin units)
    """
    data = np.load(file)
    tensor = data[data.files[0]] if len(data.files) > 0 else data['arr_0']
    normalized_tensor = np.zeros_like(tensor)
    
    # Track the minimum ground index across all columns
    min_ground_index = float('inf')
    
    for x in range(tensor_size):
        for y in range(tensor_size):
            column = tensor[x, y, :]
            indices = np.nonzero(column)[0]
            if len(indices) > 0:
                # First hit is assumed to be ground
                ground_index = indices[0]
                min_ground_index = min(min_ground_index, ground_index)
                
                # Extract data from ground upward
                tree_data = column[ground_index:]
                height = len(tree_data)
                normalized_tensor[x, y, 0:height] = tree_data
    
    avg_ground_height_bins = min_ground_index if min_ground_index != float('inf') else 0
    avg_ground_height_meters = avg_ground_height_bins * bin_size
    
    return normalized_tensor, avg_ground_height_meters

def compute_NRD(normalized_tensor, bin_size, hag_meters):
    """
    Compute Normalized Return Density (NRD) and Gap Fraction (Gf) at a specific height above ground.
    
    Args:
        normalized_tensor: Voxel tensor normalized to ground (z=0 at ground)
        bin_size: Height of each bin in meters
        hag_meters: Height above ground at which to compute NRD (in meters)
    
    Returns:
        NRD: Normalized Return Density at the specified height
        Gf: Gap fraction (1 - NRD)
    """
    total_hits = 0
    bin_hits = 0
    x_size, y_size, z_size = normalized_tensor.shape
    
    bin_start_index = int(hag_meters / bin_size)
    bin_end_index = int((hag_meters + bin_size) / bin_size)
    
    for x in range(x_size):
        for y in range(y_size):
            # Total hits from ground to top of the bin
            total_hits += np.sum(normalized_tensor[x, y, :bin_end_index])
            # Hits within the specific bin
            bin_hits += np.sum(normalized_tensor[x, y, bin_start_index:bin_end_index])
    
    if total_hits == 0:
        return 0, 1
    
    NRD = bin_hits / total_hits
    Gf = 1 - NRD
    return NRD, Gf

def compute_PAD(Gf, clump, FMA, bin_size):
    """
    Compute Plant Area Density from Gap Fraction.
    PAD = (-ln(Gf) * cos(theta)) / (0.5 * omega * d)
    where omega is the clumping factor and d is the bin depth.
    """
    cos_theta = 0.75  # Simplified scanning angle
    if Gf <= 0 or Gf >= 1:
        return 0
    PAD = (-np.log(Gf) * cos_theta) / (0.5 * clump * bin_size)
    return PAD

def compute_BD(PAD, FMA):
    """
    Compute Bulk Density from Plant Area Density.
    BD = PAD * FMA
    """
    BD = PAD * FMA
    return BD

def get_max_height(normalized_tensor, bin_size):
    """
    Get the maximum vegetation height from the normalized tensor.
    
    Args:
        normalized_tensor: Voxel tensor normalized to ground
        bin_size: Height of each bin in meters
    
    Returns:
        max_height: Maximum height in meters
    """
    max_z_index = 0
    x_size, y_size, z_size = normalized_tensor.shape
    
    for x in range(x_size):
        for y in range(y_size):
            column = normalized_tensor[x, y, :]
            indices = np.nonzero(column)[0]
            if len(indices) > 0:
                max_z_index = max(max_z_index, indices[-1])
    
    return (max_z_index + 1) * bin_size  # Convert to meters

results = []
for file_path in directory_path.iterdir():
    if file_path.suffix == '.npz':
        filename = file_path.stem
        parts = filename.split('_')
        x = float(parts[0])
        y = float(parts[1])
        
        # Normalize the tensor to ground level
        norm_tensor, ground_height = normalize_height_to_ground(file_path)
        
        # Get maximum vegetation height
        max_height = get_max_height(norm_tensor, bin_size)
        
        # Compute metrics at the specified HAG sampling height
        NRD, Gf = compute_NRD(norm_tensor, bin_size, HAG_sample)
        PAD = compute_PAD(Gf, clump, FMA, bin_size)
        BD = compute_BD(PAD, FMA)
        
        results.append({
            'X': x,
            'Y': y,
            'Ground_Height': ground_height,  
            'Max_Height': max_height,  
            'Sampling_HAG': HAG_sample,  
            'NRD': NRD,
            'PAD': PAD,
            'BD': BD
        })

df = pd.DataFrame(results)

# Create results_tensor directory if it doesn't exist
results_dir = Path("/Users/avnee/LiDAR/PointCloudLidarReplicate/results_tensor")
results_dir.mkdir(exist_ok=True)

output_file = results_dir / f"{directory_path.name}_metrics.csv"
df.to_csv(output_file, index=False)
print(f"Metrics saved to {output_file}")
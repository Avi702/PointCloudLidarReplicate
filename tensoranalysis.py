import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import le_tools 

# --- CONFIGURATION ---
hhdcs_dir = Path('/Users/avnee/PointCloudLidarReplicate/hhdc_sim-main/hhdcfiles/hhdc_casals')

# Constants
BIN_SIZE = 0.5    
CLUMP = 0.77  
FMA = 0.2         
COS_THETA = 0.85

# --- FILTERING & NORMALIZATION ---

def apply_kernel(dtm, center, k_size):
    x_0 = center[0] - k_size // 2
    y_0 = center[1] - k_size // 2
    x_0 = max(0, x_0)
    y_0 = max(0, y_0)
    return dtm[x_0: center[0] + k_size // 2 + 1, y_0: center[1] + k_size // 2 + 1]

def adaptive_dtm_filter(dtm):
    k_size = 7
    dtm_out = dtm.copy()
    for i in range(dtm.shape[0]):
        for j in range(dtm.shape[1]): 
            subsection = apply_kernel(dtm, (i, j), k_size).flatten()
            if subsection.size > 0:
                le_per = np.percentile(subsection, [70])
                if dtm[i,j] > le_per[0]:
                    dtm_out[i,j] = subsection[subsection <= le_per[0]].mean()
    return dtm_out

def get_normalized_tensor(hhdc):
    """Shifts the tensor so Z=0 is the ground."""
    raw_dtm = le_tools.get_dtm(hhdc)
    smooth_dtm = adaptive_dtm_filter(raw_dtm)
    
    nx, ny, nz = hhdc.shape
    norm_tensor = np.zeros_like(hhdc)
    
    for x in range(nx):
        for y in range(ny):
            ground_bin = int(smooth_dtm[x, y])
            if ground_bin < nz:
                col = hhdc[x, y, ground_bin:]
                norm_tensor[x, y, :len(col)] = col
    return norm_tensor

# --- CORE CALCULATION ---

def calculate_global_metrics(normalized_tensor):
    """
    Returns tuple: (Global_CBD, Max_Height)
    """
    total_counts = np.sum(normalized_tensor, axis=(0, 1))
    
    if np.sum(total_counts) == 0: 
        return 0.0, 0.0

    # Find top of canopy
    valid_bins = np.nonzero(total_counts)[0]
    if len(valid_bins) == 0: 
        return 0.0, 0.0
        
    max_z = valid_bins[-1]
    max_height = max_z * BIN_SIZE  # Convert bin index to meters

    cbd_values = []
    cdf = np.cumsum(total_counts)
    
    for z in range(max_z + 1):
        N_bin = total_counts[z]
        N_cumulative = cdf[z]
        
        NRD = (N_bin / N_cumulative) if N_cumulative > 0 else 0
        Gf = max(0.00001, min(0.99999, 1.0 - NRD))
        
        PAD = (-np.log(Gf) * COS_THETA) / (0.5 * CLUMP * BIN_SIZE)
        CBD = PAD * FMA
        cbd_values.append(CBD)
        
    # Return Mean CBD (excluding ground bin 0)
    if len(cbd_values) > 1:
        global_cbd = np.mean(cbd_values[1:])
    else:
        global_cbd = 0.0
        
    return global_cbd, max_height

# --- MAIN EXECUTION ---

if not hhdcs_dir.exists():
    print(f"ERROR: Directory not found: {hhdcs_dir}")
else:
    files = le_tools.get_files(str(hhdcs_dir), concat_dir=True)
    print(f"Found {len(files)} files. Processing...")

    results = []

    for f in tqdm(files):
        path = Path(f)
        if path.suffix != '.npz': continue
        
        try:
            # 1. Load Data
            data = np.load(path)
            hhdc = data['arr_0']
            
            # 2. Parse Coords
            parts = path.stem.split('_')
            try:
                c_x = float(parts[0])
                c_y = float(parts[1])
            except ValueError:
                c_x = float(parts[1])
                c_y = float(parts[2])

            # 3. Calculate
            norm_tensor = get_normalized_tensor(hhdc)
            
            # Now returns two values
            cbd, height = calculate_global_metrics(norm_tensor)
            
            results.append({
                'X': c_x, 
                'Y': c_y, 
                'CBD': cbd,
                'H': height  # Added Height column
            })
            
        except Exception as e:
            print(f"Skipping {path.name}: {e}")

    # --- SAVE ---
    if results:
        df = pd.DataFrame(results)
        df.to_csv("global_cbd_metrics.csv", index=False) 
        print(f"\nDone! Saved {len(df)} rows to 'global_cbd_metrics.csv'")
        print(df.head())
    else:
        print("\nNo results were generated.")
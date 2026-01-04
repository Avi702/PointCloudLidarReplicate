import pandas as pd
import numpy as np


r_df = pd.read_csv("/Users/avnee/LiDAR/PointCloudLidarReplicate/results/NEON_4_profile.csv")

# Extract the unique profile from the R results
# We group by H and take the first value (since they are identical for the same H)
r_profile = r_df.groupby('H')[['PAD', 'CBD', 'NRD', 'SD_PAD', 'Ni', 'N', 'CBD_rollM']].first().reset_index()
r_profile = r_profile.sort_values('H')

py_df = pd.read_csv("/Users/avnee/LiDAR/PointCloudLidarReplicate/results_tensor/hhdc_1x1_metrics.csv")

print("\nChecking coordinate consistency...")
# Coordinate consistency checks between R and Python outputs
coords = ['X', 'Y']
stats = ['min', 'max', 'mean']

print(f"{'Metric':<10} | {'R_Value':<15} | {'Py_Value':<15} | {'Diff':<15}")
print("-" * 60)

for coord in coords:
    for stat in stats:
        r_val = getattr(r_df[coord], stat)()
        py_val = getattr(py_df[coord], stat)()
        diff = r_val - py_val
        print(f"{coord}_{stat:<6} | {r_val:<15.4f} | {py_val:<15.4f} | {diff:<15.4f}")
print("="*60)

# Aggregate Python results to get a global average profile to compare with R
# We group by H and take the mean
py_profile = py_df.groupby('H')[['PAD', 'CBD', 'NRD', 'SD_PAD', 'Ni', 'N', 'CBD_rollM']].mean().reset_index()
py_profile = py_profile.sort_values('H')

# Merge the two profiles on Height
# R uses 'H', Python uses 'H' (renamed from HAG)
merged = pd.merge(r_profile, py_profile, on='H', suffixes=('_R', '_Py'))

print(f"Merged columns: {merged.columns.tolist()}")

# Calculate errors
# We'll calculate Mean Absolute Error (MAE) 

metrics = [
    ('NRD', 'NRD_R', 'NRD_Py'),
    ('PAD', 'PAD_R', 'PAD_Py'),
    ('CBD', 'CBD_R', 'CBD_Py'),
    ('SD_PAD', 'SD_PAD_R', 'SD_PAD_Py'),
    ('Ni', 'Ni_R', 'Ni_Py'),
    ('N', 'N_R', 'N_Py'),
    ('CBD_rollM', 'CBD_rollM_R', 'CBD_rollM_Py')
]

print("\n" + "="*80)
print(f"{'Metric':<10} | {'MAE':<10} | {'MAPE (%)':<10} | {'wMAPE (%)':<10} | {'R_Mean':<10} | {'Py_Mean':<10}")
print("-" * 80)

for name, col_r, col_py in metrics:
    # Filter out rows where both are effectively zero to avoid skewing MAPE with 0/0 or div by zero
    
    valid_data = merged[[col_r, col_py]].dropna()
    
    if len(valid_data) == 0:
        print(f"{name:<10} | No valid data for comparison")
        continue
        
    r_vals = valid_data[col_r]
    py_vals = valid_data[col_py]
    
    mae = np.mean(np.abs(r_vals - py_vals))
    

        
    print(f"{name:<10} | {mae:<10.4f} | {r_vals.mean():<10.4f} | {py_vals.mean():<10.4f}")

print("="*80)


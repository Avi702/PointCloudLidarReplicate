import numpy as np
import matplotlib.pyplot as plt
import le_tools
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

hhdcs_dir = '/Users/avnee/PointCloudLidarReplicate/hhd/hhdc_casals/'
hhdcs_files = le_tools.get_files(hhdcs_dir, concat_dir=True)

random_hhdc = np.random.choice(hhdcs_files, 1)[0]
print(f"Selected File: {random_hhdc}")

hhdc_hr = np.load(random_hhdc)['arr_0']

try:
    hhdc_lr = np.load(random_hhdc.replace('hhdc_casals', 'hhdc_1x1'))['arr_0']
except:
    pass


def apply_kernel(dtm, center, k_size):
    x_0 = center[0] - k_size // 2
    y_0 = center[1] - k_size // 2
    x_0 = 0 if x_0 < 0 else x_0
    y_0 = 0 if y_0 < 0 else y_0
    return dtm[x_0: center[0] + k_size // 2 + 1, y_0: center[1] + k_size // 2 + 1]

def adaptive_dtm_filter(dtm):
    k_size = 7
    dtm = dtm.copy()
    for i in range(0, dtm.shape[0]):
        for j in range(0, dtm.shape[1]): 
            subsection = apply_kernel(dtm, (i, j), k_size).flatten()
            if subsection.shape[0] > 0:
                le_per = np.percentile(subsection, [70])
                if dtm[i,j] > le_per[0]:
                    dtm[i,j] = subsection[np.where(subsection <= le_per[0])].mean()
    return dtm

def filter_percentiles(view, percentiles):
    lims = np.percentile(view, percentiles)
    filtered_view = np.where(view < lims[0], lims[0], view)
    filtered_view = np.where(filtered_view > lims[1], lims[1], filtered_view)
    return filtered_view

def get_views(hhdc):
    hhdc_dem = le_tools.get_dem(hhdc)
    hhdc_dem = filter_percentiles(hhdc_dem, [2, 98])

    hhdc_dtm = le_tools.get_dtm(hhdc)
    hhdc_dtm_filter1 = filter_percentiles(hhdc_dtm, [5, 97])
    hhdc_dtm_filter2 = adaptive_dtm_filter(hhdc_dtm_filter1.copy())
    hhdc_dtm_filter3 = filter_percentiles(hhdc_dtm_filter2, [5, 99])

    hhdc_chm = hhdc_dem - hhdc_dtm_filter3
    hhdc_chm = filter_percentiles(hhdc_chm, [2, 98])

    hhdc_chm_unfiltered = hhdc_dem - hhdc_dtm_filter1
    hhdc_chm_unfiltered = filter_percentiles(hhdc_chm_unfiltered, [2, 98])

    return hhdc_dem, (hhdc_dtm_filter1, hhdc_dtm_filter2, hhdc_dtm_filter3), (hhdc_chm, hhdc_chm_unfiltered)

def create_cropped_comparison(csv_path, tensor_chm, center_coordinates, crop_size):
    center_x, center_y = center_coordinates
    size_x, size_y = crop_size


    tensor_chm_flipped = np.flipud(tensor_chm)
    

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip() 
    

    height_col = 'Z'
    if height_col not in df.columns:
        print(f"Error: Could not find height column (Z, H, or Height) in CSV. Columns: {df.columns}")
        return

    min_x, max_x = center_x - size_x/2, center_x + size_x/2
    min_y, max_y = center_y - size_y/2, center_y + size_y/2
    
    mask = (df['X'] >= min_x) & (df['X'] <= max_x) & (df['Y'] >= min_y) & (df['Y'] <= max_y)
    df_crop = df[mask].copy()
    
    if df_crop.empty:
        print(f"No points found in bounds: X[{min_x:.1f}, {max_x:.1f}], Y[{min_y:.1f}, {max_y:.1f}]")
        return


    x_coords = df_crop['X'] - center_x
    y_coords = df_crop['Y'] - center_y
    values_cbd = df_crop['CBD'].values
    values_height = df_crop[height_col].values  
    
    ny, nx = tensor_chm_flipped.shape
    grid_x_1d = np.linspace(-size_x/2, size_x/2, nx)
    grid_y_1d = np.linspace(-size_y/2, size_y/2, ny)
    grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d) 
    points = np.column_stack((x_coords, y_coords))

    grid_cbd = griddata(points, values_cbd, (grid_x, grid_y), method='linear')
    

    grid_height = griddata(points, values_height, (grid_x, grid_y), method='linear')


    mask_nan_cbd = np.isnan(grid_cbd)
    if np.any(mask_nan_cbd):
        grid_cbd[mask_nan_cbd] = griddata(points, values_cbd, (grid_x[mask_nan_cbd], grid_y[mask_nan_cbd]), method='nearest')
        
    mask_nan_h = np.isnan(grid_height)
    if np.any(mask_nan_h):
        grid_height[mask_nan_h] = griddata(points, values_height, (grid_x[mask_nan_h], grid_y[mask_nan_h]), method='nearest')

 
    
    height_threshold = 1.25 # 10% of the CBH
    grid_cbd[grid_height <= height_threshold] = 0



    grid_cbd = filter_percentiles(grid_cbd, [2, 98])


    fig, axes = plt.subplots(1, 2, figsize=(12, 12))
    extent = [-size_x/2, size_x/2, -size_y/2, size_y/2]


    im0 = axes[0].imshow(tensor_chm_flipped, origin='lower', extent=extent, cmap='viridis')
    axes[0].set_title('Tensor CHM (Height)')
    plt.colorbar(im0, ax=axes[0], label='Height (m)')


    colors = ["darkblue","darkgreen", "green", "yellow", "orange","red","darkred","brown"]
    cmap = mcolors.LinearSegmentedColormap.from_list("fire_risk", colors)
    norm = plt.Normalize(vmin=0.0, vmax=0.10)
    
    im1 = axes[1].imshow(grid_cbd, origin='lower', extent=extent, cmap=cmap, norm=norm)
    axes[1].set_title('Point Cloud CBD')
    axes[1].set_xlabel('X (meters)')
    axes[1].set_ylabel('Y (meters)')
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    
    plt.colorbar(im1, ax=axes[1], label='Canopy Bulk Density (kg/m³)')
    
    plt.tight_layout()
    plt.show()


hhdc_hr_dem, hhdc_hr_dtm, hhdc_hr_chm = get_views(hhdc_hr)


filename_base = random_hhdc.split('/')[-1]
parts = filename_base.split('_')
center_x = float(parts[0]) 
center_y = float(parts[1]) 

sy_pixels, sx_pixels = hhdc_hr.shape[-2:]
resolution = 1.0 
size_x = sx_pixels * resolution
size_y = sy_pixels * resolution

print(f"Center Coords: {center_x}, {center_y}")
print(f"Dimensions: {size_x}m x {size_y}m")


pointcloud_csv = "/Users/avnee/PointCloudLidarReplicate/results/NEON_4_profile.csv"
create_cropped_comparison(pointcloud_csv, hhdc_hr_chm[1], (center_x, center_y), (size_x, size_y))
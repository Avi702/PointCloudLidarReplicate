import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import os
import random
import le_tools
from neuralnet import MyCNN

 
hhdcs_dir = '/Users/avnee/PointCloudLidarReplicate/HHDC'
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
def compare_random_sample(model_path, input_dir, target_dir):
    device = torch.device('cpu')
    model = MyCNN()
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"ERROR: Model file not found at {model_path}")
        return

    model.eval()

    input_files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]
    valid_pairs = []
    
    # We now strictly look for the corresponding .npy file since we want to plot from array, not image
    cbd_arrays_dir = '/Users/avnee/PointCloudLidarReplicate/cbd_arrays'

    for f_in in input_files:
        stem = os.path.splitext(f_in)[0]
        # We need the numpy target to exist
        f_target_npy = f"{stem}.npy"
        
        if os.path.exists(os.path.join(cbd_arrays_dir, f_target_npy)):
            valid_pairs.append(f_in)
            
    if not valid_pairs:
        print("No matches found (checked for corresponding .npy files in cbd_arrays).")
        return

    chosen_input_file = random.choice(valid_pairs)
    print(f"Comparing: {chosen_input_file}")

    full_input_path = os.path.join(input_dir, chosen_input_file)
    stem = os.path.splitext(chosen_input_file)[0]
    npy_target_path = os.path.join(cbd_arrays_dir, f"{stem}.npy")

    npz_data = np.load(full_input_path)
    if 'tensor' in npz_data.files:
        input_array = npz_data['tensor'].astype(np.float32)
    else:
        input_array = npz_data[npz_data.files[0]].astype(np.float32)
    
    input_array = input_array.transpose(2, 0, 1) 
    input_tensor = torch.from_numpy(input_array).unsqueeze(0)

    # Resize input to match model training dimensions (32, 64)
    target_size = (32, 64)
    input_tensor = torch.nn.functional.interpolate(input_tensor, size=target_size, mode='bilinear', align_corners=False)

    # Load and interpolate the target array
    raw_target = np.load(npy_target_path)
    t_tensor = torch.from_numpy(raw_target).float().unsqueeze(0).unsqueeze(0)
    # Use nearest neighbor interpolation for the ground truth grid to preserve values
    t_tensor = torch.nn.functional.interpolate(t_tensor, size=target_size, mode='nearest')
    target_data_for_plot = t_tensor.squeeze().numpy()

    with torch.no_grad():
        prediction_tensor = model(input_tensor)
    
    prediction_array = prediction_tensor.squeeze().numpy()

    # Save output as .npy
    #output_npy_filename = f"prediction_{chosen_input_file.replace('.npz', '')}.npy"
   #np.save(output_npy_filename, prediction_array)
    #print(f"Saved model output to {output_npy_filename}")

    # compute CHM for the specific sample
    try:
        if 'arr_0' in npz_data:
            hhdc_obj = npz_data['arr_0']
        elif 'tensor' in npz_data:
             hhdc_obj = npz_data['tensor']
        else:
             hhdc_obj = npz_data[npz_data.files[0]]
        _, _, (filtered_chm, _) = get_views(hhdc_obj)
        filtered_chm = np.flipud(filtered_chm)
    except Exception as e:
        print(f"Error calculating CHM: {e}")
        # fallback if calculation fails
        hhdc_hr_dem, hhdc_hr_dtm, hhdc_hr_chm = get_views(hhdc_hr)
        filtered_chm = hhdc_hr_chm[1]

    colors = ["darkblue","darkgreen", "green","lightgreen", "yellow", "orange", "red", "darkred", "brown"]
    custom_cmap = LinearSegmentedColormap.from_list("forest_fire", colors, N=256)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(filtered_chm, cmap='viridis')
    plt.title(f"Input: CHM", fontsize=14)
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(target_data_for_plot, cmap=custom_cmap, vmin=0.0, vmax=0.10)
    plt.title("Target (Ground Truth)", fontsize=14)
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(prediction_array, cmap=custom_cmap, vmin=0.0, vmax=0.10)
    plt.title("Model Prediction", fontsize=14)
    plt.colorbar()

    plt.suptitle(f"Verification: {chosen_input_file}", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    MODEL_PATH = "best_model.pth"
    INPUT_DIR = '/Users/avnee/PointCloudLidarReplicate/HHDC'
    TARGET_DIR = '/Users/avnee/PointCloudLidarReplicate/cbd_images'
    
    compare_random_sample(MODEL_PATH, INPUT_DIR, TARGET_DIR)
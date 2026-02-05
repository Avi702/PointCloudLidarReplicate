import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import os
import random
import le_tools
try:
    from neuralnet import MyCNN
except ImportError:
    class MyCNN(nn.Module):
        def __init__(self):
            super(MyCNN, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
            self.relu3 = nn.ReLU()
            self.conv4 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, padding=1)
            self.relu4 = nn.ReLU()
            self.conv5 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, padding=1)
            self.relu5 = nn.ReLU()
            self.conv6 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.conv3(x)
            x = self.relu3(x)
            x = self.conv4(x)
            x = self.relu4(x)
            x = self.conv5(x)
            x = self.relu5(x)
            x = self.conv6(x)
            return x
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
    
    for f_in in input_files:
        stem = os.path.splitext(f_in)[0]
        f_target = f"{stem}.png"
        
        if os.path.exists(os.path.join(target_dir, f_target)):
            valid_pairs.append((f_in, f_target))
            
    if not valid_pairs:
        print("No matches found using exact filename matching.")
        return

    chosen_input_file, chosen_target_file = random.choice(valid_pairs)
    print(f"Comparing: {chosen_input_file}  <--->  {chosen_target_file}")

    full_input_path = os.path.join(input_dir, chosen_input_file)
    full_target_path = os.path.join(target_dir, chosen_target_file)

    npz_data = np.load(full_input_path)
    if 'tensor' in npz_data.files:
        input_array = npz_data['tensor'].astype(np.float32)
    else:
        input_array = npz_data[npz_data.files[0]].astype(np.float32)
    
    input_array = input_array.transpose(2, 0, 1) 
    input_tensor = torch.from_numpy(input_array).unsqueeze(0)


    dest_height = input_array.shape[1]
    dest_width = input_array.shape[2]
    

    original_target_img = Image.open(full_target_path).convert("RGB")
    original_target_img = original_target_img.resize((dest_width, dest_height))
    original_target_array = np.array(original_target_img)

    with torch.no_grad():
        prediction_tensor = model(input_tensor)
    
    prediction_array = prediction_tensor.squeeze().numpy()

    # Save output as .npy
    #output_npy_filename = f"prediction_{chosen_input_file.replace('.npz', '')}.npy"
   #np.save(output_npy_filename, prediction_array)
    #print(f"Saved model output to {output_npy_filename}")
    hhdc_hr_dem, hhdc_hr_dtm, hhdc_hr_chm = get_views(hhdc_hr)
    colors = ["darkblue", "darkgreen", "green", "yellow", "orange", "red", "darkred", "brown"]
    custom_cmap = LinearSegmentedColormap.from_list("forest_fire", colors, N=256)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(hhdc_hr_chm[1], cmap='viridis')
    plt.title(f"Input: CHM", fontsize=14)
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(prediction_array, cmap=custom_cmap)
    plt.title("Model Prediction", fontsize=14)
    plt.colorbar()


    plt.subplot(1, 3, 3)
    plt.imshow(original_target_array)
    
    plt.suptitle(f"Verification: {chosen_input_file}", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    MODEL_PATH = "best_model.pth"
    INPUT_DIR = '/Users/avnee/PointCloudLidarReplicate/HHDC'
    TARGET_DIR = '/Users/avnee/PointCloudLidarReplicate/cbd_images'
    
    compare_random_sample(MODEL_PATH, INPUT_DIR, TARGET_DIR)
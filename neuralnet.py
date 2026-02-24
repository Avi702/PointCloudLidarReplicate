import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
from torch import from_numpy
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import le_tools
input_dir = '/Users/avnee/PointCloudLidarReplicate/HHDC'
target_dir = '/Users/avnee/PointCloudLidarReplicate/cbd_arrays'
image_dir = '/Users/avnee/PointCloudLidarReplicate/cbd_images'
hhdcs_dir = '/Users/avnee/PointCloudLidarReplicate/HHDC'
hhdcs_files = le_tools.get_files(hhdcs_dir, concat_dir=True)

random_hhdc = np.random.choice(hhdcs_files, 1)[0]
print(f"Selected File: {random_hhdc}")

hhdc_hr = np.load(random_hhdc)['arr_0']


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        
        # --- BLOCK 1: 128 -> 64 channels (Residual) ---
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # 1x1 Conv to match channels for the skip connection
        self.skip1 = nn.Conv2d(128, 64, kernel_size=1) 
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.01)
        
        # --- BLOCK 2: 64 -> 32 channels (Residual) ---
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        # 1x1 Conv to match channels for the skip connection
        self.skip2 = nn.Conv2d(64, 32, kernel_size=1)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.01)
        
        # --- BLOCK 3: 32 -> 16 channels (Residual) ---
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        # 1x1 Conv to match channels for the skip connection
        self.skip3 = nn.Conv2d(32, 16, kernel_size=1)
        self.lrelu3 = nn.LeakyReLU(negative_slope=0.01)
        
        # --- Standard blocks for small features ---
        # Block 4: 16 -> 4 channels
        self.conv4 = nn.Conv2d(16, 4, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(4)
        self.lrelu4 = nn.LeakyReLU(negative_slope=0.01)
        
        # Block 5: 4 -> 2 channels
        self.conv5 = nn.Conv2d(4, 2, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(2)
        self.lrelu5 = nn.LeakyReLU(negative_slope=0.01)
        
        # Block 6: Final 1-channel projection
        self.conv6 = nn.Conv2d(2, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Block 1 (Residual)
        identity = self.skip1(x)       # Project input to 64 channels
        out = self.conv1(x)
        out = self.bn1(out)
        out += identity                # Add Residual
        x = self.lrelu1(out)           # Activation after addition

        # Block 2 (Residual)
        identity = self.skip2(x)       # Project input to 32 channels
        out = self.conv2(x)
        out = self.bn2(out)
        out += identity                # Add Residual
        x = self.lrelu2(out)

        # Block 3 (Residual)
        identity = self.skip3(x)       # Project input to 16 channels
        out = self.conv3(x)
        out = self.bn3(out)
        out += identity                # Add Residual
        x = self.lrelu3(out)

        # Remaining standard blocks
        x = self.lrelu4(self.bn4(self.conv4(x)))
        x = self.lrelu5(self.bn5(self.conv5(x)))
        
        # Final convolution
        x = self.conv6(x)
        
        return x

class ForestDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.image_dir = image_dir
        self.input_files = []
        self.target_files = []
        self.image_files = []
        raw_inputs = sorted([f for f in os.listdir(input_dir) if f.endswith('.npz')])
        raw_targets = set(os.listdir(target_dir))
        raw_images = set(os.listdir(image_dir))

        print(f"Found {len(raw_inputs)} input files. Matching with targets...")
        
        for f_input in raw_inputs:
            file_stem = os.path.splitext(f_input)[0]
            expected_target = f"{file_stem}.npy"
            expected_image = f"{file_stem}.png"
            if expected_target in raw_targets and expected_image in raw_images:
                self.input_files.append(os.path.join(input_dir, f_input))
                self.target_files.append(os.path.join(target_dir, expected_target))
                self.image_files.append(os.path.join(image_dir, expected_image))
            else:
                print(f"Skipping {f_input}: Target {expected_target} not found.")

        print(f"Successfully matched {len(self.input_files)} pairs.")

    def __getitem__(self, idx):
        npz_data = np.load(self.input_files[idx])
        npz_target = np.load(self.target_files[idx])
        if 'tensor' in npz_data.files:
            input_array = npz_data['tensor'].astype(np.float32)
        else:
            input_array = npz_data[npz_data.files[0]].astype(np.float32)
        
        if input_array.shape[-1] == 128:
             input_array = input_array.transpose(2, 0, 1) 
        
        target_array = npz_target.astype(np.float32)
        
        input_tensor = from_numpy(input_array)
        target_tensor = from_numpy(target_array).unsqueeze(0) # Add channel dim: (1, H, W)

        
        target_size = (32, 64)
        
        input_tensor = torch.nn.functional.interpolate(input_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
        target_tensor = torch.nn.functional.interpolate(target_tensor.unsqueeze(0), size=target_size, mode='nearest').squeeze(0)

        return input_tensor, target_tensor, self.input_files[idx]

    def __len__(self):
        return len(self.input_files)
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

def create_dataloader(input_dir, target_dir, batch_size=32, shuffle=True):
    dataset = ForestDataset(input_dir, target_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def create_model():
    return MyCNN()

def create_criterion():
    return nn.MSELoss()

def create_optimizer(model, learning_rate=0.001):
    return torch.optim.Adam(model.parameters(), lr=learning_rate)

def training_loop(dataloader, model, criterion, optimizer, num_epochs=10):
    model.train() 
    
    best_loss = float('inf') 
    lr_reduced = False 
    track_loss = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            inputs, targets = batch[0], batch[1]  
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = (epoch_loss / len(dataloader))*100000000
        track_loss.append(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f} new min error")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        if avg_loss < 5 and not lr_reduced:
            print("Loss < 5. Reducing learning rate...")
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001 
            lr_reduced = True
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), track_loss, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss vs. Epochs')
    plt.legend()
    plt.grid()
    plt.savefig('loss_curve.png')
    print("Loss curve saved to 'loss_curve.png'")

def visualize_results(model, dataloader):
    print("Visualizing results...")
    model.eval() 
    
    # Get a batch including paths
    batch = next(iter(dataloader))
    inputs, targets, paths = batch[0], batch[1], batch[2]
    
    with torch.no_grad():
        predictions = model(inputs)

    image_sample = targets[0].squeeze().numpy()
    prediction_sample = predictions[0].squeeze().numpy()
    path = paths[0]
    print(f"Visualizing sample from: {path}")
    
    try:
        hhdc_data = np.load(path)
        if 'arr_0' in hhdc_data:
            hhdc_obj = hhdc_data['arr_0']
        elif 'tensor' in hhdc_data:
             hhdc_obj = hhdc_data['tensor']
        else:
             hhdc_obj = hhdc_data[hhdc_data.files[0]]
        _, _, (filtered_chm, _) = get_views(hhdc_obj)

        filtered_chm = np.flipud(filtered_chm)
        
        title_text = "Input: CHM (Matched)"
    except Exception as e:
        print(f"Could not calculate CHM for {path}: {e}")
        filtered_chm = np.zeros_like(image_sample)
        title_text = "Input: CHM (Error)"

    colors = ["darkblue", "darkgreen", "green", "yellow", "orange", "red", "darkred", "brown"]

    custom_cmap = LinearSegmentedColormap.from_list("forest_fire", colors, N=256)

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(filtered_chm, cmap='viridis') 
    plt.title(title_text)
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(image_sample, cmap=custom_cmap, vmin=0.0,vmax=0.1)
    plt.title("Target (Ground Truth CBD)")
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(prediction_sample, cmap=custom_cmap, vmin=0.0,vmax=0.1)
    plt.title("Model Prediction")
    plt.colorbar()
    
    plt.show()

if __name__ == "__main__":
    dataloader = create_dataloader(input_dir, target_dir, batch_size=64) 
    model = create_model()
    criterion = create_criterion()
    optimizer = create_optimizer(model)
    
    if os.path.exists('best_model.pth'):
        print("Using existing model to continue training...")
        model.load_state_dict(torch.load('best_model.pth'))

    print("Starting training...")
    try:
        training_loop(dataloader, model, criterion, optimizer, num_epochs=200)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Loading best model found so far...")

    if os.path.exists('best_model.pth'):
        print("\nLoading the best model...")
        model.load_state_dict(torch.load('best_model.pth'))
        visualize_results(model, dataloader)
    else:
        print("No best_model.pth found.")
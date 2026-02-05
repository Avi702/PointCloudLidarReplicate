import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
from torch import from_numpy
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

input_dir = '/Users/avnee/PointCloudLidarReplicate/HHDC'
target_dir = '/Users/avnee/PointCloudLidarReplicate/cbd_arrays'
image_dir = '/Users/avnee/PointCloudLidarReplicate/cbd_images'

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
        
        # Ensure channel first: (H, W, C) -> (C, H, W) OR (C, H, W) already?
        # Standardize check: if last dim is 128, it's (H, W, C).
        if input_array.shape[-1] == 128:
             input_array = input_array.transpose(2, 0, 1) 
        
        target_array = npz_target.astype(np.float32)
        
        input_tensor = from_numpy(input_array)
        target_tensor = from_numpy(target_array).unsqueeze(0) # Add channel dim: (1, H, W)

        
        target_size = (32, 64)
        
        input_tensor = torch.nn.functional.interpolate(input_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
        target_tensor = torch.nn.functional.interpolate(target_tensor.unsqueeze(0), size=target_size, mode='nearest').squeeze(0)

        return input_tensor, target_tensor

    def __len__(self):
        return len(self.input_files)
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
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = (epoch_loss / len(dataloader))*10000000
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
    
    inputs, targets = next(iter(dataloader))
    
    with torch.no_grad():
        predictions = model(inputs)
    
    input_sample = inputs[0].numpy()
    image_sample = targets[0].squeeze().numpy()
    prediction_sample = predictions[0].squeeze().numpy()
    colors = ["darkblue", "darkgreen", "green", "yellow", "orange", "red", "darkred", "brown"]

    custom_cmap = LinearSegmentedColormap.from_list("forest_fire", colors, N=256)

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(np.sum(input_sample, axis=0), cmap='viridis') 
    plt.title("Input Summed (Lidar Density)")
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(image_sample, cmap=custom_cmap)
    plt.title("Target (Ground Truth CBD)")
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(prediction_sample, cmap=custom_cmap)
    plt.title("Model Prediction")
    plt.colorbar()
    
    plt.show()

if __name__ == "__main__":
    dataloader = create_dataloader(input_dir, target_dir, batch_size=4) 
    model = create_model()
    criterion = create_criterion()
    optimizer = create_optimizer(model)
    
    if os.path.exists('best_model.pth'):
        print("Using existing model to continue training...")
        model.load_state_dict(torch.load('best_model.pth'))
    
    print("Starting training...")
    try:
        training_loop(dataloader, model, criterion, optimizer, num_epochs=1000)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Loading best model found so far...")

    if os.path.exists('best_model.pth'):
        print("\nLoading the best model...")
        model.load_state_dict(torch.load('best_model.pth'))
        visualize_results(model, dataloader)
    else:
        print("No best_model.pth found.")
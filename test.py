import numpy as np

# Load the file
# If it's a .npy file:
try:
    np.set_printoptions(threshold=np.inf)
    data = np.load('/Users/avnee/PointCloudLidarReplicate/cbd_arrays/361204.0_4301389.0_NEON_1_hhdc_casals.npy')
    print("Shape:", data.shape)
    print("Data content:\n", data)
except FileNotFoundError:
    print("File not found. Please check the path.")

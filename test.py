import numpy as np

# Load the file
# If it's a .npy file:
try:
    np.set_printoptions(threshold=np.inf)
    data = np.load('/Users/avnee/PointCloudLidarReplicate/cbd_arrays/359011.0_4304417.0_NEON_D02_SERC_DP1_L026-1_2022052911_unclassified_point_cloud_hhdc_casals.npy')
    print("Shape:", data.shape)
    print("Data content:\n", data)
except FileNotFoundError:
    print("File not found. Please check the path.")

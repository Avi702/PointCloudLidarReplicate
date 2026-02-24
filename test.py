import numpy as np

# Load the file
# If it's a .npy file:
try:
    np.set_printoptions(threshold=np.inf)
    data = np.load('/Users/avnee/PointCloudLidarReplicate/HHDC/NEON_D02_SERC_DP1_364000_4307000_classified_point_cloud_hhdc_casals.npz')
    array_data = data['arr_0']
    print("Shape:", array_data.shape)
    #print("Data content:\n", data)
except FileNotFoundError:
    print("File not found. Please check the path.")

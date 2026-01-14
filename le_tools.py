from os import listdir
from os.path import isfile, join

import numpy as np

def get_files(folder, concat_dir=False):
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]

    if concat_dir:
        onlyfiles = [join(folder, f) for f in onlyfiles]
    
    return onlyfiles

def get_square_centers(pos_matrix, square_size, nonoverlap_factor, mask_bin_factor=0.1, offset=(2,2,2,2)):
    center_selection_mask = np.random.rand(pos_matrix.shape[0], pos_matrix.shape[1]) < mask_bin_factor

    center_selection_mask[:(square_size//2+offset[0]), :] = 0
    center_selection_mask[-(square_size//2+offset[1]):, :] = 0
    center_selection_mask[:, :(square_size//2+offset[2])] = 0
    center_selection_mask[:, -(square_size//2+offset[3]):] = 0

    center_indices = np.where(center_selection_mask > 0)
    dense_mask = np.ones((pos_matrix.shape[0], pos_matrix.shape[1]))

    selected_centers_x = []
    selected_centers_y = []

    for i in range(0, center_indices[0].shape[0]):
        indx, indj = center_indices[0][i], center_indices[1][i]

        ssl_x, ssr_x = indx - square_size // 2, indx + square_size // 2
        ssl_y, ssr_y = indj - square_size // 2, indj + square_size // 2

        if dense_mask[ssl_x:ssr_x, ssl_y:ssr_y].sum() > (((square_size)**2)*nonoverlap_factor):
            dense_mask[ssl_x:ssr_x, ssl_y:ssr_y] = 0

            selected_centers_x.append(indx)
            selected_centers_y.append(indj)

    return selected_centers_x, selected_centers_y

def get_dem(hhdc): # The height dimension must be the last one
    dem = np.zeros((hhdc.shape[0], hhdc.shape[1]))

    for i in range(hhdc.shape[0]):
        for j in range(hhdc.shape[1]):
            c = hhdc[i,j,:].cumsum()

            if np.sum(c) > 0:
                c = c/np.max(c)

            fp = np.argmax(c>0.98)
            
            dem[i,j] = fp

    return dem

def get_dtm(hhdc): # The height dimension must be the last one
    dtm = np.zeros((hhdc.shape[0], hhdc.shape[1]))

    for i in range(hhdc.shape[0]):
        for j in range(hhdc.shape[1]):
            c = hhdc[i,j,:].cumsum()

            if np.sum(c) > 0:
                c = c/np.max(c)

            lp = np.argmax(c>0.001)
            
            dtm[i,j] = lp

    return dtm
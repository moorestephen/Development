import numpy as np
import h5py
from pygrappa import grappa
import os

path_path = "/home/stephen.moore/MDSC508/Data/PathologicalData/GBMData/"
path_in_data = [item for item in os.listdir(path_path) if '_target' not in item]
path_target_data = [item for item in os.listdir(path_path) if '_target' in item]

os.mkdir('/home/stephen.moore/MDSC508/Data/PathologicalData/InputData')
os.mkdir('/home/stephen.moore/MDSC508/Data/PathologicalData/TargetData')

for i in range(len(path_in_data)):

    file_path = os.path.join(path_path, path_in_data[i])
    path_target_path = os.path.join(path_path, path_target_data[i])

    with h5py.File(file_path, 'r') as file:
        data = file['kspace']
        np_data = np.array(data)
        for slice in range(np_data.shape[2]):
            sample_data = np_data[:, :, slice, :]
            ctr, pd = int(256 / 2), 12 # Central 24 are ACS
            calib = sample_data[:, ctr-pd:ctr+pd, :].copy()
            res = grappa(sample_data, calib)
            res_converted = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(res, axes = (0, 1)), axes = (0, 1)), axes = (0, 1))
            res_cropped = res_converted[19:237, 43:213, :]
            res = np.fft.fftshift(np.fft.fft2(res_cropped, axes = (0, 1)), axes = (0, 1))
            save_path = '/home/stephen.moore/MDSC508/Data/PathologicalData/InputData/' + path_in_data[i][:-5] + f'_s{slice + 1}'
            np.save(save_path, res)

    with h5py.File(path_target_path, 'r') as file:
        data = file['kspace']
        np_data = np.array(data)
        for slice in range(np_data.shape[2]):
            sample_data = np_data[19:237, 43:213, slice]
            target_save_path = '/home/stephen.moore/MDSC508/Data/PathologicalData/TargetData/' + path_target_data[i][:-12] + f'_s{slice + 1}'
            np.save(target_save_path, sample_data)
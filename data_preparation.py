import e2e_varnet
import numpy as np
import torch
from torch import nn
import os
from torch.utils.data import random_split, Dataset, DataLoader, ConcatDataset
import random
import torch.optim as optim
# import matplotlib.pyplot as plt
from natsort import natsorted

def get_target(input_file : str, target_data : str):
    '''
    Returns the target reconstruction filename for the input image

    Parameters
    ----------
    input_file : str
        Input file string name
    '''
    id = "_".join(input_file.split(".")[0].split("_"))
    for target_file in target_data:
        if id in target_file:
            return target_file
    raise ValueError("No match in target data")

def get_patient_ids(data : list):
    '''
    Returns the different patient ids for the dataset (to split according to dataset)

    Parameters
    ----------
    data : list
        List of patient ids (which are strings)
    '''
    to_return = []
    for file in data:
        if not file.split("_")[1] in to_return:
            to_return.append(file.split("_")[1])
    return(to_return)

def load_file_data(folder : str, file : str) -> np.ndarray:
    '''
    Returns the np.ndarray of the queried file

    Parameters
    ----------
    folder : str
        Directory name where the queried file is
    file : str
        Name of the file

    Returns
    -------
    out : np.ndarray:
        Numpy array of shape (218, 170, 12) or (218, 180, 12) containing complex raw k-space data
    
    '''
    try:
        out = np.load(os.path.join(folder, file))
    except:
        raise ValueError("Loading file failed")
    
    return out
    
def subset_data_by_patient_id(data : list) -> list:
    '''
    Returns a dictionary mapping each patient id to their slices.
    For use in train/validation/test splitting.

    Parameters
    ----------
    data : list
        List of slices in a folder to be subsetted

    Returns
    -------
    dic : dict
        Dictionary that maps patient ID to a list containing their slices

    '''
    patient_ids = get_patient_ids(data)
    dic = {}
    for id in patient_ids:
        slices = []
        for x in data:
            if id in x:
                slices.append(x)
        dic[id] = natsorted(slices)[55:201]
    return(dic)

def get_slices_from_ids(ids: list, id_slice_dic : dict) -> list:
    '''
    Returns a list holding all slices from the patient ids included in the passed list ids

    Parameters
    ----------
    ids : list
        Patient IDs being queried
    id_slice_dic : dict
        Dictionary mapping patient IDs to their slices. Output of the subset_data_by_patient_id function

    Returns
    -------
    slices : list
        List of slice file names from the patients included in ids
    '''

    slices = []
    for id in ids:
        slices.extend(id_slice_dic[id])
    return slices

    
class MDSC508_Dataset(Dataset):
    def __init__(self, slices, input_path, target_path, input_data, target_data):
        self.slices = slices
        self.input_path = input_path
        self.target_path = target_path
        self.input_data = input_data
        self.target_data = target_data

    def __len__(self):
        return len(self.slices)
    
    def process_input(self, slice_index : int) -> torch.Tensor:
        '''
        Helper function which processes a slice query index and returns the data

        Parameters
        ----------
        slice : index
            Index into self.slices to query particular slice
        
        Returns
        -------
        mag : torch.Tensor
            Undersampled magnitude image
        '''



        input_data = torch.tensor(load_file_data(self.input_path, self.slices[slice_index]))
        # print(f'Input data shape: {input_data.shape}')
        
        input_data = input_data.permute(2, 0, 1)
        # h, w, c = input_data.shape
        # input_data = input_data.reshape(c, h, w) 

        # print(f'Reshaped tensor shape: {input_data.shape}')

        # FOR DEBUGGING
        input_data = input_data[:, :, :170]

        input_max = torch.max(torch.abs(torch.view_as_real(input_data)))

        input_data = torch.div(input_data, input_max)

        # fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 16))
        # input_data_idx = 0
        # for i in range(4):
        #     for j in range(3):
        #         ax = axes[i, j]
        #         converted = np.fft.ifft2(np.fft.ifftshift(input_data[input_data_idx, :, :].cpu().numpy()))
        #         real_part = np.abs(converted) # magnitude
        #         # real_part = np.angle(converted) #phase
        #         ax.imshow(real_part, cmap='gray')
        #         ax.set_title(f'Coil {input_data_idx + 1}')
        #         input_data_idx += 1

        # plt.tight_layout()
        # plt.show()
        
        return input_data
    
    def process_target(self, slice_index : int) -> torch.Tensor:
        '''
        Helper function which processes the target for a slice query

        Parameters
        ----------
        slice_index : int
            Index into self.slices to query particular slice

        Returns
        -------
        mag : torch.Tensor
            Target magnitude image
        '''

        target_data = torch.tensor(load_file_data(self.target_path, get_target(self.slices[slice_index], self.target_data)))

        mag = torch.abs(target_data) # Get magnitude data for comparison
        c, h, w = mag.shape
        mag = mag.view(c, h, w) # Reshape mag to be compatible with model dimension order (oops)

        ind_l = (w - 170) // 2
        ind_r = ind_l + 170

        mag = mag[:, :, ind_l:ind_r] # FOR DEBUGGING

        # norm_mag = torch.nn.functional.normalize(mag, dim = (1, 2))   
        norm_scale = torch.max(torch.abs(mag))
        norm_mag = torch.div(mag, norm_scale)

        return norm_mag
    
    def __getitem__(self, index):
        to_return = {
            'input': self.process_input(index),
            'target' : self.process_target(index)
        }
        return to_return

def build_exp_ds(patho_ratio, train_ids, val_ids, test_ids, patient_id_groups, input_path, target_path):
    '''
    Function to build an experimental dataset depending on the ratio of acceleration factor = 4

    Parameters
    ----------
    patho_ratio : double
        Proportion of data designated to be pathological
    train_ids : list
        List of patient ids (str) designated for training
    val_ids : list
        List of patient ids (str) desginated for validation
    test_ids : list
        List of patient ids (str) designated for testing
    patient_id_groups : dict
        Dictionary mapping patient ids to their slices 

    Returns
    -------
    out : dict
        Dictionary containing maps to 'train', 'val', 'test_r4', and 'test_r8' datasets 
    '''

    out = {} # Iniitalize dicionary to eventually output

    # Get train, validation, and test designated slices 
    train_slices = get_slices_from_ids(train_ids, patient_id_groups)
    random.shuffle(train_slices)
    val_slices = get_slices_from_ids(val_ids, patient_id_groups)
    random.shuffle(val_slices)
    test_slices = get_slices_from_ids(test_ids, patient_id_groups)
    random.shuffle(test_slices)

    '''
    NEED TO REFINE - UNDER CONSTRUCTION

    # Build train dataset
    train_r4 = MDSC508_Dataset(train_slices[:int(r4_ratio * len(train_slices))],
                                        4, input_path, target_path)
    train_r8 = MDSC508_Dataset(train_slices[int(r4_ratio * len(train_slices)):],
                                        8, input_path, target_path)
    out['train'] = ConcatDataset([train_r4, train_r8])

    # Build val dataset
    val_r4 = MDSC508_Dataset(val_slices[:int(r4_ratio * len(val_slices))],
                                      8, input_path, target_path)
    val_r8 = MDSC508_Dataset(val_slices[int(r4_ratio * len(val_slices)):],
                                      8, input_path, target_path)
    out['val'] = ConcatDataset([val_r4, val_r8])

    # Build test_r4
    out['test_r4'] = MDSC508_Dataset(test_slices[:int(len(test_slices) / 2)],
                                              4, input_path, target_path)
    
    # Build test_r8
    out['test_r8'] = MDSC508_Dataset(test_slices[int(len(test_slices) / 2):],
                                              8, input_path, target_path)
    
    '''

    return out


# TODO: ADOPT FOR NORMAL/PATHO EXPERIMENTAL SETUP
def build_exp_dls(train_ds, val_ds, test_r4_ds, test_r8_ds, batch_size):
    '''
    Function to build the dataloaders

    Parameters
    ----------
    train_ds:
        Training dataset
    val_ds:
        Validation dataset
    test_r4_ds:
        Testing dataset (R = 4)
    test_r8_ds:
        Testing dataset (R = 8)
    batch_size:
        Batch size to use for dataloaders

    Returns
    -------
    out : dict
        Dicionary mapping 'train', 'val', test_r4', and test_r8' to their respective dataloaders
    '''

    out = {}
    out['train'] = DataLoader(train_ds, batch_size, shuffle = True)
    out['val'] = DataLoader(val_ds, batch_size, shuffle = True)
    out['test_r4'] = DataLoader(test_r4_ds, batch_size, shuffle = True)
    out['test_r8'] = DataLoader(test_r8_ds, batch_size, shuffle = True)

    return out
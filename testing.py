import torch
import random
from torchmetrics.image import StructuralSimilarityIndexMeasure
import argparse
from e2e_varnet import E2EVarNet
import torch.optim as optim
import sys
import data_preparation as dp
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {torch.cuda.get_device_name(device)}")

# model = E2EVarNet(device).type(torch.float32)

model = torch.load('Testing1.pth')

# checkpoint = torch.load('Testing1.pth')
# model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)

ssim = StructuralSimilarityIndexMeasure(data_range = (0, 1)).to(device)

input_path = "C:/Users/steph/OneDrive/Documents/University/Research/VIL/Thesis Project/NormalData/Train_H5/slices"
target_path = "C:/Users/steph/OneDrive/Documents/University/Research/VIL/Thesis Project/NormalData/Train_H5/target_slice_images"
input_data = os.listdir(input_path)
target_data = os.listdir(target_path)

# Num patients in train/val/test subsets - small for preliminary results
num_patients_train = 5
num_patients_val = 1
num_patients_test = 8

patient_id_groups = dp.subset_data_by_patient_id(input_data)
patient_ids = list(patient_id_groups.keys())

print(f"Total data available includes {len(input_data)} slices from {len(patient_ids)} patients")

random.shuffle(patient_ids) # Shuffle patient ids for random distribution

if (num_patients_train + num_patients_val + num_patients_test <= len(patient_ids)):
    train_ids = patient_ids[0:num_patients_train]
    val_ids = patient_ids[(num_patients_train + 1):(num_patients_train + 1 + num_patients_val)]
    test_ids = patient_ids[(num_patients_train + 1 + num_patients_val + 1):(num_patients_train + 1 + num_patients_val + 1+ num_patients_test)]
else: 
    raise ValueError("Not enough patients for the number in split proposed")

# Get train, validation, and test designated slices 
test_slices = dp.get_slices_from_ids(test_ids, patient_id_groups)
random.shuffle(test_slices)

print(f"Number of slices in test subset: {len(test_slices)}")

test_ds = dp.MDSC508_Dataset(test_slices, input_path, target_path, input_data, target_data)

test_loader = DataLoader(test_ds, shuffle = True)

# Test Loop:
model.eval() # Set model to evaluation mode
total_test_ssim = 0.0

with torch.no_grad():
    for batch in test_loader:
        inputs = batch['input'].to(device)
        target = batch['target'].to(device) # Will probably need to change datatype

        # Forward pass
        outputs = model(inputs)

        # Calculate the loss
        loss = ssim(outputs, target)
        print(loss)

        fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (6, 8))
        axes[0].imshow(outputs[0, 0, :, :].cpu().detach().numpy(), cmap = 'gray')
        axes[1].imshow(target[0, 0, :, :].cpu().detach().numpy(), cmap = 'gray')
        plt.tight_layout()
        plt.show()

        total_test_ssim += loss.item()

# Valvulate average validation loss
avg_test_ssim = total_test_ssim / len(test_loader)

# Print current epoch and losses
print(f'Average Test Loss: {avg_test_ssim:.4f}')

print(f'Finished testing')
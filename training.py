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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2E-Model Training")

    parser.add_argument('--experiment_name', type = str, help = 'Experiment name')
    parser.add_argument('--num_epochs', type = int, help = 'Number of epochs')
    parser.add_argument('--optimizer_lr', type = float, default = 0.001, help = 'Adam optimizer learning rate')
    parser.add_argument('--batch_size', type = int, default = 32, help = 'Batch size to use during training')

    args = parser.parse_args()

    output_file = open(args.experiment_name + ".txt", 'w')
    sys.stdout = output_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {torch.cuda.get_device_name(device)}")

model = E2EVarNet(device)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr = args.optimizer_lr)

ssim = StructuralSimilarityIndexMeasure(data_range = (0, 1)).to(device)


seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)

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
train_slices = dp.get_slices_from_ids(train_ids, patient_id_groups)
random.shuffle(train_slices)
val_slices = dp.get_slices_from_ids(val_ids, patient_id_groups)
random.shuffle(val_slices)
test_slices = dp.get_slices_from_ids(test_ids, patient_id_groups)
random.shuffle(test_slices)

print(f"Number of slices in training subset: {len(train_slices)}")
print(f"Number of slices in validation subset: {len(val_slices)}")
print(f"Number of slices in test subset: {len(test_slices)}")

train_ds = dp.MDSC508_Dataset(train_slices, input_path, target_path, input_data, target_data)
val_ds = dp.MDSC508_Dataset(val_slices, input_path, target_path, input_data, target_data)
test_ds = dp.MDSC508_Dataset(test_slices, input_path, target_path, input_data, target_data)

batch_size = args.batch_size
epochs = args.num_epochs

train_loader = DataLoader(train_ds, batch_size, shuffle = True)
val_loader = DataLoader(val_ds, batch_size, shuffle = True)
test_loader = DataLoader(test_ds, batch_size, shuffle = True)

for epoch in range(epochs):

    model.train()
    train_loss = 0.0
    
    for i, batch in enumerate(train_loader, 0):
        inputs = batch['input'].to(device)
        target = batch['target'].to(device) # Will probably need to change datatype

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(inputs)

        # fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (6, 8))
        # axes[0].imshow(output[0, 0, :, :].cpu().detach().numpy(), cmap = 'gray')
        # axes[1].imshow(target[0, 0, :, :].cpu().detach().numpy(), cmap = 'gray')
        # plt.tight_layout()
        # plt.show()

        # Calculate the loss
        loss = 1.0 - ssim(output, target)
        # print(loss)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation Loop:
    model.eval() # Set model to evaluation mode
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['input'].to(device)
            target = batch['target'].to(device) # Will probably need to change datatype

            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            loss = 1.0 - ssim(outputs, target)

            val_loss += loss.item()
    
    # Valvulate average validation loss
    avg_val_loss = val_loss / len(val_loader)

    # Print current epoch and losses
    print(f'Epoch {epoch + 1}/{epochs} => Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

print(f'Finished training. Saving model ...')

torch.save(model, args.experiment_name + ".pth")

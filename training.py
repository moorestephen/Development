import torch
import random
from torchmetrics.image import StructuralSimilarityIndexMeasure
from monai.losses.ssim_loss import SSIMLoss
import argparse
from e2e_varnet import E2EVarNet
import torch.optim as optim
import sys
import data_preparation as dp
import os
import sys
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2E-Model Training")

    parser.add_argument('--experiment_name', type = str, help = 'Experiment name')
    parser.add_argument('--num_epochs', type = int, help = 'Number of epochs')
    parser.add_argument('--optimizer_lr', type = float, default = 0.001, help = 'Adam optimizer learning rate')
    parser.add_argument('--batch_size', type = int, default = 32, help = 'Batch size to use during training')
    parser.add_argument('--ratio_pathological', type = float, default = 0.0, help = 'Ratio of training and validation datasets which are pathological data')

    args = parser.parse_args()

print(f'Experiment name: {args.experiment_name}')
print(f'Number of epochs: {args.num_epochs}')
print(f'Initial Optimizer LR: {args.optimizer_lr}')
print(f'Batch Size: {args.batch_size}')
print(f'Ratio Pathological: {args.ratio_pathological}')
print()

# Device management
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {torch.cuda.get_device_name(device)}")

# Model initialization and optimizer selection
model = E2EVarNet(device)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr = args.optimizer_lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones = [25], gamma = 0.1
    # optimizer, milestones = [15, 30, 100], gamma = 0.5
)
torch.autograd.set_detect_anomaly(True) # To try to spit out when NaN comes in

# Initialize ssim for loss
ssim = StructuralSimilarityIndexMeasure().to(device)

# Random seeding
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

# Dataset preparation:
#-----------------------------

healthy_input_path = "/home/stephen.moore/MDSC508/Data/NormalData/Train_H5/slices"
healthy_target_path = "/home/stephen.moore/MDSC508/Data/NormalData/Train_H5/target_slice_images"
healthy_input_data = os.listdir(healthy_input_path)
healthy_target_data = os.listdir(healthy_target_path)

# Max num vols in train/val subsets (healthy) --> determined by pathological dataset size
max_num_healthy_vols_train = 48 
max_num_healthy_vols_val = 12

num_healthy_vols_train = int(max_num_healthy_vols_train * (1.0 - args.ratio_pathological))
num_healthy_vols_val = int(max_num_healthy_vols_val * (1.0 - args.ratio_pathological))

healthy_id_groups = dp.subset_data_by_patient_id(healthy_input_data)
healthy_ids = list(healthy_id_groups.keys())

random.shuffle(healthy_ids) # Shuffle patient ids for random distribution

if (num_healthy_vols_train + num_healthy_vols_val <= len(healthy_ids)):
    healthy_train_ids = healthy_ids[0:num_healthy_vols_train]
    healthy_val_ids = healthy_ids[num_healthy_vols_train:(num_healthy_vols_train + num_healthy_vols_val)]
else: 
    raise ValueError("Not enough patients for the number in split proposed")

# Get train and validation designated healthy slices 
healthy_train_slices = dp.get_slices_from_ids(healthy_train_ids, healthy_id_groups)
random.shuffle(healthy_train_slices)
healthy_val_slices = dp.get_slices_from_ids(healthy_val_ids, healthy_id_groups)
random.shuffle(healthy_val_slices)

healthy_train_ds = dp.MDSC508_Dataset(healthy_train_slices, healthy_input_path, healthy_target_path, healthy_input_data, healthy_target_data)
healthy_val_ds = dp.MDSC508_Dataset(healthy_val_slices, healthy_input_path, healthy_target_path, healthy_input_data, healthy_target_data)

print(f"Number of slices in healthy training subset: {len(healthy_train_ds)}")
print(f"Number of slices in healthy validation subset: {len(healthy_val_ds)}")

# GBM dataset preparation:
tvl_splits = {
    'test': ['e17391s3_P10752', 'e17447s3_P13824', 'e17406s3_P01536', 'e17390s3_P03584', 
    'e17322s3_P19968', 'e17353s3_P08704', 'e17474s3_P12288', 'e17315s3_P58368', 'e17647s3_P03584'],
    'validate' : ['e17565s3_P30720', 'e17349s6_P15360', 'e17595s3_P19968', 'e17396s3_P13824', 
    'e17614s3_P15872', 'e17410s3_P37888'],
    'train' : ['e17573s3_P31232', 'e17757s3_P15360', 'e17385s3_P03584', 'e17448s3_P20992', 
    'e17264s9_P25600', 'e17658s3_P03584', 'e17282s3_P28160', 'e17758s4_P22528', 'e17559s3_P24064', 
    'e17346s3_P42496', 'e17660s3_P01536', 'e17600s3_P03584', 'e17786s3_P12800', 'e17785s3_P05632', 
    'e17626s3_P12800', 'e17420s3_P07680', 'e17609s3_P27136', 'e17424s3_P10240', 'e17431s3_P23040', 
    'e17638s3_P11776', 'e17756s3_P08704', 'e17480s3_P19968', 'e17553s3_P20480', 'e17307s3_P03584']
}

gbm_data_directory = '/home/stephen.moore/MDSC508/Data/PathologicalData'
if args.ratio_pathological == 0.5:
    patho_pull_ratio = 1.0
elif args.ratio_pathological == 0.25:
    patho_pull_ratio = 0.5
elif args.ratio_pathological == 0.0:
    patho_pull_ratio = 0.0

if patho_pull_ratio != 0.0:
    gbm_train_ds = dp.GBMDataset(gbm_data_directory, tvl_splits['train'], patho_pull_ratio)
    gbm_val_ds = dp.GBMDataset(gbm_data_directory, tvl_splits['validate'], patho_pull_ratio)
    
    print(f'Number of slices in pathological training subset: {len(gbm_train_ds)}')
    print(f'Number of slices in pathological validation subset: {len(gbm_val_ds)}')

    train_ds = ConcatDataset([healthy_train_ds, gbm_train_ds])
    val_ds = ConcatDataset([healthy_val_ds, gbm_val_ds])
else:
    train_ds = healthy_train_ds
    val_ds = healthy_val_ds

# Prepare for model training:
#----------------------------

# Setting training hyperparameters
batch_size = args.batch_size
epochs = args.num_epochs
best_val_loss = 2.0
patience = 10
current_patience = 0

train_loader = DataLoader(train_ds, batch_size, shuffle = True)
val_loader = DataLoader(val_ds, batch_size, shuffle = True)

training_losses = []
validation_losses = []

for epoch in range(epochs):

    model.train()
    train_loss = 0.0
    
    for i, batch in enumerate(train_loader, start = 0):
        inputs = batch['input'].to(device)
        target = batch['target'].to(device) # Will probably need to change datatype

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(inputs)

        # Calculate the loss
        loss = 1 - ssim(output, target)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # avg_train_loss = train_loss / len(train_loader)
    train_loss /= i + 1

    # Validation Loop:
    model.eval() # Set model to evaluation mode
    val_loss = 0.0

    with torch.no_grad():
        for i, batch in enumerate(val_loader, start = 0):
            inputs = batch['input'].to(device)
            target = batch['target'].to(device) 

            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            loss = 1 - ssim(outputs, target)
            # loss = -ssim(outputs, target)

            val_loss += loss.item()
    
        # Valvulate average validation loss
        val_loss /= i + 1

    # Print current epoch and losses
    print(f'Epoch {epoch + 1} => Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    training_losses.append(train_loss)
    validation_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # current_patience = 0
        torch.save(model, args.experiment_name + "_BEST_MODEL.pth")
    # else:
    #     current_patience += 1

    # if current_patience >= patience:
    #     print(f'Early stopping at epoch {epoch}')
    #     break

    scheduler.step()

print(f'Finished training. Saving model ...')

torch.save(model, args.experiment_name + "_COMPLETE.pth")

plt.plot(range(1, epochs + 1), training_losses, label = 'Training Loss')
plt.plot(range(1, epochs + 1), validation_losses, label = 'Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()
plt.savefig(f'{os.getcwd()}/{args.experiment_name}_LossPlot.png')

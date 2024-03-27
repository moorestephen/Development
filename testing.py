import torch
import random
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
import argparse
from e2e_varnet import E2EVarNet
import torch.optim as optim
import sys
import data_preparation as dp
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp 
import pandas as pd
import statistics
import csv

# cwd = os.getcwd()
# new_dir = "FinalExpArm3_HalfPatho_TestRecon"
# path = os.path.join(cwd, new_dir)
# os.mkdir(path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {torch.cuda.get_device_name(device)}")

# First iteration of results:
# arm1_model = torch.load('FinalExpArm1_NoPatho_COMPLETE.pth')
# arm1_model.to(device)
# arm2_model = torch.load('FinalExpArm2_QuarterPatho_BEST_MODEL.pth')
# arm2_model.to(device)
# arm3_model = torch.load('FinalExpArm3_HalfPatho_V2_BEST_MODEL.pth')
# arm3_model.to(device)

arm1_model = torch.load('FinalExpArm1_NoPatho_COMPLETE.pth')
arm1_model.to(device)
arm2_model = torch.load('FinalExpArm2_QuarterPatho_V3_BEST_MODEL.pth')
arm2_model.to(device)
arm3_model = torch.load('FinalExpArm3_HalfPatho_BEST_MODEL.pth')
arm3_model.to(device)

ssim = StructuralSimilarityIndexMeasure().to(device)
psnr = PeakSignalNoiseRatio().to(device)

input_path = "/home/stephen.moore/MDSC508/Data/NormalData/Test_H5/slices"
target_path = "/home/stephen.moore/MDSC508/Data/NormalData/Test_H5/target_slice_images"
input_data = os.listdir(input_path)
target_data = os.listdir(target_path)

# Num patients in train/val/test subsets - small for preliminary results
num_patients_test = 9

patient_id_groups = dp.subset_data_by_patient_id(input_data)
patient_ids = list(patient_id_groups.keys())

# random.shuffle(patient_ids) # Shuffle patient ids for random distribution

test_ids = patient_ids[:num_patients_test]

# Get train, validation, and test designated slices 
test_slices = dp.get_slices_from_ids(test_ids, patient_id_groups)
random.shuffle(test_slices)

print(f"Number of healthy slices in test subset: {len(test_slices)}")

test_ds = dp.MDSC508_Dataset(test_slices, input_path, target_path, input_data, target_data)

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

GBM_test_ds = dp.GBMDataset(gbm_data_directory, tvl_splits['test'], 1.0)
test_loader_healthy = DataLoader(test_ds, shuffle = True)
test_loader_pathological = DataLoader(GBM_test_ds, shuffle = True)

# Test Loop:
arm1_model.eval() # Set model to evaluation mode
arm2_model.eval()
arm3_model.eval()
h_test_ssim_arm1 = []
h_test_ssim_arm2 = []
h_test_ssim_arm3 = []
h_test_psnr_arm1 = []
h_test_psnr_arm2 = []
h_test_psnr_arm3 = []
p_test_ssim_arm1 = []
p_test_ssim_arm2 = []
p_test_ssim_arm3 = []
p_test_psnr_arm1 = []
p_test_psnr_arm2 = []
p_test_psnr_arm3 = []
healthy_number_plotted = 0
pathological_number_plotted = 0

with torch.no_grad():

    # Healthy Data
    for batch in test_loader_healthy:
        inputs = batch['input'].to(device)
        target = batch['target'].to(device) # Will probably need to change datatype

        # Forward pass
        h_outputs_arm1 = arm1_model(inputs)
        h_outputs_arm2 = arm2_model(inputs)
        h_outputs_arm3 = arm3_model(inputs)

        # Calculate SSIM
        h_ssim_arm1 = ssim(h_outputs_arm1, target)
        h_ssim_arm2 = ssim(h_outputs_arm2, target)
        h_ssim_arm3 = ssim(h_outputs_arm3, target)

        # Calculate pSNR
        h_psnr_arm1 = psnr(h_outputs_arm1, target)
        h_psnr_arm2 = psnr(h_outputs_arm2, target)
        h_psnr_arm3 = psnr(h_outputs_arm3, target)

        if healthy_number_plotted < 50:
            fig, axes = plt.subplots(nrows = 1, ncols = 4, dpi = 1200)
            axes[0].imshow(h_outputs_arm1[0, 0, :, :].cpu().detach().numpy(), cmap = 'gray')
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            # axes[0].text(0.03, 0.97, f"SSIM = {h_ssim_arm1.item()}\npSNR = {h_psnr_arm1.item()}",
            #              transform = plt.gca().transAxes, ha = 'left', va = 'top')
            axes[1].imshow(h_outputs_arm2[0, 0, :, :].cpu().detach().numpy(), cmap = 'gray')
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            # axes[0].text(0.03, 0.97, f"SSIM = {h_ssim_arm2.item()}\npSNR = {h_psnr_arm2.item()}",
            #              transform = plt.gca().transAxes, ha = 'left', va = 'top')
            axes[2].imshow(h_outputs_arm3[0, 0, :, :].cpu().detach().numpy(), cmap = 'gray')
            axes[2].set_xticks([])
            axes[2].set_yticks([])
            # axes[0].text(0.03, 0.97, f"SSIM = {h_ssim_arm3.item()}\npSNR = {h_psnr_arm3.item()}",
            #              transform = plt.gca().transAxes, ha = 'left', va = 'top')
            axes[3].imshow(target[0, 0, :, :].cpu().detach().numpy(), cmap = 'gray')
            axes[3].set_xticks([])
            axes[3].set_yticks([])
            fig.tight_layout()
            fig.savefig(f'{os.getcwd()}/Results_v6/Healthy_Reconstruction_{healthy_number_plotted + 1}.pdf', dpi = 1200)
            plt.close(fig)
            print(f'Healthy Reconstruction {healthy_number_plotted + 1} SSIMs: {h_ssim_arm1.item()}, {h_ssim_arm2.item()}, {h_ssim_arm3.item()}')
            print(f'Healthy Reconstruction {healthy_number_plotted + 1} pSNRs: {h_psnr_arm1.item()}, {h_psnr_arm2.item()}, {h_psnr_arm3.item()}')
            healthy_number_plotted += 1

        h_test_ssim_arm1.append(h_ssim_arm1.item())
        h_test_ssim_arm2.append(h_ssim_arm2.item())
        h_test_ssim_arm3.append(h_ssim_arm3.item())

        h_test_psnr_arm1.append(h_psnr_arm1.item())
        h_test_psnr_arm2.append(h_psnr_arm2.item())
        h_test_psnr_arm3.append(h_psnr_arm3.item())

    # Pathological Data
    for batch in test_loader_pathological:
        inputs = batch['input'].to(device)
        target = batch['target'].to(device) # Will probably need to change datatype

        # Forward pass
        p_outputs_arm1 = arm1_model(inputs)
        p_outputs_arm2 = arm2_model(inputs)
        p_outputs_arm3 = arm3_model(inputs)

        # Calculate SSIM
        p_ssim_arm1 = ssim(p_outputs_arm1, target)
        p_ssim_arm2 = ssim(p_outputs_arm2, target)
        p_ssim_arm3 = ssim(p_outputs_arm3, target)

        # Calculate pSNR
        p_psnr_arm1 = psnr(p_outputs_arm1, target)
        p_psnr_arm2 = psnr(p_outputs_arm2, target)
        p_psnr_arm3 = psnr(p_outputs_arm3, target)

        if pathological_number_plotted < 75:
            fig, axes = plt.subplots(nrows = 1, ncols = 4, dpi = 1200)
            axes[0].imshow(p_outputs_arm1[0, 0, :, :].cpu().detach().numpy(), cmap = 'gray')
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            # axes[0].text(0.03, 0.97, f"SSIM = {p_ssim_arm1.item()}\npSNR = {p_psnr_arm1.item()}",
            #              transform = plt.gca().transAxes, ha = 'left', va = 'top')
            axes[1].imshow(p_outputs_arm2[0, 0, :, :].cpu().detach().numpy(), cmap = 'gray')
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            # axes[1].text(0.03, 0.97, f"SSIM = {p_ssim_arm2.item()}\npSNR = {p_psnr_arm2.item()}",
            #              transform = plt.gca().transAxes, ha = 'left', va = 'top')
            axes[2].imshow(p_outputs_arm3[0, 0, :, :].cpu().detach().numpy(), cmap = 'gray')
            axes[2].set_xticks([])
            axes[2].set_yticks([])
            # axes[2].text(0.03, 0.97, f"SSIM = {p_ssim_arm3.item()}\npSNR = {p_psnr_arm3.item()}",
            #              transform = plt.gca().transAxes, ha = 'left', va = 'top')
            axes[3].imshow(target[0, 0, :, :].cpu().detach().numpy(), cmap = 'gray')
            axes[3].set_xticks([])
            axes[3].set_yticks([])
            fig.tight_layout()
            fig.savefig(f'{os.getcwd()}/Results_v6/Pathological_Reconstruction_{pathological_number_plotted + 1}.pdf', dpi = 1200)
            plt.close(fig)
            print(f'Pathological Reconstruction {pathological_number_plotted + 1} SSIMs: {p_ssim_arm1.item()}, {p_ssim_arm2.item()}, {p_ssim_arm3.item()}')
            print(f'Pathological Reconstruction {pathological_number_plotted + 1} pSNRs: {p_psnr_arm1.item()}, {p_psnr_arm2.item()}, {p_psnr_arm3.item()}')
            pathological_number_plotted += 1

        p_test_ssim_arm1.append(p_ssim_arm1.item())
        p_test_ssim_arm2.append(p_ssim_arm2.item())
        p_test_ssim_arm3.append(p_ssim_arm3.item())

        p_test_psnr_arm1.append(p_psnr_arm1.item())
        p_test_psnr_arm2.append(p_psnr_arm2.item())
        p_test_psnr_arm3.append(p_psnr_arm3.item())

# h_test_ssim_arm1 = h_test_ssim_arm1.cpu()
# h_test_ssim_arm2 = h_test_ssim_arm2.cpu()
# h_test_ssim_arm3 = h_test_ssim_arm3.cpu()

# h_test_psnr_arm1 = h_test_psnr_arm1.cpu()
# h_test_psnr_arm2 = h_test_psnr_arm2.cpu()
# h_test_psnr_arm3 = h_test_psnr_arm3.cpu()

# p_test_ssim_arm1 = p_test_ssim_arm1.cpu()
# p_test_ssim_arm2 = p_test_ssim_arm2.cpu()
# p_test_ssim_arm3 = p_test_ssim_arm3.cpu()

# p_test_psnr_arm1 = p_test_psnr_arm1.cpu()
# p_test_psnr_arm2 = p_test_psnr_arm2.cpu()
# p_test_psnr_arm3 = p_test_psnr_arm3.cpu()

def print_mean_med(arm_name : str, arm : list):
    print(f'{arm_name} median = {statistics.median(arm)}; mean = {statistics.mean(arm)}')

print_mean_med('h_ssim_arm1', h_test_ssim_arm1)
print_mean_med('h_ssim_arm2', h_test_ssim_arm2)
print_mean_med('h_ssim_arm3', h_test_ssim_arm3)

print_mean_med('h_psnr_arm1', h_test_psnr_arm1)
print_mean_med('h_psnr_arm2', h_test_psnr_arm2)
print_mean_med('h_psnr_arm3', h_test_psnr_arm3)

print_mean_med('p_ssim_arm1', p_test_ssim_arm1)
print_mean_med('p_ssim_arm2', p_test_ssim_arm2)
print_mean_med('p_ssim_arm3', p_test_ssim_arm3)

print_mean_med('p_psnr_arm1', p_test_psnr_arm1)
print_mean_med('p_psnr_arm2', p_test_psnr_arm2)
print_mean_med('p_psnr_arm3', p_test_psnr_arm3)

h_ssim_statistic, h_ssim_p_value = friedmanchisquare(
                                        h_test_ssim_arm1,
                                        h_test_ssim_arm2,
                                        h_test_ssim_arm3)
print(f'Friedman\'s Test (Healthy Reconstruction, SSIM): {h_ssim_statistic}')
print(f'P-value (Healthy Reconstruction, SSIM): {h_ssim_p_value}')

p_ssim_statistic, p_ssim_p_value = friedmanchisquare(
                                        p_test_ssim_arm1,
                                        p_test_ssim_arm2,
                                        p_test_ssim_arm3)
print(f'Friedman\'s Test (Pathological Reconstruction, SSIM): {p_ssim_statistic}')
print(f'P-value (Healthy Reconstruction, SSIM): {p_ssim_p_value}')

h_psnr_statistic, h_psnr_p_value = friedmanchisquare(
                                        h_test_psnr_arm1, 
                                        h_test_psnr_arm2,
                                        h_test_psnr_arm3)
print(f'Friedman\'s Test (Healthy Reconstruction, pSNR): {h_psnr_statistic}')
print(f'P-value (Healthy Reconstruction, pSNR): {h_psnr_p_value}')

p_psnr_statistic, p_psnr_p_value = friedmanchisquare(
                                        p_test_psnr_arm1, 
                                        p_test_psnr_arm2,
                                        p_test_psnr_arm3)
print(f'Friedman\'s Test (Pathological Reconstruction, pSNR): {p_psnr_statistic}')
print(f'P-value (Pathological Reconstruction, pSNR): {p_psnr_p_value}')

h_ssim_combined = [h_test_ssim_arm1, h_test_ssim_arm2, h_test_ssim_arm3]
p_ssim_combined = [p_test_ssim_arm1, p_test_ssim_arm2, p_test_ssim_arm3]

h_psnr_combined = [h_test_psnr_arm1, h_test_psnr_arm2, h_test_psnr_arm3]
p_psnr_combined = [p_test_psnr_arm1, p_test_psnr_arm2, p_test_psnr_arm3]

# h_ssim_ph_a1a2 = sp.posthoc_dunn(h_ssim_combined, p_adjust = 'bonferroni')
# print('Healthy Reconstruction SSIM PostHoc:')
# print(h_ssim_ph)

# p_ssim_ph = sp.posthoc_dunn(p_ssim_combined, p_adjust = 'bonferroni')
# print('Pathological Reconstruction SSIM PostHoc:')
# print(p_ssim_ph)

# h_psnr_ph = sp.posthoc_dunn(h_psnr_combined, p_adjust = 'bonferroni')
# print('Healthy Reconstruction pSNR PostHoc:')
# print(h_psnr_ph)

# p_psnr_ph = sp.posthoc_dunn(p_psnr_combined, p_adjust = 'bonferroni')
# print('Pathological Reconstruction pSNR PostHoc:')
# print(p_psnr_ph)

data_together = zip(h_test_ssim_arm1, h_test_ssim_arm2, h_test_ssim_arm3,
                    p_test_ssim_arm1, p_test_ssim_arm2, p_test_ssim_arm3,
                    h_test_psnr_arm1, h_test_psnr_arm2, h_test_psnr_arm3,
                    p_test_psnr_arm1, p_test_psnr_arm2, p_test_psnr_arm3)

results_csv = "reconstruction_performance_results.csv"
with open(results_csv, mode = 'w', newline = '') as file:
    writer = csv.writer(file)
    writer.writerow(['Healthy SSIM Arm 1', 'Healthy SSIM Arm 2', 'Healthy SSIM Arm 3',
                     'Pathological SSIM Arm 1', 'Pathological SSIM Arm 2', 'Pathological SSIM Arm 3',
                     'Healthy pSNR Arm 1', 'Healthy pSNR Arm 2', 'Healthy pSNR Arm 3',
                     'Pathological pSNR Arm 1', 'Pathological pSNR Arm 2', 'Pathological pSNR Arm 3'])
    for row in data_together:
        writer.writerow(row)


# Healthy SSIM

h_ssim_ph_a1a2 = wilcoxon(h_test_ssim_arm1, h_test_ssim_arm2)
print('Healthy Reconstruction SSIM PostHoc (Arm 1, Arm 2):')
print(h_ssim_ph_a1a2)

h_ssim_ph_a1a3 = wilcoxon(h_test_ssim_arm1, h_test_ssim_arm3)
print('Healthy Reconstruction SSIM PostHoc (Arm 1, Arm 3):')
print(h_ssim_ph_a1a3)

h_ssim_ph_a2a3 = wilcoxon(h_test_ssim_arm2, h_test_ssim_arm3)
print('Healthy Reconstruction SSIM PostHoc (Arm 2, Arm 3):')
print(h_ssim_ph_a2a3)

# Pathological SSIM

p_ssim_ph_a1a2 = wilcoxon(p_test_ssim_arm1, p_test_ssim_arm2)
print('Pathological Reconstruction SSIM PostHoc (Arm 1, Arm 2):')
print(p_ssim_ph_a1a2)

p_ssim_ph_a1a3 = wilcoxon(p_test_ssim_arm1, p_test_ssim_arm3)
print('Pathological Reconstruction SSIM PostHoc (Arm 1, Arm 3):')
print(p_ssim_ph_a1a3)

p_ssim_ph_a2a3 = wilcoxon(p_test_ssim_arm2, p_test_ssim_arm3)
print('Pathological Reconstruction SSIM PostHoc (Arm 2, Arm 3):')
print(p_ssim_ph_a2a3)

# Healthy pSNR

h_psnr_ph_a1a2 = wilcoxon(h_test_psnr_arm1, h_test_psnr_arm2)
print('Healthy Reconstruction pSNR PostHoc (Arm 1, Arm 2):')
print(h_psnr_ph_a1a2)

h_psnr_ph_a1a3 = wilcoxon(h_test_psnr_arm1, h_test_psnr_arm3)
print('Healthy Reconstruction pSNR PostHoc (Arm 1, Arm 3):')
print(h_psnr_ph_a1a3)

h_psnr_ph_a2a3 = wilcoxon(h_test_psnr_arm2, h_test_psnr_arm3)
print('Healthy Reconstruction pSNR PostHoc (Arm 2, Arm 3):')
print(h_psnr_ph_a2a3)

# Pathological pSNR

p_psnr_ph_a1a2 = wilcoxon(p_test_psnr_arm1, p_test_psnr_arm2)
print('Pathological Reconstruction pSNR PostHoc (Arm 1, Arm 2):')
print(p_psnr_ph_a1a2)

p_psnr_ph_a1a3 = wilcoxon(p_test_psnr_arm1, p_test_psnr_arm3)
print('Pathological Reconstruction pSNR PostHoc (Arm 1, Arm 3):')
print(p_psnr_ph_a1a3)

p_psnr_ph_a2a3 = wilcoxon(p_test_psnr_arm2, p_test_psnr_arm3)
print('Pathological Reconstruction pSNR PostHoc (Arm 2, Arm 3):')
print(p_psnr_ph_a2a3)

# np.savetxt(f'{os.getcwd()}/Results/h_ssim_arm1.csv', np.array(h_ssim_arm1), delimiter = ',')
# np.savetxt(f'{os.getcwd()}/Results/h_ssim_arm2.csv', np.array(h_ssim_arm2), delimiter = ',')
# np.savetxt(f'{os.getcwd()}/Results/h_ssim_arm3.csv', np.array(h_ssim_arm3), delimiter = ',')
# np.savetxt(f'{os.getcwd()}/Results/p_ssim_arm1.csv', np.array(p_ssim_arm1), delimiter = ',')
# np.savetxt(f'{os.getcwd()}/Results/p_ssim_arm2.csv', np.array(p_ssim_arm2), delimiter = ',')
# np.savetxt(f'{os.getcwd()}/Results/p_ssim_arm3.csv', np.array(p_ssim_arm3), delimiter = ',')

# np.savetxt(f'{os.getcwd()}/Results/h_psnr_arm1.csv', np.array(h_psnr_arm1), delimiter = ',')
# np.savetxt(f'{os.getcwd()}/Results/h_psnr_arm2.csv', np.array(h_psnr_arm2), delimiter = ',')
# np.savetxt(f'{os.getcwd()}/Results/h_psnr_arm3.csv', np.array(h_psnr_arm3), delimiter = ',')
# np.savetxt(f'{os.getcwd()}/Results/p_psnr_arm1.csv', np.array(p_psnr_arm1), delimiter = ',')
# np.savetxt(f'{os.getcwd()}/Results/p_psnr_arm2.csv', np.array(p_psnr_arm2), delimiter = ',')
# np.savetxt(f'{os.getcwd()}/Results/p_psnr_arm3.csv', np.array(p_psnr_arm3), delimiter = ',')

fig = plt.figure(dpi = 1200)
plt.boxplot([h_test_ssim_arm1, h_test_ssim_arm2, h_test_ssim_arm3], labels = ['Arm 1', 'Arm 2', 'Arm 3'])
# plt.title('Reconstruction Performance (SSIM) of Healthy Data by Experimental Arm Model')
plt.xlabel('Experimental Arm')
plt.ylabel('Reconstruction Performance (SSIM)')
plt.savefig(f'{os.getcwd()}/Results_v6/h_ssim_boxplot.pdf', dpi = 1200)

fig = plt.figure(dpi = 1200)
plt.boxplot([p_test_ssim_arm1, p_test_ssim_arm2, p_test_ssim_arm3], labels = ['Arm 1', 'Arm 2', 'Arm 3'])
# plt.title('Reconstruction Performance (SSIM) of Pathological Data by Experimental Arm Model')
plt.xlabel('Experimental Arm')
plt.ylabel('Reconstruction Performance (SSIM)')
plt.savefig(f'{os.getcwd()}/Results_v6/p_ssim_boxplot.pdf', dpi = 1200)

fig = plt.figure(dpi = 1200)
plt.boxplot([h_test_psnr_arm1, h_test_psnr_arm2, h_test_psnr_arm3], labels = ['Arm 1', 'Arm 2', 'Arm 3'])
# plt.title('Reconstruction Performance (pSNR) of Healthy Data by Experimental Arm Model')
plt.xlabel('Experimental Arm')
plt.ylabel('Reconstruction Performance (pSNR)')
plt.savefig(f'{os.getcwd()}/Results_v6/h_psnr_boxplot.pdf', dpi = 1200)

fig = plt.figure(dpi = 1200)
plt.boxplot([p_test_psnr_arm1, p_test_psnr_arm2, p_test_psnr_arm3], labels = ['Arm 1', 'Arm 2', 'Arm 3'])
# plt.title('Reconstruction Performance (pSNR) of Healthy Data by Experimental Arm Model')
plt.xlabel('Experimental Arm')
plt.ylabel('Reconstruction Performance (pSNR)')
plt.savefig(f'{os.getcwd()}/Results_v6/p_psnr_boxplot.pdf', dpi = 1200)

print(f'Finished testing')
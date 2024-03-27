from scipy.stats import shapiro
import scikit_posthocs as sp 
import pandas as pd
import statistics
import csv
import matplotlib.pyplot as plt
import os

columns = []

with open('reconstruction_performance_results.csv', newline = '') as file:
    reader = csv.reader(file)

    headers = next(reader)

    for _ in headers:
        columns.append([])
    
    for row in reader:
        for i, value in enumerate(row):
            columns[i].append(float(value))

h_ssim_arm1 = columns[0]
h_ssim_arm2 = columns[1]
h_ssim_arm3 = columns[2]

p_ssim_arm1 = columns[3]
p_ssim_arm2 = columns[4]
p_ssim_arm3 = columns[5]

h_psnr_arm1 = columns[6]
h_psnr_arm2 = columns[7]
h_psnr_arm3 = columns[8]

p_psnr_arm1 = columns[9]
p_psnr_arm2 = columns[10]
p_psnr_arm3 = columns[11]

with open('results_statistics_output2.txt', 'w') as f:

    def print_results(name : str, data : list):
        median = statistics.median(data)
        data.sort()

        q1 = statistics.median(data[:len(data) // 2])
        q3 = statistics.median(data[len(data) // 2 + len(data) % 2:])
        iqr = q3 - q1

        statistic, p_value = shapiro(data)

        print(f'\n{name} Results:\nMedian = {median}; IQR = {q1} - {q3}\nShapiro-Wilks Statistic = {statistic}; P = {p_value}', file = f)

    print_results('Healthy SSIM Arm 1', h_ssim_arm1)
    print_results('Healthy SSIM Arm 2', h_ssim_arm2)
    print_results('Healthy SSIM Arm 3', h_ssim_arm3)

    print_results('Pathological SSIM Arm 1', p_ssim_arm1)
    print_results('Pathological SSIM Arm 2', p_ssim_arm2)
    print_results('Pathological SSIM Arm 3', p_ssim_arm3)  

    print_results('Healthy pSNR Arm 1', h_psnr_arm1)
    print_results('Healthy pSNR Arm 2', h_psnr_arm2)
    print_results('Healthy pSNR Arm 3', h_psnr_arm3)  

    print_results('Pathological pSNR Arm 1', p_psnr_arm1)
    print_results('Pathological pSNR Arm 2', p_psnr_arm2)
    print_results('Pathological pSNR Arm 3', p_psnr_arm3)      

significant_combinations = [[(1, 3), 1.0e-4], 
                            [(1, 2), 1.0e-4], 
                            [(2, 3), 1.0e-4]]

fig = plt.figure(dpi = 1200)
plt.boxplot([h_ssim_arm1, h_ssim_arm2, h_ssim_arm3], labels = ['Arm 1', 'Arm 2', 'Arm 3'])
plt.title('Reconstruction Performance (SSIM) of Healthy Data')
bottom, top = plt.gca().get_ylim()
y_range = top - bottom
for i, significant_combination in enumerate(significant_combinations):
    # Columns corresponding to the datasets of interest
    x1 = significant_combination[0][0]
    x2 = significant_combination[0][1]
    # What level is this bar among the bars above the plot?
    level = len(significant_combinations) - i
    # Plot the bar
    bar_height = (y_range * 0.07 * level) + top
    bar_tips = bar_height - (y_range * 0.02)
    plt.plot(
        [x1, x1, x2, x2],
        [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k'
    )
    # Significance level
    p = significant_combination[1]
    if p < 0.001:
        sig_symbol = '***'
    elif p < 0.01:
        sig_symbol = '**'
    elif p < 0.05:
        sig_symbol = '*'
    text_height = bar_height + (y_range * 0.01)
    plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k')
plt.xlabel('Experimental Arm')
plt.ylabel('SSIM')
plt.savefig(f'{os.getcwd()}/Results_v7/h_ssim_boxplot.pdf', dpi = 1200)

fig = plt.figure(dpi = 1200)
plt.boxplot([p_ssim_arm1, p_ssim_arm2, p_ssim_arm3], labels = ['Arm 1', 'Arm 2', 'Arm 3'])
plt.title('Reconstruction Performance (SSIM) of Pathological Data')
bottom, top = plt.gca().get_ylim()
y_range = top - bottom
for i, significant_combination in enumerate(significant_combinations):
    # Columns corresponding to the datasets of interest
    x1 = significant_combination[0][0]
    x2 = significant_combination[0][1]
    # What level is this bar among the bars above the plot?
    level = len(significant_combinations) - i
    # Plot the bar
    bar_height = (y_range * 0.07 * level) + top
    bar_tips = bar_height - (y_range * 0.02)
    plt.plot(
        [x1, x1, x2, x2],
        [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k'
    )
    # Significance level
    p = significant_combination[1]
    if p < 0.001:
        sig_symbol = '***'
    elif p < 0.01:
        sig_symbol = '**'
    elif p < 0.05:
        sig_symbol = '*'
    text_height = bar_height + (y_range * 0.01)
    plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k')
plt.xlabel('Experimental Arm')
plt.ylabel('SSIM')
plt.savefig(f'{os.getcwd()}/Results_v7/p_ssim_boxplot.pdf', dpi = 1200)

fig = plt.figure(dpi = 1200)
plt.boxplot([h_psnr_arm1, h_psnr_arm2, h_psnr_arm3], labels = ['Arm 1', 'Arm 2', 'Arm 3'])
plt.title('Reconstruction Performance (pSNR) of Healthy Data')
bottom, top = plt.gca().get_ylim()
y_range = top - bottom
for i, significant_combination in enumerate(significant_combinations):
    # Columns corresponding to the datasets of interest
    x1 = significant_combination[0][0]
    x2 = significant_combination[0][1]
    # What level is this bar among the bars above the plot?
    level = len(significant_combinations) - i
    # Plot the bar
    bar_height = (y_range * 0.07 * level) + top
    bar_tips = bar_height - (y_range * 0.02)
    plt.plot(
        [x1, x1, x2, x2],
        [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k'
    )
    # Significance level
    p = significant_combination[1]
    if p < 0.001:
        sig_symbol = '***'
    elif p < 0.01:
        sig_symbol = '**'
    elif p < 0.05:
        sig_symbol = '*'
    text_height = bar_height + (y_range * 0.01)
    plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k')
plt.xlabel('Experimental Arm')
plt.ylabel('pSNR')
plt.savefig(f'{os.getcwd()}/Results_v7/h_psnr_boxplot.pdf', dpi = 1200)

fig = plt.figure(dpi = 1200)
plt.boxplot([p_psnr_arm1, p_psnr_arm2, p_psnr_arm3], labels = ['Arm 1', 'Arm 2', 'Arm 3'])
plt.title('Reconstruction Performance (pSNR) of Pathological Data')
bottom, top = plt.gca().get_ylim()
y_range = top - bottom
for i, significant_combination in enumerate(significant_combinations):
    # Columns corresponding to the datasets of interest
    x1 = significant_combination[0][0]
    x2 = significant_combination[0][1]
    # What level is this bar among the bars above the plot?
    level = len(significant_combinations) - i
    # Plot the bar
    bar_height = (y_range * 0.07 * level) + top
    bar_tips = bar_height - (y_range * 0.02)
    plt.plot(
        [x1, x1, x2, x2],
        [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k'
    )
    # Significance level
    p = significant_combination[1]
    if p < 0.001:
        sig_symbol = '***'
    elif p < 0.01:
        sig_symbol = '**'
    elif p < 0.05:
        sig_symbol = '*'
    text_height = bar_height + (y_range * 0.01)
    plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', va='bottom', c='k')
plt.xlabel('Experimental Arm')
plt.ylabel('pSNR')
plt.savefig(f'{os.getcwd()}/Results_v7/p_psnr_boxplot.pdf', dpi = 1200)

'''
Description: 
version: 
Author: Rain
Date: 2023-08-09 16:10:43
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-12-02 23:16:22
'''
import os
import glob
import pandas as pd

# os.chdir("results/lipo/")
# os.chdir("results/atom_mask_0.75/t1_2")

os.chdir("results/readjustparameter/t1_2/")
files = os.listdir()
files = glob.glob("**/*_result.csv", recursive=True)
df = pd.concat((pd.read_csv(f) for f in files))


val_mse_loss = list(df.val_mae_loss)
# test_acc = list(df.test_acc)
experiment_name = list(files)


best_val_mse_loss = float('inf')
best_experiment_name = experiment_name[0]
for experiment, val_a in zip(experiment_name, val_mse_loss):
    if val_a < best_val_mse_loss:
        best_val_mse_loss = val_a
        best_experiment_name = experiment
        

print('experiment_name:', best_experiment_name, 'val_value:', best_val_mse_loss)


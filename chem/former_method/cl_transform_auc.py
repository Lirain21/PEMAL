'''
Author: Lirain21 17860633515@163.com
Date: 2023-10-13 00:26:51
LastEditors: Lirain21 17860633515@163.com
LastEditTime: 2023-10-13 00:26:51
FilePath: /Copy_GraphMAE_AUC/chem/cl_transform_auc.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas as pd 
import numpy as np 

# smiles,values,log_value,init_value 
cl_train_data = pd.read_csv('dataset/cl_train/raw/cl_train.csv')
cl_train_smiles = cl_train_data.smiles 
cl_train_values = cl_train_data.value

auc_train_data = pd.read_csv('dataset/auc_train/raw/auc_train.csv')
auc_train_smiles = auc_train_data.smiles 

auc_valid_data = pd.read_csv('dataset/auc_valid/raw/auc_valid.csv')
auc_valid_smiles = auc_valid_data.smiles 

auc_smiles = auc_train_smiles + auc_valid_smiles


aux_smiles = list(set(cl_train_smiles)-set(auc_smiles))
print(len(aux_smiles))
new_auc_smiles = []
aux_values = []
aux_labels = []
for smi in aux_smiles:
    # print(smi)
    index = cl_train_smiles.tolist().index(smi)
    # print(index)
    
    value = cl_train_values[index]
    # print(value)
    if 7.95 < 16842.0 / float(value) < 14400.0:
        aux_values.append(float(value))
        new_auc_smiles.append(smi)
        
        
    
    
aux_auc_values = 16842.0/np.array(aux_values)

for value in aux_auc_values:
    if value <= 500.0:
        aux_labels.append(0)
    elif 500.0 < value < 2500.0:
        aux_labels.append(1)
    else:
        aux_labels.append(2)
    

    
    
    
print(len(aux_auc_values))
# max 336840.0
# min 0.3602566844919786

print('max',max(aux_auc_values))
print('min', min(aux_auc_values))

count = 0
for i in range(len(aux_auc_values)):
    if aux_auc_values[i] > 14400:
        count += 1
    
print(count)


# new_auc_smiles 
# value = aux_values 
import math 

log_values = np.log(np.array(aux_values))
log_values = log_values.tolist()


init_values = ((np.log(np.array(aux_values)) - math.log(7.95))/(math.log(14787.0) - math.log(7.95))).tolist()

dataframe = pd.DataFrame({'smiles':new_auc_smiles, 'value':aux_values, 'labels':aux_labels,'log_values':log_values, 'init_values':init_values})
dataframe.to_csv('aux_auc_data.csv', index=None)








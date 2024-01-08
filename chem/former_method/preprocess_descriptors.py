'''
Author: Lirain21 17860633515@163.com
Date: 2023-10-12 02:55:08
LastEditors: Lirain21 17860633515@163.com
LastEditTime: 2023-11-13 07:22:49
FilePath: /Copy_GraphMAE_AUC/chem/preprocess.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from rdkit import DataStructs
from rdkit.Chem import (
    Mol,
    RDConfig,
    Descriptors,
    MolFromSmiles,
    rdFingerprintGenerator,
)
from rdkit.Chem.QED import qed
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import ExactMolWt, BertzCT

from rdkit.Chem import MACCSkeys

# fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
# fp_phaErGfp = AllChem.GetErGFingerprint(mol,fuzzIncrement=0.3,maxPath=21,minPath=1)
# fp_pubcfp = GetPubChemFPs(mol)

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys


import pandas as pd 
import numpy as np 

import torch 

# fingerprints = torch.from_numpy(pd.read_csv("dataset/auc_train/processed/auc_fingerprint_init.csv", header=None).to_numpy()[:,:])
# # descriptors = torch.from_numpy(pd.read_csv(os.path.join(self.processed_dir,'auc_init_descritors.csv'), header=None).to_numpy()[:,:])
# print(fingerprints.size())
# exit()


auc_train_data = pd.read_csv('dataset_reg/4_train/raw/4_train.csv')
auc_valid_data = pd.read_csv('dataset_reg/4_valid/raw/4_valid.csv')


# smiles,values,label,log_value,init_value 

auc_train_smiles = auc_train_data.smiles 
auc_valid_smiles = auc_valid_data.smiles 

# auc_train_values = auc_train_data.value
# auc_valid_values = auc_valid_data.value 

# auc_train_label = auc_train_data.label 
# auc_valid_label = auc_valid_data.label 

# auc_train_log_value = auc_train_data.log_value 
# auc_valid_log_value = auc_valid_data.log_value 

# auc_train_init_value = auc_train_data.init_value 
# auc_valid_init_value = auc_valid_data.init_value 

auc_train_len = len(auc_train_smiles)
auc_valid_len = len(auc_valid_smiles)


auc_train_fingerprint = []
auc_valid_fingerprint = []

auc_train_maccs = []
auc_valid_maccs = []

auc_train_descriptors = []
auc_valid_descriptors = []

auc_train_ergfingerprints = []
auc_valid_ergfingerprints = []

for smi in auc_train_smiles:
    rdkit_mol = MolFromSmiles(smi)
    
    per_fingerprint = rdFingerprintGenerator.GetCountFPs(
                [rdkit_mol], fpType=rdFingerprintGenerator.MorganFP
            )[0]
    fp_numpy = np.zeros((0,), np.int8)  # Generate target pointer to fill
    DataStructs.ConvertToNumpyArray(per_fingerprint, fp_numpy)
    per_fingerprint= fp_numpy.tolist()
    
    fp_phaErGfp = AllChem.GetErGFingerprint(rdkit_mol,fuzzIncrement=0.3, maxPath=21, minPath=1)
      
    
    per_descriptor = []
    for descr in Descriptors._descList:
        _, descr_calc_fn = descr
        try:
            # print(descr_calc_fn(rdkit_mol))
         
            per_descriptor.append(descr_calc_fn(rdkit_mol))
        except Exception:
            print('Fail:', smi)

    per_maccs = list(MACCSkeys.GenMACCSKeys(rdkit_mol).ToBitString())
    for i in range(len(per_maccs)):
        per_maccs[i] = int(per_maccs[i])

    auc_train_fingerprint.append(per_fingerprint)
    auc_train_descriptors.append(per_descriptor)
    auc_train_maccs.append(per_maccs)
    auc_train_ergfingerprints.append(fp_phaErGfp)
    
 
for smi in auc_valid_smiles:
    rdkit_mol = MolFromSmiles(smi)
    
    per_fingerprint = rdFingerprintGenerator.GetCountFPs(
                [rdkit_mol], fpType=rdFingerprintGenerator.MorganFP
            )[0]
    fp_numpy = np.zeros((0,), np.int8)  # Generate target pointer to fill
    DataStructs.ConvertToNumpyArray(per_fingerprint, fp_numpy)
    per_fingerprint= fp_numpy.tolist()
    
    fp_phaErGfp = AllChem.GetErGFingerprint(rdkit_mol,fuzzIncrement=0.3, maxPath=21, minPath=1)
  
    per_descriptor = []
    for descr in Descriptors._descList:
        _, descr_calc_fn = descr
        try:
            per_descriptor.append(descr_calc_fn(rdkit_mol))
        except Exception:
            print('Fail:', smi)
        
    per_maccs = list(MACCSkeys.GenMACCSKeys(rdkit_mol).ToBitString())
    for i in range(len(per_maccs)):
        per_maccs[i] = int(per_maccs[i])
    
    # print(per_maccs)
    auc_valid_fingerprint.append(per_fingerprint)
    auc_valid_descriptors.append(per_descriptor)
    auc_valid_maccs.append(per_maccs)
    auc_valid_ergfingerprints.append(fp_phaErGfp)

print(len(auc_train_fingerprint))



np.savetxt("4_train_fingerprint_init.csv", auc_train_fingerprint, delimiter=',')
np.savetxt("4_valid_fingerprint_init.csv", auc_valid_fingerprint, delimiter=',')

np.savetxt("4_train_init_descritors.csv", auc_train_descriptors, delimiter=',')
np.savetxt("4_valid_init_descritors.csv", auc_valid_descriptors, delimiter=',')

np.savetxt("4_train_init_maccs.csv", auc_train_maccs, delimiter=',')
np.savetxt("4_valid_init_maccs.csv", auc_valid_maccs, delimiter=',')


exit()

for smi in auc_train_smiles:
    rdkit_mol = MolFromSmiles(smi)
 
## TODO: normalize the finperprints and descriptors 
## add the train and valid data togather 
## then, cal the most small and biggest num of per dimension 
## execute normalization operation 

auc_fingerprints = auc_train_fingerprint+auc_valid_fingerprint
auc_descriptors = auc_train_descriptors+auc_valid_descriptors


# print(len(auc_fingerprints))
# print(len(auc_descriptors))

# auc_fingerprints_max = np.max(np.array(auc_fingerprints), axis=0)
# auc_fingerprints_min = np.min(np.array(auc_fingerprints), axis=0)

fingerprint_len = len(auc_fingerprints[0])
descriptor_len = len(auc_descriptors[0])

auc_fingerprints_min = []
auc_fingerprints_max = []

auc_descriptors_min = []
auc_descriptors_max = []

# for i in range(fingerprint_len):
#     per_auc_fingerprints = []
#     for fin in auc_fingerprints:
#         per_auc_fingerprints.append(fin[i])
    
#     per_min = min(per_auc_fingerprints)
#     per_max = max(per_auc_fingerprints)
#     auc_fingerprints_min.append(per_min)
#     auc_fingerprints_max.append(per_max)
    
for i in range(descriptor_len):
    per_auc_descriptors = []
    for fin in auc_descriptors:
        per_auc_descriptors.append(fin[i])
    
    per_min = min(per_auc_descriptors)
    per_max = max(per_auc_descriptors)
    auc_descriptors_min.append(per_min)
    auc_descriptors_max.append(per_max)
   


# fingerprint_init_list = [] 
# for fingerprint in auc_fingerprints:
#     per_init_fingerprint = []
#     for i in range(len(fingerprint)):
#         if fingerprint[i] - auc_fingerprints_min[i] != 0. and (auc_fingerprints_max[i] - auc_fingerprints_min[i]) != 0.: 
#             fingerprint_init = (fingerprint[i] - auc_fingerprints_min[i])/(auc_fingerprints_max[i] - auc_fingerprints_min[i])
#             per_init_fingerprint.append(fingerprint_init) 
#         else:
#             per_init_fingerprint.append(0.)
            
#     fingerprint_init_list.append(per_init_fingerprint)
    
    
descritor_init_list = []
for descriptor in auc_descriptors:
    per_init_descriptor = []
    for i in range(len(descriptor)):
        if (descriptor[i]-auc_descriptors_min[i]) != 0. and (auc_descriptors_max[i] - auc_descriptors_min[i]) != 0.:
            descriptor_init = (descriptor[i] - auc_descriptors_min[i])/(auc_descriptors_max[i] - auc_descriptors_min[i])
            per_init_descriptor.append(descriptor_init)
        else:
            per_init_descriptor.append(0.)
    descritor_init_list.append(per_init_descriptor)
    
# print(len(fingerprint_init_list))
# print(len(descritor_init_list))
    
# auc_train_init_fingerprints = fingerprint_init_list[:auc_train_len]
# auc_valid_init_fingerprints = fingerprint_init_list[auc_train_len:]

auc_train_init_dscriptors = descritor_init_list[:auc_train_len]
auc_valid_init_dscriptors = descritor_init_list[auc_train_len:]

# np.savetxt("auc_train_fingerprint_init.csv", auc_train_init_fingerprints, delimiter=',')
# np.savetxt("auc_valid_fingerprint_init.csv", auc_valid_init_fingerprints, delimiter=',')

# np.savetxt("auc_train_init_descritors.csv", auc_train_init_dscriptors, delimiter=',')
# np.savetxt("auc_valid_init_descritors.csv", auc_valid_init_dscriptors, delimiter=',')

# np.savetxt("auc_train_init_maccs.csv", auc_train_maccs, delimiter=',')
# np.savetxt("auc_valid_init_maccs.csv", auc_valid_maccs, delimiter=',')

print(len(auc_train_ergfingerprints))
print(len(auc_valid_ergfingerprints))


np.savetxt("auc_train_init_erg.csv", np.array(auc_train_ergfingerprints), delimiter=',')
np.savetxt("auc_valid_init_erg.csv", np.array(auc_valid_ergfingerprints), delimiter=',')


# np.savetxt("auc_train_fingerprint_init.csv", auc_train_fingerprint, delimiter=',')
# np.savetxt("auc_valid_fingerprint_init.csv", auc_valid_fingerprint, delimiter=',')

# np.savetxt("auc_train_init_descritors.csv", auc_train_descriptors, delimiter=',')
# np.savetxt("auc_valid_init_descritors.csv", auc_valid_descriptors, delimiter=',')










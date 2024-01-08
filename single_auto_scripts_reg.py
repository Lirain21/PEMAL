'''
Description: 
version: 
Author: Rain
Date: 2023-08-09 11:03:27
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-12-09 12:44:24
'''
import os
import subprocess

script_path = 'scripts/'
# save_path_1 = "res/auc"
device =2
seed = 4
# weight_decay = 0.00000001
# lr_cls = 0.001
# batch_size = 20

processes_file = []

# for decay in [0.0000000001,  0.000000001, 0.0000001]: #
for decay in [0.0001, 0.00001, 0.000001, 0.0000001]:
    for lr in [0.0005, 0.0001, 0.001, 0.005]:
        for batch_size in [128]:
            # for seed in [5, 6]: # 42 

                with open(os.path.join(script_path, 'batch_size'+'_'+str(batch_size)+'_'+'decay'+'_'+str(decay)+'_'+'lr'+'_'+str(lr)+'_'+'script.sh'), 'w') as f:  # +'_'+'seed'+'_'+str(seed)

                    f.write("#!/bin/bash\n")
                    f.write("batch_size=" + str(batch_size) + "\n")
                    f.write("seed=" +str(seed) + "\n")
                    f.write("decay=" + str(decay) + "\n")
                    f.write("lr=" + str(lr) + "\n")
                    f.write("device=" + str(device) + "\n")
                    
                    f.write("python chem/pfc_nopretrain.py " + "\\" + "\n")
                    f.write("--batch_size " + str(batch_size) + " \\" + "\n")
                    f.write("--seed " + str(seed) +" \\" + "\n")
                    f.write("--decay " + str(decay) + " \\" + "\n")
                    f.write("--lr " + str(lr) + " \\" + "\n")                
                    f.write("--device " + str(device) + " \\" + "\n")

                os.chmod(os.path.join(script_path, 'batch_size'+'_'+str(batch_size)+'_'+'decay'+'_'+str(decay)+'_'+'lr'+'_'+str(lr)+'_''script.sh'), 0o755)   # +'seed'+'_'+str(seed)+'_'+
                processes_file.append(os.path.join(script_path, 'batch_size'+'_'+str(batch_size)+'_'+'decay'+'_'+str(decay)+'_'+'lr'+'_'+str(lr)+'_'+'script.sh'))  # +'_'+'seed'+'_'+str(seed)

for file in processes_file:
    subprocess.Popen(['sh', file])

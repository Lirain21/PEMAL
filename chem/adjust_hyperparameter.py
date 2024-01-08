'''
Description: 
version: 
Author: Rain
Date: 2023-08-09 11:03:27
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-01-01 20:57:04
'''
import os
import subprocess

script_path = 'scripts/'
# save_path_1 = "res/auc"
device = 0
seed = 4
# weight_decay = 0.00000001
# lr_cls = 0.001
# batch_size = 20

processes_file = []

for t1_2_model_scale in [0.01, 0.05, 0.1, 0.5, 1.5, 2.0, 2.5, 3., 3.5, 4.0, 4.5, 5.]:
                with open(os.path.join(script_path, 't1_2_model_scale'+'_'+str(t1_2_model_scale)+'_'+'script.sh'), 'w') as f:  # +'_'+'seed'+'_'+str(seed)

                    f.write("#!/bin/bash\n")
                    f.write("t1_2_model_scale=" + str(t1_2_model_scale) + "\n")
                    f.write("device=" + str(device) + "\n")
                    
                    f.write("python chem/physical_equation_4_tasks_con.py " + "\\" + "\n")
                    f.write("--t1_2_model_scale " + str(t1_2_model_scale) + " \\" + "\n")              
                    f.write("--device " + str(device) + " \\" + "\n")

                os.chmod(os.path.join(script_path, 't1_2_model_scale'+'_'+str(t1_2_model_scale)+'_''script.sh'), 0o755)   # +'seed'+'_'+str(seed)+'_'+
                processes_file.append(os.path.join(script_path, 't1_2_model_scale'+'_'+str(t1_2_model_scale)+'_'+'script.sh'))  # +'_'+'seed'+'_'+str(seed)

for file in processes_file:
    subprocess.Popen(['sh', file])

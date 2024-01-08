'''
Description: 
version: 
Author: Rain
Date: 2023-08-09 11:03:27
LastEditors: Rain
LastEditTime: 2023-08-09 16:18:27
'''
import os
import subprocess
import glob
import pandas as pd
import re 

# os.chdir("checkpoints/")
# files = os.listdir()
# files = glob.glob("**/*.pth", recursive=True)

# save_part_name=[]
# save_path_name = []
# scripts_path_list = []
# checkpoint_path_list =[]
# save_path = "results/auc"
# script_path = 'scripts/auc'
# for file in files:
#     new_file_name = file.replace("/", "")
#     new_file_name = new_file_name.replace(".pth","")
#     # new_file_name = new_file_name+"/"
#     checkpoint_path = os.path.join("checkpoint", file) 
#     script_name = os.path.join(script_path, new_file_name) # script_path+new_file_name
#     save_name = os.path.join(save_path, new_file_name) # save_path+new_file_name 
#     checkpoint_path_list.append(checkpoint_path)
#     save_part_name.append(new_file_name)
#     save_path_name.append(save_name)
#     scripts_path_list.append(script_name)
 

# print(scripts_path_list)

# print('******')
# print('******')

# print(save_path_name)

# print('******')
# print('******')

# print(checkpoint_path_list)
# exit()

scripts_path_list = ['scripts/auc/ginbest_model', 'scripts/auc/gin_50', 'scripts/auc/gin', 'scripts/auc/gin_100', 'scripts/auc/lr_0.0001_gin_110', 'scripts/auc/lr_0.0001_gin_140', 'scripts/auc/lr_0.0001_gin_40', 'scripts/auc/lr_0.0001_gin_150', 'scripts/auc/lr_0.0001_gin_120', 'scripts/auc/lr_0.0001_gin_130', 'scripts/auc/lr_0.0001_gin_160', 'scripts/auc/lr_0.0001_gin_100', 'scripts/auc/lr_0.0001_gin_2', 'scripts/auc/lr_0.0001_gin_1', 'scripts/auc/lr_0.0001_gin_180', 'scripts/auc/lr_0.0001_gin_60', 'scripts/auc/lr_0.0001_gin_10', 'scripts/auc/lr_0.0001_gin_190', 'scripts/auc/lr_0.0001_gin_170', 'scripts/auc/lr_0.0001_gin_20', 'scripts/auc/lr_0.0001_gin_90', 'scripts/auc/lr_0.0001_gin_30', 'scripts/auc/lr_0.0001_gin_80', 'scripts/auc/lr_0.0001_gin_70', 'scripts/auc/lr_0.0001_gin_200', 'scripts/auc/lr_0.0001_gin_50', 'scripts/auc/lr_0.001_gin_40', 'scripts/auc/lr_0.001_gin_100', 'scripts/auc/lr_0.001_gin_2', 'scripts/auc/lr_0.001_gin_1', 'scripts/auc/lr_0.001_gin_60', 'scripts/auc/lr_0.001_gin_10', 'scripts/auc/lr_0.001_gin_20', 'scripts/auc/lr_0.001_gin_90', 'scripts/auc/lr_0.001_gin_30', 'scripts/auc/lr_0.001_gin_80', 'scripts/auc/lr_0.001_gin_70', 'scripts/auc/lr_0.001_gin_50']
save_path_name = ['results/auc/ginbest_model', 'results/auc/gin_50', 'results/auc/gin', 'results/auc/gin_100', 'results/auc/lr_0.0001_gin_110', 'results/auc/lr_0.0001_gin_140', 'results/auc/lr_0.0001_gin_40', 'results/auc/lr_0.0001_gin_150', 'results/auc/lr_0.0001_gin_120', 'results/auc/lr_0.0001_gin_130', 'results/auc/lr_0.0001_gin_160', 'results/auc/lr_0.0001_gin_100', 'results/auc/lr_0.0001_gin_2', 'results/auc/lr_0.0001_gin_1', 'results/auc/lr_0.0001_gin_180', 'results/auc/lr_0.0001_gin_60', 'results/auc/lr_0.0001_gin_10', 'results/auc/lr_0.0001_gin_190', 'results/auc/lr_0.0001_gin_170', 'results/auc/lr_0.0001_gin_20', 'results/auc/lr_0.0001_gin_90', 'results/auc/lr_0.0001_gin_30', 'results/auc/lr_0.0001_gin_80', 'results/auc/lr_0.0001_gin_70', 'results/auc/lr_0.0001_gin_200', 'results/auc/lr_0.0001_gin_50', 'results/auc/lr_0.001_gin_40', 'results/auc/lr_0.001_gin_100', 'results/auc/lr_0.001_gin_2', 'results/auc/lr_0.001_gin_1', 'results/auc/lr_0.001_gin_60', 'results/auc/lr_0.001_gin_10', 'results/auc/lr_0.001_gin_20', 'results/auc/lr_0.001_gin_90', 'results/auc/lr_0.001_gin_30', 'results/auc/lr_0.001_gin_80', 'results/auc/lr_0.001_gin_70', 'results/auc/lr_0.001_gin_50']
checkpoint_path_list = ['checkpoint/ginbest_model.pth', 'checkpoint/gin_50.pth', 'checkpoint/gin.pth', 'checkpoint/gin_100.pth', 'checkpoint/lr_0.0001/_gin_110.pth', 'checkpoint/lr_0.0001/_gin_140.pth', 'checkpoint/lr_0.0001/_gin_40.pth', 'checkpoint/lr_0.0001/_gin_150.pth', 'checkpoint/lr_0.0001/_gin_120.pth', 'checkpoint/lr_0.0001/_gin_130.pth', 'checkpoint/lr_0.0001/_gin_160.pth', 'checkpoint/lr_0.0001/_gin_100.pth', 'checkpoint/lr_0.0001/_gin_2.pth', 'checkpoint/lr_0.0001/_gin_1.pth', 'checkpoint/lr_0.0001/_gin_180.pth', 'checkpoint/lr_0.0001/_gin_60.pth', 'checkpoint/lr_0.0001/_gin_10.pth', 'checkpoint/lr_0.0001/_gin_190.pth', 'checkpoint/lr_0.0001/_gin_170.pth', 'checkpoint/lr_0.0001/_gin_20.pth', 'checkpoint/lr_0.0001/_gin_90.pth', 'checkpoint/lr_0.0001/_gin_30.pth', 'checkpoint/lr_0.0001/_gin_80.pth', 'checkpoint/lr_0.0001/_gin_70.pth', 'checkpoint/lr_0.0001/_gin_200.pth', 'checkpoint/lr_0.0001/_gin_50.pth', 'checkpoint/lr_0.001/_gin_40.pth', 'checkpoint/lr_0.001/_gin_100.pth', 'checkpoint/lr_0.001/_gin_2.pth', 'checkpoint/lr_0.001/_gin_1.pth', 'checkpoint/lr_0.001/_gin_60.pth', 'checkpoint/lr_0.001/_gin_10.pth', 'checkpoint/lr_0.001/_gin_20.pth', 'checkpoint/lr_0.001/_gin_90.pth', 'checkpoint/lr_0.001/_gin_30.pth', 'checkpoint/lr_0.001/_gin_80.pth', 'checkpoint/lr_0.001/_gin_70.pth', 'checkpoint/lr_0.001/_gin_50.pth']
i = 0
# for checkpoint_file, script_file, save_file in zip(checkpoint_path_list, scripts_path_list, save_path_name):
checkpoint_file, script_file, save_file = checkpoint_path_list[37], scripts_path_list[37], save_path_name[37]
input_model_file = checkpoint_file

device = 5
count = 0
    
    # save = ''
    # if not os.path.exists(script_file):
    #     os.makedirs(script_file)
    #     print('***')
    #     print(i)
    #     print(checkpoint_file)
    #     print(script_file)
    #     print(save_file)
    #     i += 1

    #     print(len(scripts_path_list))
    #     print(len(save_path_name))
    #     print(len(checkpoint_path_list))
    #     if not os.path.exists(script_file):
    #         os.makedirs(script_file)
    #     if not os.path.exists(save_file):
    #         os.makedirs(save_file)

    # exit()
processes_file = []

# for input_model_file in 
for decay in [0.0000000001,  0.000000001, 0.00000000001]:
    for lr in [0.1, 0.01, 0.05]:
        for batch_size in [128, 256]:
            for seed in [6, 42]:
                if count > 0 and count % 8 == 0 and device<10:
                    device += 1
                count += 1
                # script_path = os.path.join(script_file, 'bz'+'_'+str(batch_size)+'_'+'decay'+'_'+str(decay)+'_'+'lr'+'_'+str(lr)+'_'+'seed'+'_'+str(seed)+'_'+'script.sh')
                
                with open(os.path.join(script_file, 'bz'+'_'+str(batch_size)+'_'+'decay'+'_'+str(decay)+'_'+'lr'+'_'+str(lr)+'_'+'seed'+'_'+str(seed)+'_'+'script.sh'), 'w') as f:
                    save = os.path.join(save_file,  'bz'+'_'+str(batch_size)+'_'+'decay'+'_'+str(decay)+'_'+'lr'+'_'+str(lr)+'_'+'seed'+'_'+str(seed))
                    # if not os.path.exists(dir_path):
                    #     os.makedirs(dir_path)

                    f.write("#!/bin/bash\n")
                    f.write("batch_size=" + str(batch_size) + "\n")
                    f.write("seed=" + str(seed) + "\n")
                    f.write("decay=" + str(decay) + "\n")
                    f.write("lr=" + str(lr) + "\n")
                    f.write("device=" + str(device) + "\n")
                    f.write("input_model_file=" + "\"" + input_model_file + "\"" + "\n")
                    f.write("save=" + "\"" + save + "\"" + "\n")
                    
                    # f.write("python mutli_task_fix.py" + " --seed " + "${" + str(seed) + "} " + "--freeze_layer " + "${" + str(freeze_layer) + "} " + "--weight_decay " + "${" + str(weight_decay) + "} " + "--device " + "${" + str(device) + "} " + "--save_path " + "${" + str(save_path) + "}")
                    f.write("python chem/finetune.py " + "\\" + "\n")
                    f.write("--batch_size " + str(batch_size) + " \\" + "\n")
                    f.write("--seed " + str(seed) +" \\" + "\n")
                    f.write("--decay " + str(decay) + " \\" + "\n")
                    f.write("--lr " + str(lr) + " \\" + "\n")                
                    f.write("--device " + str(device) + " \\" + "\n")
                    f.write("--input_model_file " + "\"" + input_model_file + "\"" + " \\" + "\n")
                    f.write("--save " + "\"" + save + "\"")

                os.chmod(os.path.join(script_file, 'bz'+'_'+str(batch_size)+'_'+'decay'+'_'+str(decay)+'_'+'lr'+'_'+str(lr)+'_'+'seed'+'_'+str(seed)+'_'+'script.sh'), 0o755)
                processes_file.append(os.path.join(script_file, 'bz'+'_'+str(batch_size)+'_'+'decay'+'_'+str(decay)+'_'+'lr'+'_'+str(lr)+'_'+'seed'+'_'+str(seed)+'_'+'script.sh'))

for file in processes_file:
    subprocess.Popen(['sh', file])


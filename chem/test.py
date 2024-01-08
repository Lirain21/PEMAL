'''
Author: Lirain21 17860633515@163.com
Date: 2023-10-04 16:52:34
LastEditors: Lirain21 17860633515@163.com
LastEditTime: 2023-10-06 06:49:21
FilePath: /Copy_GraphMAE_AUC/chem/test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas 

# cl_train = pandas.read_csv('dataset/cl_train/raw/cl_train.csv')
# cl_valid = pandas.read_csv('dataset/cl_valid/raw/cl_valid.csv')

# vdss_train = pandas.read_csv('dataset/vdss_train/raw/vdss_train.csv')
# vdss_valid = pandas.read_csv('dataset/vdss_valid/raw/vdss_valid.csv')

# t1_2_train = pandas.read_csv('dataset/t1_2_train/raw/t1_2_train.csv')
# t1_2_valid = pandas.read_csv('dataset/t1_2_valid/raw/t1_2_valid.csv')

# cl_train_value = cl_train['values']
# cl_valid_value = cl_valid['values'] 

# vdss_train_value = vdss_train['values']
# vdss_valid_value = vdss_valid['values'] 

# t1_2_train_value = t1_2_train['values']
# t1_2_valid_value = t1_2_valid['values'] 


# cl_value = cl_valid_value + cl_train_value

# t1_2_value = t1_2_valid_value + t1_2_train_value

# vdss_value = vdss_valid_value + vdss_train_value 

# print(vdss_value.max())
# print(vdss_value.min())

auc_260 = pandas.read_csv('dataset/auc_260.csv')
auc_2600 = pandas.read_csv('dataset/fuse_common_data_function2_1_after.csv')

auc_260_mol_id = list(set(list(auc_260.mol_ids)[160:]))
auc_2600_mol_id = list(auc_2600.mol_ids)[2109:] 

print('2600', len(auc_2600_mol_id))

# print('260_dis_dup', len(set(auc_260_mol_id)))
print('260', len(auc_260_mol_id))



count = 0
for id in auc_260_mol_id:
    if id in auc_2600_mol_id:
        count += 1
    else:
        print(id)
        

print(count)

        

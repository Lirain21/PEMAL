'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-12-09 13:12:17
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-01-08 10:08:40
FilePath: /Copy_GraphMAE_AUC/chem/finetune_reg.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import argparse
import copy

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN_graphpred
from sklearn.metrics import roc_auc_score, accuracy_score

from splitters import scaffold_split
import pandas as pd

import os
import shutil

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import pandas 
import math 

import matplotlib.pyplot as plt 

from model import MLPregression 
from sklearn.manifold import TSNE 

def plot_t_sne(data, label):
    
    print('data_shape',data.shape)
    
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    
    
    result = tsne.fit_transform(data)
    print('result:', result.shape)
    
    x_data = result[:, 0]
    y_data = result[:, 1]
    
    plt.scatter(x_data, y_data, c=label, s=label, marker='*')
    plt.savefig('physical_cl_t_sne.png')

    

CL_MAX = torch.tensor(math.log(124000))
CL_MIN = torch.tensor(math.log(0.05))

T1_2_MAX = torch.tensor(math.log(40.0))
T1_2_MIN = torch.tensor(math.log(0.05))

VDSS_MAX = torch.tensor(math.log(62.0))
VDSS_MIN = torch.tensor(math.log(0.01832)) 

AUC_MAX = torch.tensor(math.log(14787.0)) 
AUC_MIN = torch.tensor(math.log(7.95)) 

K_MEAN = 909.4
# criterion = nn.MSELoss()
criterion = nn.L1Loss()


def cutoff(k):
    max_value = 503.13
    min_value = 1315.3 
    
    if k < min_value:
        k = min_value
    if k > max_value:
        k = max_value
    
    return k 

def train(args, model, device, loader, optimizer, epoch): # para_contrastive_loss,
    model.train()

    total_loss = 0 
    count = 0 
    
    train_auc_pred_all = []
    train_auc_label_all = []
  
    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
      
        count += 1 
        pred_log = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True)
        y = batch.y.reshape(batch.y.size(0), 1)
       
        loss = criterion(pred_log, y)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_auc_pred_all.append(pred_log)
        train_auc_label_all.append(y)  
    
    train_auc_pred_all = torch.cat(train_auc_pred_all, dim=0).detach().cpu().numpy()
    train_auc_label_all = torch.cat(train_auc_label_all, dim=0).detach().cpu().numpy()
   
    return (total_loss)/count

   
def eval(args, model, device, loader, epoch): # para_contrastive_loss,
    model.eval()
  
    y_scores = []
    y_list = []
  
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred_log, representation = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, False)
          
            y = batch.y.reshape(batch.y.size(0), 1)
            y_scores.append(pred_log)
            y_list.append(y)
         
            
    y_scores = torch.cat(y_scores, dim=0)
    y_list = torch.cat(y_list, dim=0)
    loss = criterion(y_scores, y_list).cpu().detach().item()

    return loss
  

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=3,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32, ## 32: 1.009 # 64: 
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=60, # 1500
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001, # 0.0001
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=1e-9,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.0,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = '4', help='root directory of dataset. For now, only classification.')
    
    
    ## loading the pretrained model 
    parser.add_argument('--input_model_file', type=str, default='checkpoint/motifs_0.5_contras_0.5/_1.pth', help='filename to read the model (if there is any)')
    parser.add_argument('--seed', type=int, default=4, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=1, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default=0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
    parser.add_argument('--scheduler', action="store_true", default=False)
    parser.add_argument('--experiment_name', type=str, default="graphmae")
    
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--delta_mse_loss', type=float, default=0.1)
    
    ## add the multi-task alpha 
    parser.add_argument('--t1_2_alpha', type=float, default=1.)
    parser.add_argument('--k3_alphla', type=float, default=1.)
    
    ## add some argument 
    parser.add_argument('--dataset_type', type=int, default=1)
    parser.add_argument('--save', type=str, default='')
    args = parser.parse_args()


    args.use_early_stopping = args.dataset in ("muv", "hiv")
    args.scheduler = args.dataset in ("bace")
    
    all_best_val_mean_loss = []
    
    best_epoch = 0
    for args.runseed in [1, 2, 3, 4, 5]:
        torch.manual_seed(args.runseed)
        np.random.seed(args.runseed)
        # args.seed = args.runseed 
        args.experiment_name = 'lr'+'_'+str(args.lr)+'_'+'decay'+'_'+str(args.decay)+'_'+'bz'+'_'+str(args.batch_size)+'_'+'seed'+'_'+str(args.seed)

        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.runseed)

        ## dataset 
        if args.dataset == "cl":
            num_tasks = 1
            train_dataset_name = "cl_train"
            valid_dataset_name = "cl_valid"
            test_dataset_name = "cl_test"
        elif args.dataset == "vdss":
            num_tasks = 1
            train_dataset_name = "vdss_train"
            valid_dataset_name = "vdss_valid"
            test_dataset_name = "vdss_test"
        elif args.dataset == "t1_2":
            num_tasks = 1
            train_dataset_name = "t1_2_train"
            valid_dataset_name = "t1_2_valid"
            test_dataset_name = "t1_2_test"
        else:
            raise ValueError("Invalid dataset name.")

        ## set up pk dataset 
        train_dataset = MoleculeDataset("dataset_reg/"+train_dataset_name, dataset=train_dataset_name)
        valid_dataset = MoleculeDataset("dataset_reg/"+valid_dataset_name, dataset=valid_dataset_name)
        
        print(train_dataset)
        print(valid_dataset)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        
        #set up model
        model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, classnum=args.dataset_type)
      
        if not args.input_model_file == "":
            print("load pretrained model from:", args.input_model_file)
            model.from_pretrained(args.input_model_file, device=device)
        model.to(device)
      
        #set up optimizer
        #different learning rate for different part of GNN
        model_param_group = []
        model_param_group.append({"params": model.gnn.parameters()})
        
       
        if args.graph_pooling == "attention":
            model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
        model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
        print(optimizer)

        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=45)
        else:
            scheduler = None

        epoch_list = np.arange(0, args.epochs, 1)
        print('epoch_list_len',len(epoch_list))
    
        best_val_mse_loss=float('inf')
        for epoch in range(1, args.epochs+1):
            print("====epoch " + str(epoch))
            
            train_mae_loss = train(args, model, device, train_loader, optimizer, epoch) # para_contrastive_loss,
            val_mae_loss = eval(args, model, device, val_loader, epoch) # para_contrastive_loss,
            
            if scheduler is not None:
                scheduler.step(metrics=val_mae_loss)
                
            if best_val_mse_loss > val_mae_loss:
                best_val_mse_loss = val_mae_loss
                best_epoch = epoch 
             
            print('best epoch:', best_epoch)
            print('best_val_mse_loss', best_val_mse_loss)
            print("train: %f val: %f " %(train_mae_loss, best_val_mse_loss))
      
        all_best_val_mean_loss.append(best_val_mse_loss)

    mean_val_mse_loss = np.mean(np.array(all_best_val_mean_loss))
 
    dataframe = pandas.DataFrame({'val_mse_loss':[mean_val_mse_loss]})
    dataframe.to_csv("results/freeze3/"+args.experiment_name+"_"+"result.csv", index=False)
    
    
if __name__ == "__main__":
    main()

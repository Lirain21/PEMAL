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

from model import GNN, GNN_graphpred
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

def train(args, vdss_model, t1_2_model, cl_model,  para_loss, para_function_cl_vdss_t1_2, para_function_auc_cl, para_function_auc_vdss_t1_2, device, loader, optimizer, epoch): # para_contrastive_loss,
    vdss_model.train()
    t1_2_model.train()
    cl_model.train()
   
    total_loss = 0 
    count = 0 
    
    train_auc_pred_all = []
    train_auc_label_all = []
    
    train_cl_pred_all = []
    train_cl_label_all = []
    
    train_vdss_pred_all = []
    train_vdss_label_all = []
    
    train_t1_2_pred_all = []
    train_t1_2_label_all = []
    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
      
        count += 1 
        vdss_pred_log = vdss_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True)
        t1_2_pred_log = t1_2_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True)
        cl_pred_log = cl_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True)
       
        vdss_pred = torch.exp(vdss_pred_log)
        t1_2_pred = torch.exp(t1_2_pred_log)
        cl_pred = torch.exp(cl_pred_log)
    
        # auc_pred_1 = torch.log(para_function_auc_cl*16000.0/cl_pred)
        ## TODO: leverage the mean +- 3 * std to avaluate the k 
        # k = para_function_auc_vdss_t1_2 * K_MEAN
        # k = cutoff(k)
        
        k_1 = para_function_auc_cl * 16846.39 
        k_2 = para_function_auc_vdss_t1_2 * 1025.28
        k_3 = para_function_cl_vdss_t1_2 * 17.71
        
        auc_pred = k_1/cl_pred 
        auc_pred_log = torch.log(auc_pred)
        
        auc_y = batch.auc_y.reshape(batch.auc_y.size(0), 1)
        cl_y = batch.cl_y.reshape(batch.cl_y.size(0), 1)
        vdss_y = batch.vdss_y.reshape(batch.vdss_y.size(0), 1)
        t1_2_y = batch.t1_2_y.reshape(batch.t1_2_y.size(0), 1)
        
        loss_auc = criterion(auc_pred_log, auc_y)
        loss_cl = criterion(cl_pred_log, cl_y)
        loss_vdss = criterion(vdss_pred_log, vdss_y)
        loss_t1_2 = criterion(t1_2_pred_log, t1_2_y)
        
        loss_k3 = criterion(torch.log(k_3 * vdss_pred), torch.log(t1_2_pred * cl_pred))
        loss_k2 = criterion(torch.log(auc_pred * vdss_pred), torch.log(k_2 * t1_2_pred))
        loss = loss_auc + loss_cl + loss_vdss + loss_t1_2  + loss_k2  + loss_k3
       
        total_loss += loss.item()
        # train_pred_1.append(auc_pred_1)
        # train_label.append(y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_auc_pred_all.append(auc_pred_log)
        train_cl_pred_all.append(cl_pred_log)
        train_vdss_pred_all.append(vdss_pred_log)
        train_t1_2_pred_all.append(t1_2_pred_log)
        
        train_auc_label_all.append(auc_y)
        train_cl_label_all.append(cl_y)
        train_vdss_label_all.append(vdss_y)
        train_t1_2_label_all.append(t1_2_y)
        
        
  
    # if epoch == 8:
    #     train_pred_1 = torch.cat(train_pred_1, dim=0).detach().cpu().numpy()
    #     # train_pred_2 = torch.cat(train_pred_2, dim=0).detach().cpu().numpy()
    #     train_label = torch.cat(train_label, dim=0).detach().cpu().numpy()
    #     # train_label = train_label.cpu().detach().numpy()
    #     # train_pred_1 = train_pred_1.cpu().detach().numpy()
        
    #     plt.plot(train_label, train_label)
    #     plt.scatter(train_label, train_pred_1)
    #     plt.savefig('imgs/physical_auc_scatter.png')
    
    train_auc_pred_all = torch.cat(train_auc_pred_all, dim=0).detach().cpu().numpy()
    train_cl_pred_all = torch.cat(train_cl_pred_all, dim=0).detach().cpu().numpy()
    train_vdss_pred_all = torch.cat(train_vdss_pred_all, dim=0).detach().cpu().numpy()
    train_t1_2_pred_all = torch.cat(train_t1_2_pred_all, dim=0).detach().cpu().numpy()
    
    train_auc_label_all = torch.cat(train_auc_label_all, dim=0).detach().cpu().numpy()
    train_cl_label_all = torch.cat(train_cl_label_all, dim=0).detach().cpu().numpy()
    train_vdss_label_all = torch.cat(train_vdss_label_all, dim=0).detach().cpu().numpy()
    train_t1_2_label_all = torch.cat(train_t1_2_label_all, dim=0).detach().cpu().numpy()
    
    
    return (total_loss)/count, train_auc_pred_all, train_cl_pred_all, train_vdss_pred_all, train_t1_2_pred_all,\
            train_auc_label_all, train_cl_label_all, train_vdss_label_all, train_t1_2_label_all



def eval(args, vdss_model, t1_2_model, cl_model, para_loss, para_function_cl_vdss_t1_2, para_function_auc_cl, para_function_auc_vdss_t1_2,  device, loader, epoch): # para_contrastive_loss,
    vdss_model.eval()
    t1_2_model.eval()
    cl_model.eval()
    
    y_scores_cl = []
    y_scores_vdss = []
    y_scores_t1_2 = []
    y_scores_auc = []
    
    y_cl = []
    y_vdss = []
    y_t1_2 = []
    y_auc = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            vdss_pred_log = vdss_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True)
            t1_2_pred_log = t1_2_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True)
            cl_pred_log = cl_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True)
           
            # y_true_value.append(batch.y.reshape(batch.y.size(0), 1))
        
            vdss_pred = torch.exp(vdss_pred_log)
            t1_2_pred = torch.exp(t1_2_pred_log)
            cl_pred = torch.exp(cl_pred_log)
                
            # k = para_function_cl_vdss_t1_2 * K_MEAN
            # k = cutoff(k)
            # auc_pred_1 = torch.log(k * t1_2_pred / vdss_pred)
            # y_scores_1.append(auc_pred_1)
            
            k_1 = para_function_auc_cl * 16846.39 
            k_2 = para_function_auc_vdss_t1_2 * 1025.28
            k_3 = para_function_cl_vdss_t1_2 * 17.71
            auc_pred_log = torch.log(k_1/cl_pred)
            
            auc_y = batch.auc_y.reshape(batch.auc_y.size(0), 1)
            cl_y = batch.cl_y.reshape(batch.cl_y.size(0), 1)
            vdss_y = batch.vdss_y.reshape(batch.vdss_y.size(0), 1)
            t1_2_y = batch.t1_2_y.reshape(batch.t1_2_y.size(0), 1)
            
            y_scores_auc.append(auc_pred_log)
            y_scores_cl.append(cl_pred_log)
            y_scores_vdss.append(vdss_pred_log)
            y_scores_t1_2.append(t1_2_pred_log)
            
            y_auc.append(auc_y)
            y_cl.append(cl_y)
            y_vdss.append(vdss_y)
            y_t1_2.append(t1_2_y)
            
    y_scores_auc = torch.cat(y_scores_auc, dim=0)
    y_scores_cl = torch.cat(y_scores_cl, dim=0)
    y_scores_vdss = torch.cat(y_scores_vdss, dim=0)
    y_scores_t1_2 = torch.cat(y_scores_t1_2, dim=0)
    
    y_auc = torch.cat(y_auc, dim=0)
    y_cl = torch.cat(y_cl, dim=0)
    y_vdss = torch.cat(y_vdss, dim=0)
    y_t1_2 = torch.cat(y_t1_2, dim=0)
    
    loss_auc = criterion(y_scores_auc, y_auc).cpu().detach().item()
    loss_cl = criterion(y_scores_cl, y_cl).cpu().detach().item()
    loss_vdss = criterion(y_scores_vdss, y_vdss).cpu().detach().item()
    loss_t1_2 = criterion(y_scores_t1_2, y_t1_2).cpu().detach().item()
 
    # if epoch == 8:
    #     y_scores_1 = y_scores_1.cpu().detach().numpy()
    #     y_true_value = y_true_value.cpu().detach().numpy()
        
    #     plt.plot(y_true_value, y_true_value)
    #     plt.scatter(y_true_value, y_scores_1)
    #     plt.savefig('imgs/physical_auc_scatter.png')
    
    loss_all = (loss_auc+loss_cl+loss_vdss+loss_t1_2)
   
    return loss_auc, loss_cl, loss_vdss, loss_t1_2, loss_all,\
            y_scores_auc.cpu().detach().numpy(), y_scores_cl.cpu().detach().numpy(), y_scores_vdss.cpu().detach().numpy(), y_scores_t1_2.cpu().detach().numpy(),\
            y_auc.cpu().detach().numpy(), y_cl.cpu().detach().numpy(), y_vdss.cpu().detach().numpy(), y_t1_2.cpu().detach().numpy()
            
  

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32, ## 32: 1.009 # 64: 
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=60, # 1500
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001,
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
    
    ## load three datasets' pretrain model 
    parser.add_argument('--input_vdss_model_file', type=str, default = 'results/11_13/finetune/vdss/graphmae_lr_0.0005_decay_1e-10_bz_64_seed_4_5model.pt', help='filename to read the model (if there is any)')
    parser.add_argument('--input_t1_2_model_file', type=str, default = 'results/11_13/finetune/t1_2/graphmae_lr_0.0001_decay_1e-10_bz_128_seed_4_1model.pt', help='filename to read the model (if there is any)')
    parser.add_argument('--input_cl_model_file', type=str, default = 'results/11_13/finetune/cl/graphmae_lr_0.0001_decay_1e-10_bz_64_seed_4_1model.pt', help='filename to read the model (if there is any)')

    
    parser.add_argument('--filename', type=str, default = 'checkpoints/gin_100.pth', help='output filename')
    parser.add_argument('--seed', type=int, default=4, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=1, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default=0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
    parser.add_argument('--scheduler', action="store_true", default=False)
    parser.add_argument('--experiment_name', type=str, default="graphmae")
    
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--delta_mse_loss', type=float, default=0.1)
    
    ## add some argument 
    parser.add_argument('--dataset_type', type=int, default=1)
    parser.add_argument('--save', type=str, default='')
    args = parser.parse_args()

    args.use_early_stopping = args.dataset in ("muv", "hiv")
    args.scheduler = args.dataset in ("bace")
    
    para_loss = nn.Parameter(torch.tensor(0.5, device=args.device), requires_grad=True)
    para_function_cl_vdss_t1_2 = nn.Parameter(torch.tensor(1.0, device=args.device), requires_grad=True)
    para_function_auc_cl = nn.Parameter(torch.tensor(1.0, device=args.device), requires_grad=True)
    para_function_auc_vdss_t1_2 = nn.Parameter(torch.tensor(1.0, device=args.device), requires_grad=True)
    
    para_loss.data.fill_(0.5)
    para_loss.data.fill_(1.)
    para_loss.data.fill_(1.)
    para_loss.data.fill_(1.)
 
 
    print('para_loss',para_loss)
    print('para_function_cl_vdss_t1_2',para_function_cl_vdss_t1_2)
    print('para_function_auc_cl', para_function_auc_cl)
    print('para_function_auc_vdss_t1_2', para_function_auc_vdss_t1_2)
    
 
    all_best_val_mean_loss = []
    all_best_auc_mean_loss = []
    all_best_cl_mean_loss = []
    all_best_t1_2_mean_loss = []
    all_best_vdss_mean_loss = []
    
    
    best_epoch = 0
    for args.runseed in [1, 2, 3, 4, 5]:
        torch.manual_seed(args.runseed)
        np.random.seed(args.runseed)
        # args.seed = args.runseed 
        args.experiment_name = 'lr'+'_'+str(args.lr)+'_'+'decay'+'_'+str(args.decay)+'_'+'bz'+'_'+str(args.batch_size)+'_'+'seed'+'_'+str(args.seed)

        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.runseed)

        #Bunch of classification tasks
        if args.dataset == "tox21":
            num_tasks = 12
        elif args.dataset == "auc":
            num_tasks = 1
            train_dataset_name = "auc_train"
            valid_dataset_name = "auc_valid"
            test_dataset_name = "auc_test"
        elif args.dataset == "cl":
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
            
        elif args.dataset == "4":
            num_tasks = 1
            train_dataset_name = "4_train"
            valid_dataset_name = "4_valid"
            test_dataset_name = "4_test"
        else:
            raise ValueError("Invalid dataset name.")

        ## set up pk dataset 
        train_dataset = MoleculeDataset("dataset_reg/"+train_dataset_name, dataset=train_dataset_name)
        valid_dataset = MoleculeDataset("dataset_reg/"+valid_dataset_name, dataset=valid_dataset_name)
        # test_dataset = MoleculeDataset("dataset/"+test_dataset_name, dataset=test_dataset_name)
        
        print(train_dataset)
        print(valid_dataset)
        # print(test_dataset)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        #set up model
        vdss_model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, classnum=args.dataset_type)
        t1_2_model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, classnum=args.dataset_type)
        cl_model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, classnum=args.dataset_type)


        if not args.input_vdss_model_file == "":
            print("load pretrained model from:", args.input_vdss_model_file)
            vdss_model.load_state_dict(torch.load(args.input_vdss_model_file, map_location=device))
        
        if not args.input_t1_2_model_file == "":
            print("load pretrained model from:", args.input_t1_2_model_file)
            t1_2_model.load_state_dict(torch.load(args.input_t1_2_model_file, map_location=device))
            
        if not args.input_cl_model_file == "":
            print("load pretrained model from:", args.input_cl_model_file)
            cl_model.load_state_dict(torch.load(args.input_cl_model_file, map_location=device))
            
        vdss_model.to(device)
        t1_2_model.to(device)
        cl_model.to(device)

        #set up optimizer
        #different learning rate for different part of GNN
        model_param_group = []
        model_param_group.append({"params": vdss_model.gnn.parameters()})
        model_param_group.append({'params': t1_2_model.gnn.parameters()})
        model_param_group.append({'params': cl_model.gnn.parameters()})

        model_param_group.append({"params": para_loss})
        model_param_group.append({"params": para_function_cl_vdss_t1_2})
        model_param_group.append({'params': para_function_auc_cl})
        model_param_group.append({'params': para_function_auc_vdss_t1_2})

        if args.graph_pooling == "attention":
            model_param_group.append({"params": vdss_model.pool.parameters(), "lr":args.lr*args.lr_scale})
            model_param_group.append({'params': t1_2_model.pool.parameters(), "lr":args.lr*args.lr_scale})
            model_param_group.append({'params': cl_model.pool.parameters(), 'lr':args.lr*args.lr_scale})
        model_param_group.append({"params": vdss_model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
        model_param_group.append({'params': t1_2_model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
        model_param_group.append({'params': cl_model.graph_pred_linear.parameters(), 'lr':args.lr*args.lr_scale })
        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
        print(optimizer)

        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)
        else:
            scheduler = None

        epoch_list = np.arange(0, args.epochs, 1)
        print('epoch_list_len',len(epoch_list))
        train_mse_loss_list = []
        val_mse_loss_list = []
        
        best_train_auc_pred = []
        best_train_cl_pred= []
        best_train_vdss_pred = []
        best_train_t1_2_pred = []
        
        best_train_auc_label = []
        best_train_cl_label = []
        best_train_vdss_label = []
        best_train_t1_2_label = []
        
        best_valid_auc_pred = []
        best_valid_cl_pred= []
        best_valid_vdss_pred = []
        best_valid_t1_2_pred = []
        
        best_valid_auc_label = []
        best_valid_cl_label = []
        best_valid_vdss_label = []
        best_valid_t1_2_label = []
        

        best_val_mse_loss=float('inf')
        best_true_val_mse_loss = float('inf')
        times = 0
        for epoch in range(1, args.epochs+1):
            print("====epoch " + str(epoch))
            
            train_mes_loss, train_auc_pred, train_cl_pred, train_vdss_pred, train_t1_2_pred,\
            train_auc_label, train_cl_label, train_vdss_label, train_t1_2_label = train(args, vdss_model, t1_2_model, cl_model, para_loss, para_function_cl_vdss_t1_2, para_function_auc_cl, para_function_auc_vdss_t1_2, device, train_loader, optimizer, epoch) # para_contrastive_loss,
            train_mse_loss_list.append(train_mes_loss)
            
            if scheduler is not None:
                scheduler.step()

            auc_val_loss, cl_val_loss, vdss_val_loss, t1_2_val_loss, all_loss, \
                valid_auc_pred, valid_cl_pred, valid_vdss_pred, valid_t1_2_pred, \
                valid_auc_label, valid_cl_label, valid_vdss_label, valid_t1_2_label = eval(args, vdss_model, t1_2_model, cl_model, para_loss, para_function_cl_vdss_t1_2, para_function_auc_cl, para_function_auc_vdss_t1_2, device, val_loader, epoch) # para_contrastive_loss,
            if best_val_mse_loss > all_loss:
                best_val_mse_loss = all_loss
                best_epoch = epoch 
                best_auc_loss = auc_val_loss
                best_cl_loss = cl_val_loss 
                best_vdss_loss = vdss_val_loss
                best_t1_2_loss = t1_2_val_loss
                
                best_train_auc_pred = train_auc_pred 
                best_train_cl_pred = train_cl_pred 
                best_train_t1_2_pred = train_t1_2_pred 
                best_train_vdss_pred = train_vdss_pred 
                
                best_train_auc_label = train_auc_label 
                best_train_cl_label = train_cl_label 
                best_train_vdss_label = train_vdss_label 
                best_train_t1_2_label = train_t1_2_label 
                
                best_valid_auc_pred = valid_auc_pred 
                best_valid_cl_pred = valid_cl_pred 
                best_valid_t1_2_pred = valid_t1_2_pred 
                best_valid_vdss_pred = valid_vdss_pred 
                
                best_valid_auc_label = valid_auc_label 
                best_valid_cl_label = valid_cl_label
                best_valid_vdss_label = valid_vdss_label 
                best_valid_t1_2_label = valid_t1_2_label 
       
                # torch.save(cl_model.state_dict(), "results/auc_models/cl/"+args.experiment_name+"_"+"cl_model.pt")
                # torch.save(t1_2_model.state_dict(), "results/auc_models/t1_2/"+args.experiment_name+"_"+"t1_2_model.pt")
                # torch.save(vdss_model.state_dict(), "results/auc_models/vdss/"+args.experiment_name+"_"+"vdss_model.pt")
                
            print('best epoch:', best_epoch)
            print('best_val_mse_loss', best_val_mse_loss)
            print('best_auc_loss:', best_auc_loss)
            print('best_cl_loss:', best_cl_loss)
            print('best_vdss_loss:', best_vdss_loss)
            print('best_t1_2_loss:', best_t1_2_loss)
            print('at the meanwhile, the true valid mse loss', best_true_val_mse_loss)
            print("train: %f val: %f " %(train_mes_loss, best_val_mse_loss))
            val_mse_loss_list.append(all_loss)
            
            # cl_val_mse_loss_list.append(cl_val_mse_loss)
            # t1_2_val_mse_loss_list.append(t1_2_val_mse_loss)
            # vdss_val_mse_loss_list.append(vdss_val_mse_loss)
            
            # test_acc_list.append(test_acc)
            # train_mse_loss_list.append(train_mse_loss)
            
            
            # dataframe_1 = pandas.DataFrame({'train_mse_loss':train_mse_loss_list,'auc_valid_mse_loss':auc_val_mse_loss_list,'cl_valid_mse_loss':cl_val_mse_loss,'t1_2_valid_mse_loss':cl_val_mse_loss_list,'t1_2_valid_mse_loss':t1_2_val_mse_loss_list, 'vdss_volid_mse_loss':vdss_val_mse_loss_list})
            # dataframe_1 = pandas.DataFrame({'train_valid_loss':train_mse_loss_list, 'auc_valid_mse_loss':val_mse_loss_list,})
            # dataframe_1.to_csv("results/physical_equation_result/11_6/auc/mae/"+args.experiment_name+"_"+str(args.runseed)+"_"+"auc_cl.csv", index=False)
            
            

        all_best_val_mean_loss.append(best_val_mse_loss)
        all_best_auc_mean_loss.append(best_auc_loss)
        all_best_cl_mean_loss.append(best_cl_loss)
        all_best_vdss_mean_loss.append(best_vdss_loss)
        all_best_t1_2_mean_loss.append(best_t1_2_loss)
        
        # plt.plot(epoch_list, train_mse_loss_list)
        # plt.plot(epoch_list, val_mse_loss_list)
        # plt.savefig('imgs/physical_loss_curve_4_tasks'+str(args.runseed)+'.png')
        
        # plt.plot(best_train_auc_label, best_train_auc_label)
        # plt.scatter(best_train_auc_label, best_train_auc_pred)
        # plt.scatter(best_valid_auc_label, best_valid_auc_pred)
        # plt.savefig('imgs/physical_scatter_auc'+str(args.runseed)+'.png')
        
        # plt.plot(best_train_cl_label, best_train_cl_label)
        # plt.scatter(best_train_cl_label, best_train_cl_pred)
        # plt.scatter(best_valid_cl_label, best_valid_cl_pred)
        # plt.savefig('imgs/physical_scatter_cl'+str(args.runseed)+'.png')
        
        # plt.plot(best_train_vdss_label, best_train_vdss_label)
        # plt.scatter(best_train_vdss_label, best_train_vdss_pred)
        # plt.scatter(best_valid_vdss_label, best_valid_vdss_pred)
        # plt.savefig('imgs/physical_scatter_vdss'+str(args.runseed)+'.png')
        
        # plt.plot(best_train_t1_2_label, best_train_t1_2_label)
        # plt.scatter(best_train_t1_2_label, best_train_t1_2_pred)
        # plt.scatter(best_valid_t1_2_label, best_valid_t1_2_pred)
        # plt.savefig('imgs/physical_scatter_t1_2'+str(args.runseed)+'.png')
            
        
        # plt.plot(epoch_list, train_mse_loss_list)
        # plt.plot(epoch_list, val_mse_loss_list)
        # # plt.x_label('epoch')
        # # plt.y_label('mae')
        # plt.savefig('imgs/physical_loss_curve_'+str(args.runseed)+'.png')
        
        # exit()
        
  
    ## draw the loss curve 
    # x_values = np.arange(epoch)
    # y_values = np.array(train_mse_loss_list)
    
    # print(x_values.shape)
    # print(y_values.shape)
    
    # plt.bar(x_values, y_values)
    # plt.title('The log_train_mse_loss_lr_0.0005')
    # plt.savefig('imgs/log_train_mse_loss_lr_0.0005.jpg')
    # plt.show()
    
    # x_values = np.arange(epoch)
    # y_values = np.array(auc_val_mse_loss_list)
    
    # plt.bar(x_values, y_values)
    # plt.title('The log_valid_mse_loss_lr_0.00005')
    # plt.savefig('imgs/log_valid_mse_loss_lr_0.00005.jpg')
    # plt.show()
        
         
    # cl_model.load_state_dict(torch.load("results/auc_models/cl/"+args.experiment_name+"_"+"model.pt", map_location=device))
    # t1_2_model.load_state_dict(torch.load("results/auc_models/t1_2/"+args.experiment_name+"_"+"model.pt", map_location=device))
    # vdss_model.load_state_dict(torch.load("results/auc_models/vdss/"+args.experiment_name+"_"+"model.pt", map_location=device))
  
    # test_mse_loss = eval(args, cl_model, t1_2_model, vdss_model, para_loss, para_function_auc_cl, para_function_auc_vdss_t1_2, para_contrastive_loss, device, test_loader)


    mean_val_mse_loss = np.mean(np.array(all_best_val_mean_loss))
    mean_auc_loss = np.mean(np.array(all_best_auc_mean_loss))
    mean_cl_loss = np.mean(np.array(all_best_cl_mean_loss))
    mean_vdss_loss = np.mean(np.array(all_best_vdss_mean_loss))
    mean_t1_2_loss = np.mean(np.array(all_best_t1_2_mean_loss))
    
    dataframe = pandas.DataFrame({'val_mse_loss':[mean_val_mse_loss], 'auc_loss':[mean_auc_loss], 'cl_loss':[mean_cl_loss], 'vdss_loss':[mean_vdss_loss], 't1_2_loss':[mean_t1_2_loss]})
    dataframe.to_csv("results/11_13/k2_k3/"+args.experiment_name+"_"+"result.csv", index=False)
    
    

if __name__ == "__main__":
    main()

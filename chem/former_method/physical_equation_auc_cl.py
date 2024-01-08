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

K_MEAN = 16796.0
# criterion = nn.MSELoss()
criterion = nn.L1Loss()


def cutoff(k):
    
    ## 3 * std 
    # max_value = 20019.8
    # min_value = 13572.2
    
    ## 1 * std 
    max_value = 15721.4
    min_value = 17870.6
    
    
    if k < min_value:
        k = min_value
    if k > max_value:
        k = max_value
    
    return k 

def train(args, cl_model, para_loss, para_function_auc_cl,  device, loader, optimizer, epoch): # para_contrastive_loss,
    cl_model.train()

    total_loss = 0 
    count = 0 
    
    train_pred_1 = []
    train_pred_2 = []
    train_label = []
    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
      
        count += 1 
        cl_pred_log = cl_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True)
        cl_pred = torch.exp(cl_pred_log)
    
        auc_pred_log = torch.log(para_function_auc_cl*16000.0/cl_pred)
        
        ## TODO: leverage the mean +- 3 * std to avaluate the k 
        # k = para_function_auc_cl * K_MEAN
        # k = cutoff(k)
        # auc_pred_1 = torch.log(k/cl_pred)
        
        auc_y = batch.y.reshape(batch.y.size(0), 1)
        cl_y = batch.cl_y.reshape(batch.cl_y.size(0), 1)
        
        auc_loss = criterion(auc_pred_log, auc_y)
        cl_loss = criterion(cl_pred_log, cl_y)
        
        # loss = (1 - para_loss)*auc_loss + para_loss*cl_loss 
        loss = auc_loss + cl_loss 
  
        total_loss += loss.item()
        # train_pred_1.append(auc_pred_1)
        # train_label.append(y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # if epoch == 8:
    #     train_pred_1 = torch.cat(train_pred_1, dim=0).detach().cpu().numpy()
    #     train_label = torch.cat(train_label, dim=0).detach().cpu().numpy()
    #     plt.plot(train_label, train_label)
    #     plt.scatter(train_label, train_pred_1)
    #     plt.savefig('imgs/physical_auc_scatter.png')
    
    return (total_loss)/count



def eval(args, cl_model, para_loss, para_function_auc_cl,  device, loader, epoch): # para_contrastive_loss,
    cl_model.eval()
    y_scores_auc = []
    y_scores_cl = []
    
    y_true_value_auc = []
    y_true_value_cl = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            cl_pred_log = cl_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True)
         
            y_true_value_auc.append(batch.y.reshape(batch.y.size(0), 1))
            y_true_value_cl.append(batch.cl_y.reshape(batch.cl_y.size(0), 1))
        
            cl_pred = torch.exp(cl_pred_log)
          
            
            auc_pred_log = torch.log(para_function_auc_cl*16000.0/cl_pred)
            # auc_pred_2 = torch.log(para_function_auc_vdss_t1_2*1000*t1_2_pred/vdss_pred)
            
            ## TODO: valid the mean +- 3 * std ‘s result 
            # k = para_function_auc_cl * K_MEAN
            # k = cutoff(k)
            # auc_pred_1 = torch.log(k/cl_pred)
        
            y_scores_auc.append(auc_pred_log)
            y_scores_cl.append(cl_pred_log)
            
    y_scores_auc = torch.cat(y_scores_auc, dim = 0)
    y_scores_cl = torch.cat(y_scores_cl, dim = 0)
    y_true_value_auc = torch.cat(y_true_value_auc, dim = 0)
    y_true_value_cl = torch.cat(y_true_value_cl, dim = 0)
    
    auc_loss = criterion(y_scores_auc, y_true_value_auc).cpu().detach().item()
    cl_loss = criterion(y_scores_cl, y_true_value_cl).cpu().detach().item()
    
    # if epoch == 8:
    #     y_scores_1 = y_scores_1.cpu().detach().numpy()
    #     y_true_value = y_true_value.cpu().detach().numpy()
        
    #     plt.plot(y_true_value, y_true_value)
    #     plt.scatter(y_true_value, y_scores_1)
    #     plt.savefig('imgs/physical_auc_scatter.png')
   
    return auc_loss, cl_loss
  

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128, ## 32: 1.009 # 64: 
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=60, # 1500
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.005,
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
    parser.add_argument('--dataset', type=str, default = 'auc_cl', help='root directory of dataset. For now, only classification.')
    
    ## load three datasets' pretrain model 
    parser.add_argument('--input_cl_model_file', type=str, default = 'results/mae/finetune/cl/graphmae_lr_0.001_decay_1e-09_bz_32_seed_4_4model.pt', help='filename to read the model (if there is any)')
    
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
    
    para_loss = nn.Parameter(torch.tensor(0.5, device=args.device))
    para_function_auc_cl = nn.Parameter(torch.tensor(1.0, device=args.device))
 
    print('para_loss',para_loss)
    print('para_function_auc_cl',para_function_auc_cl)
 
    auc_all_best_val_mean_loss = []
    cl_all_best_val_mean_loss = []
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
        elif args.dataset == 'auc_cl':
            num_tasks = 1
            train_dataset_name = "auc_cl_train"
            valid_dataset_name = "auc_cl_valid"
            test_dataset_name = "auc_cl_test"
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
        cl_model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, classnum=args.dataset_type)

        # if not args.input_cl_model_file == "":
        #     print("load pretrained model from:", args.input_cl_model_file)
        #     cl_model.load_state_dict(torch.load(args.input_cl_model_file, map_location=device))

        cl_model.to(device)

        #set up optimizer
        #different learning rate for different part of GNN
        model_param_group = []
        model_param_group.append({"params": cl_model.gnn.parameters()})

        model_param_group.append({"params": para_loss})
        model_param_group.append({"params": para_function_auc_cl})

        if args.graph_pooling == "attention":
            model_param_group.append({"params": cl_model.pool.parameters(), "lr":args.lr*args.lr_scale})
        model_param_group.append({"params": cl_model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
        print(optimizer)

        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)
        else:
            scheduler = None

        epoch_list = np.arange(0, args.epochs, 1)
        print('epoch_list_len',len(epoch_list))
        train_mse_loss_list = []
 
        best_auc_val_mse_loss=float('inf')
        best_cl_val_mse_loss = float('inf')
        times = 0
        for epoch in range(1, args.epochs+1):
            print("====epoch " + str(epoch))
            
            train_mes_loss = train(args, cl_model, para_loss, para_function_auc_cl, device, train_loader, optimizer, epoch) # para_contrastive_loss,
            # train_mse_loss_list.append(train_mes_loss)
            
            if scheduler is not None:
                scheduler.step()

            print("====Evaluation")
            auc_val_mse_loss, cl_val_mse_loss = eval(args, cl_model, para_loss, para_function_auc_cl,  device, val_loader, epoch) # para_contrastive_loss,
            if best_auc_val_mse_loss > auc_val_mse_loss:
                best_auc_val_mse_loss = auc_val_mse_loss 
                best_epoch = epoch 
            
            if best_cl_val_mse_loss > cl_val_mse_loss:
                best_cl_val_mse_loss = cl_val_mse_loss
                
        
            print('best epoch:', best_epoch)
            print('best_auc_val_mse_loss', best_auc_val_mse_loss)
            print('best cl valid mse loss', best_cl_val_mse_loss)
            print("train: %f val: %f val: %f" %(train_mes_loss, auc_val_mse_loss, cl_val_mse_loss))
          
        auc_all_best_val_mean_loss.append(best_auc_val_mse_loss)
        cl_all_best_val_mean_loss.append(best_cl_val_mse_loss)
          
    auc_mean_val_mse_loss = np.mean(np.array(auc_all_best_val_mean_loss))
    cl_mean_val_mse_loss = np.mean(np.array(cl_all_best_val_mean_loss))
    print(auc_mean_val_mse_loss)
    print(cl_mean_val_mse_loss)
    dataframe = pandas.DataFrame({'auc_val_mse_loss':[auc_mean_val_mse_loss],'cl_val_mse_loss':[cl_mean_val_mse_loss]})
    dataframe.to_csv("results/unpretrain_common_auc_cl_1/"+args.experiment_name+"_"+"result.csv", index=False)
    
    

if __name__ == "__main__":
    main()

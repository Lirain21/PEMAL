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
from model_fingerprintgnn import FingerprintGNNPred

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


# criterion = nn.MSELoss()

criterion = nn.L1Loss()

## define the label 
def calculate_label(cl_log, device):
    auc_tensor = torch.exp(cl_log)
    auc_pred_list = []
    for pre in auc_tensor:
        va = pre[0].item()
        if va < 500.0:
            auc_pred_list.append(0)
        elif 500.0 < va < 2500.0:
            auc_pred_list.append(1)
        elif 2500.0 < va:
            auc_pred_list.append(2)
        
    auc_pred_list = torch.Tensor(auc_pred_list).to(device)
    return auc_pred_list
  
def train(args, cl_model, para_loss, para_function_auc_cl,  device, loader, optimizer, epoch): # para_contrastive_loss,
    cl_model.train()

    total_loss = 0 
    count = 0 
    
    train_pred = []
    train_label = []
    
    train_pred_label = []
    train_true_label = []
    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        # print(batch)
        
    
        count += 1 
        cl_pred_log = cl_model(batch.fingerprints, batch.maccs, True)
        cl_pred = torch.exp(cl_pred_log)
    
    
        ### TODO: there is something wrong: the para_function_auc_cl is not used in this way, 
        # auc_pred_1 = torch.log(para_function_auc_cl*16796.0/cl_pred)
        # auc_pred_2 = torch.log(para_function_auc_vdss_t1_2*1000*t1_2_pred/vdss_pred)
        y = batch.t1_2_y.reshape(batch.t1_2_y.size(0), 1)
        # all_result = torch.cat((auc_pred_1, y), dim=1)
        
        # print(all_result)
   
        loss_1 = criterion(cl_pred_log, y)
  
        ## TODO: set a learnable parameter 
    
        total_loss += loss_1.item()
        train_pred.append(cl_pred_log)
        train_label.append(y)
        
        optimizer.zero_grad()
        loss_1.backward()
        optimizer.step()
      
    return total_loss/count # , acc 



def eval(args, cl_model, para_loss, para_function_auc_cl,  device, loader, epoch): # para_contrastive_loss,
    cl_model.eval()
    y_true = []
    y_scores_1 = []
    y_scores_2 = []
    
    y_true_value = []
    
    valid_pred_label = []
    valid_true_label = []

    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            cl_pred_log = cl_model(batch.fingerprints, batch.maccs, True)
    
            y_label = batch.t1_2_y.reshape(batch.t1_2_y.size(0), 1)
            y_true_value.append(y_label)
        
            # cl_pred = torch.exp(cl_pred_log)
            # auc_pred_1 = torch.log(para_function_auc_cl*16796.0/cl_pred)
           
            # y_scores_1.append(auc_pred_1)
            
            y_scores_1.append(cl_pred_log)
            
    y_scores_1 = torch.cat(y_scores_1, dim = 0)
    y_true_value = torch.cat(y_true_value, dim = 0)
    
    # valid_pred_label = torch.cat(valid_pred_label, dim=0)
    # valid_true_label = torch.cat(valid_true_label, dim=0).detach().cpu().numpy()
    # valid_pred_label = valid_pred_label.detach().cpu().numpy()
    # acc = accuracy_score(valid_true_label, valid_pred_label)
    
    loss_1 = criterion(y_scores_1, y_true_value).cpu().detach().item()
   

    # if epoch == 2000:  
        
    #     y_scores_1 = y_scores_1.detach().cpu().numpy()
    #     y_true_value = y_true_value.detach().cpu().numpy()
        
    #     plt.scatter(y_true_value, y_scores_1, marker='p', c='c')
    #     plt.plot(y_true_value, y_true_value)
    
    #     plt.title('Concat_finanddesc_valid_lr_0.0005_log'+str(epoch))
    #     plt.savefig(str(epoch)+'_concat_finanddesc_valid_lr_0.0005.jpg')
    #     # plt.show()
   
    # return math.sqrt(abs(loss)), acc 
    
    return loss_1 # , acc 
 

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,  # 32: 1.054   # 64: 1.050  # 128:1.048  # 256: 1.049
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=40,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=1e-10,
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
    parser.add_argument('--input_cl_model_file', type=str, default = 'results/atoms_0.75_motifs_0.1/cl/graphmae_lr_0.005_decay_1e-11_bz_64_seed_4_model.pt', help='filename to read the model (if there is any)')
    
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
  
  
    all_best_val_mean_loss = []
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
        elif args.dataset == "auc_cl":
            num_tasks = 1 
            train_dataset_name = "auc_cl_train"
            valid_dataset_name = "auc_cl_valid"
            test_dataset_name = "auc_cl_test"
        elif args.dataset == "4":
            num_tasks = 1
            train_dataset_name = "4_train"
            valid_dataset_name = "4_valid"
        else:
            raise ValueError("Invalid dataset name.")

        ## set up pk dataset 
        train_dataset = MoleculeDataset("dataset_reg/"+train_dataset_name, dataset=train_dataset_name)
        valid_dataset = MoleculeDataset("dataset_reg/"+valid_dataset_name, dataset=valid_dataset_name)
    
        print(train_dataset)
        print(valid_dataset)
        
        # print(test_dataset)c
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        #set up model
        
        cl_model = FingerprintGNNPred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, classnum=args.dataset_type)
        cl_model.to(device)

        #set up optimizer
        #different learning rate for different part of GNN
        model_param_group = []
    
        model_param_group.append({"params": para_loss})
        
        for pa in cl_model.fingerprint_mlp.parameters():
        
            if len(pa.size())==2:
                model_param_group.append({'params': pa, 'lr':args.lr*args.lr_scale}) 
            else:
                model_param_group.append({'params':pa, 'lr':args.lr*args.lr_scale, 'weight_decay':0.})


        for pa in cl_model.maccs_mlp.parameters():
            if len(pa.size()) == 2:
                model_param_group.append({'params': pa, 'lr':args.lr*args.lr_scale}) 
            else:
                model_param_group.append({'params':pa, 'lr':args.lr*args.lr_scale, 'weight_decay':0.})
            
        for pa in cl_model.reg_mlp.parameters():
            if len(pa.size()) == 2:
                model_param_group.append({'params': pa, 'lr':args.lr*args.lr_scale}) 
            else:
                model_param_group.append({'params':pa, 'lr':args.lr*args.lr_scale, 'weight_decay':0.})
       
        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
        
        print(optimizer)

        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)
        else:
            scheduler = None

        train_mse_loss_list = []
        auc_val_mse_loss_list = []
        
        train_acc_list = []
        val_acc_list = []

        best_val_mse_loss=float('inf')
        best_true_val_mse_loss = float('inf')
        times = 0
        for epoch in range(1, args.epochs+1):
            print("====epoch " + str(epoch))
            
            train_mes_loss = train(args, cl_model, para_loss, para_function_auc_cl, device, train_loader, optimizer, epoch) # para_contrastive_loss,
            train_mse_loss_list.append(train_mes_loss)
            # train_acc_list.append(train_acc)
            
            if scheduler is not None:
                scheduler.step()

            print("====Evaluation")
            if args.eval_train:
                train_mse_loss, train_acc = eval(args, cl_model, para_loss, para_function_auc_cl, device, train_loader) # para_contrastive_loss,
            else:
                print("omit the training accuracy computation")
                train_mse_loss = 0
            
            val_mse_loss = eval(args, cl_model, para_loss, para_function_auc_cl,  device, val_loader, epoch) # para_contrastive_loss,
            if best_val_mse_loss > val_mse_loss:
                best_val_mse_loss = val_mse_loss 
                
            print('best_val_mse_loss', best_val_mse_loss)
            print('at the meanwhile, the true valid mse loss', best_true_val_mse_loss)
            # print("train_loss: %f valid_loss: %f " %(train_mes_loss, val_mse_loss))
            # print("train_acc: %f valid_acc: %f " %(train_acc, val_acc))
            auc_val_mse_loss_list.append(val_mse_loss)
            # val_acc_list.append(val_acc)
            
            # dataframe_1 = pandas.DataFrame({'train_mse_loss':train_mse_loss_list,'auc_valid_mse_loss':auc_val_mse_loss_list,'cl_valid_mse_loss':cl_val_mse_loss,'t1_2_valid_mse_loss':cl_val_mse_loss_list,'t1_2_valid_mse_loss':t1_2_val_mse_loss_list, 'vdss_volid_mse_loss':vdss_val_mse_loss_list})
            # dataframe_1 = pandas.DataFrame({'train_valid_loss':train_mse_loss_list, 'auc_valid_mse_loss':auc_val_mse_loss_list, "auc_train_acc":train_acc_list, "auc_val_acc":val_acc_list})
            # dataframe_1.to_csv("results/auc_models/auc/"+args.experiment_name+"_"+str(args.runseed)+"_"+"concat_loss.csv", index=False)
            
            ## TODO : add the mean acc 
            
            all_best_val_mean_loss.append(best_val_mse_loss)
            
    ## draw the loss curve 
    x_values = np.arange(epoch)
    y_values = np.array(train_mse_loss_list)
    
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

    mean_val_mse_loss = np.mean(np.array(all_best_val_mean_loss))
    print(mean_val_mse_loss)
    # dataframe = pandas.DataFrame({'val_mse_loss':[mean_val_mse_loss]})
    # dataframe.to_csv("results/auc_models/auc/"+args.experiment_name+"_"+"result.csv", index=False)
    
    

if __name__ == "__main__":
    main()

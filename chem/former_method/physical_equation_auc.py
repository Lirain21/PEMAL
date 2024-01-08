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


CL_MAX = torch.tensor(math.log(124000))
CL_MIN = torch.tensor(math.log(0.05))

T1_2_MAX = torch.tensor(math.log(40.0))
T1_2_MIN = torch.tensor(math.log(0.05))

VDSS_MAX = torch.tensor(math.log(62.0))
VDSS_MIN = torch.tensor(math.log(0.01832)) 

AUC_MAX = torch.tensor(math.log(14787.0)) 
AUC_MIN = torch.tensor(math.log(7.95)) 

K_1_MEAN = 909.4
K_1_MIN = 503.13
K_1_MAX = 1315.3

K_2_MEAN = 16796.0
K_2_MIN = 13572.2
K_2_MAX = 20019.8 

# criterion = nn.MSELoss()
criterion = nn.L1Loss()

def cutoff_k1(k1):
    if k1 < K_1_MIN:
        k1 = K_1_MIN
    elif k1 > K_1_MAX:
        k1 = K_1_MAX
    
    return k1 

def cutoff_k2(k2):
    if k2 < K_2_MIN:
        k2 = K_2_MIN
    elif k2 > K_2_MAX:
        k2 = K_2_MAX
    
    return k2 

def train(args, cl_model, t1_2_model, vdss_model, auc_model, para_loss, para_function_auc_cl, para_function_auc_vdss_t1_2,  device, loader, optimizer, epoch): # para_contrastive_loss,
    cl_model.train()
    t1_2_model.train()
    vdss_model.train()
    auc_model.train()

    total_loss_1 = 0
    total_loss_2 = 0
    total_loss_3 = 0
     
    count = 0 
    
    train_pred_1 = []
    train_pred_2 = []
    train_label = []
    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        count += 1 
        cl_pred_log = cl_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True)
        t1_2_pred_log = t1_2_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True)
        vdss_pred_log = vdss_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True)
        auc_pred_log = auc_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True)
     
        cl_pred = torch.exp(cl_pred_log)
        t1_2_pred = torch.exp(t1_2_pred_log)
        vdss_pred = torch.exp(vdss_pred_log)
        auc_pred = torch.exp(auc_pred_log)
        
        
        # auc_pred_1 = torch.log(para_function_auc_cl*16000.0/cl_pred)
        # auc_pred_2 = torch.log(para_function_auc_vdss_t1_2*1000*t1_2_pred/vdss_pred)
        
        ## TODO: design different contrrastive loss to align them 
        k1 = para_function_auc_cl * K_1_MEAN
        k1 = cutoff_k1(k1)
        
        k2 = para_function_auc_vdss_t1_2 * K_2_MEAN
        k2 = cutoff_k2(k2)
        
        auc_pred_1 = t1_2_pred * k1 / vdss_pred
        auc_pred_2 = k2 / cl_pred 
        
        y = batch.y.reshape(batch.y.size(0), 1)
        all_result = torch.cat((auc_pred_1, auc_pred_2, y), dim=1)
        
        loss_1 = criterion(auc_pred_1, y)
        loss_2 = criterion(auc_pred_2, y)
        loss_3 = criterion(auc_pred, y)
        
        loss = loss_1 + loss_2 + loss_3 
     
        total_loss_1 += loss_1
        total_loss_2 += loss_2 
        total_loss_3 += loss_3 

        train_pred_1.append(auc_pred_1)
        train_pred_2.append(auc_pred_2)
    
        train_label.append(y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
  
    
    return total_loss_1/count, total_loss_2/count, total_loss_3/count 



def eval(args, cl_model, t1_2_model, vdss_model, auc_model, para_loss, para_function_auc_cl, para_function_auc_vdss_t1_2,  device, loader, epoch): # para_contrastive_loss,
    cl_model.eval()
    t1_2_model.eval()
    vdss_model.eval()
    auc_model.eval()
    
    
    y_true = []
    y_scores_1 = []
    y_scores_2 = []
    y_scores_3 = []
    
    y_true_value = []

    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            cl_pred_log = cl_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True)
            t1_2_pred_log = t1_2_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True)
            vdss_pred_log = vdss_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True)
            auc_pred_log = auc_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True)
            

            y_true_value.append(batch.y.reshape(batch.y.size(0), 1))
        
            cl_pred = torch.exp(cl_pred_log)
            t1_2_pred = torch.exp(t1_2_pred_log)
            vdss_pred = torch.exp(vdss_pred_log)
            auc_pred = torch.exp(auc_pred_log)
            
            # auc_pred_1 = torch.log(para_function_auc_cl*16000.0/cl_pred)
            # auc_pred_2 = torch.log(para_function_auc_vdss_t1_2*1000*t1_2_pred/vdss_pred)
            
            k1 = para_function_auc_cl * K_1_MEAN
            k1 = cutoff_k1(k1)
            
            k2 = para_function_auc_vdss_t1_2 * K_2_MEAN
            k2 = cutoff_k2(k2)
            
            auc_pred_1 = t1_2_pred * k1 / vdss_pred
            auc_pred_2 = k2 / cl_pred 
            
            y = batch.y.reshape(batch.y.size(0), 1)
            
            loss_1 = criterion(auc_pred_1, y)
            loss_2 = criterion(auc_pred_2, y)
            loss_3 = criterion(auc_pred, y)
            
            loss = loss_1 + loss_2 + loss_3 
     
    
            y_scores_1.append(auc_pred_1)
            y_scores_2.append(auc_pred_2)
            y_scores_3.append(auc_pred)
            
    y_scores_1 = torch.cat(y_scores_1, dim = 0)
    y_scores_2 = torch.cat(y_scores_2, dim = 0)
    y_scores_3 = torch.cat(y_scores_3, dim = 0)
    
    y_true_value = torch.cat(y_true_value, dim = 0)
    
    loss_1 = criterion(y_scores_1, y_true_value).cpu().detach().item()
    loss_2 = criterion(y_scores_2, y_true_value).cpu().detach().item()
    loss_3 = criterion(y_scores_3, y_true_value).cpu().detach().item()
            
    # loss = para_loss*loss_1 + (1-para_loss)*loss_2 

    return loss_1, loss_2, loss_3 


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.00005,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0.000000001,
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
    parser.add_argument('--dataset', type=str, default = 'auc', help='root directory of dataset. For now, only classification.')
    
    ## load three datasets' pretrain model 
    parser.add_argument('--input_vdss_model_file', type=str, default = 'results/atoms_0.75_motifs_0.1/cl/graphmae_lr_0.005_decay_1e-11_bz_32_seed_4_model.pt', help='filename to read the model (if there is any)')
    parser.add_argument('--input_cl_model_file', type=str, default = 'results/atoms_0.75_motifs_0.1/cl/graphmae_lr_0.005_decay_1e-11_bz_32_seed_4_model.pt', help='filename to read the model (if there is any)')
    parser.add_argument('--input_t1_2_model_file', type=str, default = 'results/atoms_0.75_motifs_0.1/cl/graphmae_lr_0.005_decay_1e-11_bz_32_seed_4_model.pt', help='filename to read the model (if there is any)')
    parser.add_argument('--input_auc_model_file', type=str, default= 'results/atoms_0.75_motifs_0.1/cl/graphmae_lr_0.005_decay_1e-11_bz_32_seed_4_model.pt')
    
    parser.add_argument('--filename', type=str, default = 'checkpoints/gin_100.pth', help='output filename')
    parser.add_argument('--seed', type=int, default=4, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=1, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default=0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
    parser.add_argument('--scheduler', action="store_true", default=False)
    parser.add_argument('--experiment_name', type=str, default="graphmae")
    
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--delta_mse_loss', type=float, default=0.1)
    
    ## add some argument 
    parser.add_argument('--dataset_type', type=int, default=1)
    parser.add_argument('--save', type=str, default='')
    args = parser.parse_args()

    args.use_early_stopping = args.dataset in ("muv", "hiv")
    args.scheduler = args.dataset in ("bace")
    

    para_loss = nn.Parameter(torch.tensor(0.5, device=args.device))
    para_function_auc_cl = nn.Parameter(torch.tensor(1.0, device=args.device))
    para_function_auc_vdss_t1_2 = nn.Parameter(torch.tensor(1.0, device=args.device))
    # para_contrastive_loss = nn.Parameter(torch.tensor(0.5, device=args.device))                       
    
    print('para_loss',para_loss)
    print('para_function_auc_cl',para_function_auc_cl)
    print('para_function_auc_vdss_t1_2',para_function_auc_vdss_t1_2)
    # print('para_contrastive_loss', para_contrastive_loss )

  
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
        else:
            raise ValueError("Invalid dataset name.")

        ## set up pk dataset 
        train_dataset = MoleculeDataset("dataset/"+train_dataset_name, dataset=train_dataset_name)
        valid_dataset = MoleculeDataset("dataset/"+valid_dataset_name, dataset=valid_dataset_name)
        # test_dataset = MoleculeDataset("dataset/"+test_dataset_name, dataset=test_dataset_name)
        print(train_dataset)
        print(valid_dataset)

        # print(test_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        #set up model
        cl_model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, classnum=args.dataset_type)
        t1_2_model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, classnum=args.dataset_type)
        vdss_model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, classnum=args.dataset_type)
        auc_model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type, classnum=args.dataset_type)


        if not args.input_cl_model_file == "":
            print("load pretrained model from:", args.input_cl_model_file)
            cl_model.load_state_dict(torch.load(args.input_cl_model_file, map_location=device))

        if not args.input_t1_2_model_file == "":
            print("load pretrained model from:", args.input_t1_2_model_file)
            t1_2_model.load_state_dict(torch.load(args.input_t1_2_model_file, map_location=device))
            
        if not args.input_vdss_model_file == "":
            print("load pretrained model from:", args.input_vdss_model_file)
            vdss_model.load_state_dict(torch.load(args.input_vdss_model_file, map_location=device))
            
        if not args.input_auc_model_file == "":
            print('load pretriained model form:', args.input_auc_model_file)
            auc_model.load_state_dict(torch.load(args.input_auc_model_file, map_location=device))

        cl_model.to(device)
        t1_2_model.to(device)
        vdss_model.to(device)
        auc_model.to(device)

        #set up optimizer
        #different learning rate for different part of GNN
        
        ## TODO: set different lr to different models 
        model_param_group = []
        model_param_group.append({"params": cl_model.gnn.parameters()})
        model_param_group.append({"params": t1_2_model.gnn.parameters()})
        model_param_group.append({"params": vdss_model.gnn.parameters()})
        model_param_group.append({'params': auc_model.gnn.parameters()})
        model_param_group.append({"params": para_loss})
        model_param_group.append({"params": para_function_auc_cl})
        model_param_group.append({"params": para_function_auc_vdss_t1_2})
        
        #  model_param_group.append({'params': para_contrastive_loss})

        if args.graph_pooling == "attention":
            model_param_group.append({"params": cl_model.pool.parameters(), "lr":args.lr*args.lr_scale})
            model_param_group.append({"params": t1_2_model.pool.parameters(), "lr":args.lr*args.lr_scale})
            model_param_group.append({"params": vdss_model.pool.parameters(), "lr":args.lr*args.lr_scale}) 
            
            model_param_group.append({'params': auc_model.pool.parameters(), "lr":args.lr*args.lr_scale})  
        model_param_group.append({"params": cl_model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
        model_param_group.append({"params": t1_2_model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
        model_param_group.append({"params": vdss_model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
        
        model_param_group.append({"params": auc_model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
        print(optimizer)

        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)
        else:
            scheduler = None

        train_mae_loss_list_1 = []
        train_mae_loss_list_2 = []
        train_mae_loss_list_3 = []
        
        val_mae_loss_list_1 = []
        val_mae_loss_list_2 = []
        val_mae_loss_list_3 = []

        best_val_mse_loss=float('inf')

        times = 0
        for epoch in range(1, args.epochs+1):
            print("====epoch " + str(epoch))
            
            train_loss_1, train_loss_2, train_loss_3 = train(args, cl_model, t1_2_model, vdss_model, auc_model, para_loss, para_function_auc_cl, para_function_auc_vdss_t1_2,  device, train_loader, optimizer, epoch) # para_contrastive_loss,
            
            train_mae_loss_list_1.append(train_loss_1)
            train_mae_loss_list_2.append(train_loss_2)
            train_mae_loss_list_3.append(train_loss_3)
            
            if scheduler is not None:
                scheduler.step()

            # print("====Evaluation")
            # if args.eval_train:
            #     train_mse_loss = eval(args, cl_model,t1_2_model, vdss_model, auc_model, para_loss, para_function_auc_cl, para_function_auc_vdss_t1_2,  device, train_loader) # para_contrastive_loss,
            # else:
            #     print("omit the training accuracy computation")
            #     train_mse_loss = 0
            
            
            ## TODO: need to rewrite 
            val_loss_1, val_loss_2, val_loss_3 = eval(args, cl_model, t1_2_model, vdss_model, auc_model, para_loss, para_function_auc_cl, para_function_auc_vdss_t1_2,  device, val_loader, epoch) # para_contrastive_loss,
            if best_val_mse_loss > (val_loss_1 + val_loss_2 + val_loss_3)/3:
                best_val_mse_loss = (val_loss_1 + val_loss_2 + val_loss_3)/3
                
                # torch.save(cl_model.state_dict(), "results/auc_models/cl/"+args.experiment_name+"_"+"cl_model.pt")
                # torch.save(t1_2_model.state_dict(), "results/auc_models/t1_2/"+args.experiment_name+"_"+"t1_2_model.pt")
                # torch.save(vdss_model.state_dict(), "results/auc_models/vdss/"+args.experiment_name+"_"+"vdss_model.pt")
                
                
            print('best_val_mse_loss', best_val_mse_loss)
           
            print("train: %f val: %f " %(train_loss_1, val_loss_1))
            
            
            
            # auc_val_mse_loss_list.append(val_mse_loss)
            
            # cl_val_mse_loss_list.append(cl_val_mse_loss)
            # t1_2_val_mse_loss_list.append(t1_2_val_mse_loss)
            # vdss_val_mse_loss_list.append(vdss_val_mse_loss)
            
            # test_acc_list.append(test_acc)
            # train_mse_loss_list.append(train_mse_loss)
            
            
            # dataframe_1 = pandas.DataFrame({'train_mse_loss':train_mse_loss_list,'auc_valid_mse_loss':auc_val_mse_loss_list,'cl_valid_mse_loss':cl_val_mse_loss,'t1_2_valid_mse_loss':cl_val_mse_loss_list,'t1_2_valid_mse_loss':t1_2_val_mse_loss_list, 'vdss_volid_mse_loss':vdss_val_mse_loss_list})
            dataframe_1 = pandas.DataFrame({'train_valid_loss':train_mse_loss_list, 'auc_valid_mse_loss':auc_val_mse_loss_list,})
            dataframe_1.to_csv("results/auc_models/auc/"+args.experiment_name+"_"+str(args.runseed)+"_"+"loss.csv", index=False)
            
            all_best_val_mean_loss.append(best_val_mse_loss)
            
        # cl_model.load_state_dict(torch.load("results/auc_models/cl/"+args.experiment_name+"_"+"model.pt", map_location=device))
        # t1_2_model.load_state_dict(torch.load("results/auc_models/t1_2/"+args.experiment_name+"_"+"model.pt", map_location=device))
        # vdss_model.load_state_dict(torch.load("results/auc_models/vdss/"+args.experiment_name+"_"+"model.pt", map_location=device))
    
        # test_mse_loss = eval(args, cl_model, t1_2_model, vdss_model, para_loss, para_function_auc_cl, para_function_auc_vdss_t1_2, para_contrastive_loss, device, test_loader)



    mean_val_mse_loss = np.mean(np.array(all_best_val_mean_loss))
    dataframe = pandas.DataFrame({'val_mse_loss':[mean_val_mse_loss]})
    dataframe.to_csv("results/auc_models/auc/"+args.experiment_name+"_"+"result.csv", index=False)
    
    

if __name__ == "__main__":
    main()

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

K_1 = 16846
K_2 = 17.71 # 18

def cutoff(k):
    max_value = 503.13
    min_value = 1315.3 
    
    if k < min_value:
        k = min_value
    if k > max_value:
        k = max_value
    
    return k 

def train(args, vdss_model, t1_2_model, cl_model,  para_loss, para_function_cl_vdss_t1_2, para_function_auc_cl,  device, loader, optimizer, epoch): # para_contrastive_loss,
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
        
        k_1 = para_function_auc_cl * K_1
        k_3 = para_function_cl_vdss_t1_2 * K_2
        
        auc_pred = k_1/cl_pred 
        auc_pred_log = torch.log(auc_pred)
        
        auc_y = batch.auc_y.view(batch.auc_y.size(0), 1)
        cl_y = batch.cl_y.view(batch.cl_y.size(0), 1)
        vdss_y = batch.vdss_y.view(batch.vdss_y.size(0), 1)
        t1_2_y = batch.t1_2_y.view(batch.t1_2_y.size(0), 1)
        
        loss_auc = criterion(auc_pred_log, auc_y)
        loss_cl = criterion(cl_pred_log, cl_y)
        loss_vdss = criterion(vdss_pred_log, vdss_y)
        loss_t1_2 = criterion(t1_2_pred_log, t1_2_y)
        
        loss_k3 = criterion(torch.log(k_3 * vdss_pred), torch.log(t1_2_pred * cl_pred))
        loss = 4 * loss_auc + loss_cl + loss_vdss + loss_t1_2 + loss_k3
       
        total_loss += loss.item()
        
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
   
    
    train_auc_pred_all = torch.cat(train_auc_pred_all, dim=0).detach().cpu().numpy()
    train_cl_pred_all = torch.cat(train_cl_pred_all, dim=0).detach().cpu().numpy()
    train_vdss_pred_all = torch.cat(train_vdss_pred_all, dim=0).detach().cpu().numpy()
    train_t1_2_pred_all = torch.cat(train_t1_2_pred_all, dim=0).detach().cpu().numpy()
    
    train_auc_label_all = torch.cat(train_auc_label_all, dim=0).detach().cpu().numpy()
    train_cl_label_all = torch.cat(train_cl_label_all, dim=0).detach().cpu().numpy()
    train_vdss_label_all = torch.cat(train_vdss_label_all, dim=0).detach().cpu().numpy()
    train_t1_2_label_all = torch.cat(train_t1_2_label_all, dim=0).detach().cpu().numpy()
    
    
    return (total_loss)/count
   
def eval(args, vdss_model, t1_2_model, cl_model, para_loss, para_function_cl_vdss_t1_2, para_function_auc_cl,  device, loader, epoch): # para_contrastive_loss,
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
            vdss_pred_log, vdss_representation = vdss_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, False)
            t1_2_pred_log, t1_2_representation = t1_2_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, False)
            cl_pred_log , cl_representation = cl_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, False)
           
            cl_pred = torch.exp(cl_pred_log)
         
            k_1 = para_function_auc_cl * K_1
            k_3 = para_function_cl_vdss_t1_2 * K_2
            
            auc_pred_log = torch.log(k_1/cl_pred)
            
            auc_y = batch.auc_y.view(batch.auc_y.size(0), 1)
            cl_y = batch.cl_y.view(batch.cl_y.size(0), 1)
            vdss_y = batch.vdss_y.view(batch.vdss_y.size(0), 1)
            t1_2_y = batch.t1_2_y.view(batch.t1_2_y.size(0), 1)
            
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
    
    ## mfce 
    auc_mfce = torch.exp(torch.median(torch.abs(y_scores_auc - y_auc)))
    cl_mfce = torch.exp(torch.median(torch.abs(y_scores_cl - y_cl)))
    vdss_mfce = torch.exp(torch.median(torch.abs(y_scores_vdss - y_vdss)))
    t1_2_mfce = torch.exp(torch.median(torch.abs(y_scores_t1_2 - y_t1_2)))
    
        
    loss_all = (loss_auc + loss_cl + loss_vdss + loss_t1_2 )
   
    return loss_auc, loss_cl, loss_vdss, loss_t1_2, loss_all
           

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=3,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32, ## 32: 1.009 # 64: 
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=2, # 60 # 1500
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
    
    
    ## load three datasets' pretrain model 
    parser.add_argument('--input_vdss_model_file', type=str, default = 'results/11_13/finetune/vdss/graphmae_lr_0.0005_decay_1e-10_bz_64_seed_4_5model.pt', help='filename to read the model (if there is any)')
    parser.add_argument('--input_t1_2_model_file', type=str, default = 'results/11_13/finetune/t1_2/graphmae_lr_0.0001_decay_1e-10_bz_64_seed_4_1model.pt', help='filename to read the model (if there is any)')
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
    
    
    ## add the multi-task alpha 
    parser.add_argument('--t1_2_alpha', type=float, default=1.)
    parser.add_argument('--k3_alphla', type=float, default=1.)
    
    
    ## add the each model scale 
    parser.add_argument('--cl_model_scale', type=float, default=1.)
    parser.add_argument('--vdss_model_scale', type=float, default=1.)
    parser.add_argument('--t1_2_model_scale', type=float, default=2.2)
    
    
    ## add some argument 
    parser.add_argument('--dataset_type', type=int, default=1)
    parser.add_argument('--save', type=str, default='')
    args = parser.parse_args()


    args.use_early_stopping = args.dataset in ("muv", "hiv")
    args.scheduler = args.dataset in ("bace")
    
    
    para_loss = nn.Parameter(torch.tensor(0.5, device=args.device), requires_grad=True)
    para_function_cl_vdss_t1_2 = nn.Parameter(torch.tensor(1.0, device=args.device), requires_grad=True)
    para_function_auc_cl = nn.Parameter(torch.tensor(1.0, device=args.device), requires_grad=True)
    
    
    para_loss.data.fill_(0.5)
    para_function_cl_vdss_t1_2.data.fill_(1.)
    para_function_auc_cl.data.fill_(1.)
 
 
    print('para_loss',para_loss)
    print('para_function_cl_vdss_t1_2',para_function_cl_vdss_t1_2)
    print('para_function_auc_cl', para_function_auc_cl)
    
 
    all_best_val_mean_loss = []
    all_best_auc_mean_loss = []
    all_best_cl_mean_loss = []
    all_best_t1_2_mean_loss = []
    all_best_vdss_mean_loss = []
    
    
    all_auc_mfce = []
    all_cl_mfce = []
    all_vdss_mfce = []
    all_t1_2_mfce = []
    
    
    best_epoch = 0
    for args.runseed in [1, 2, 3, 4, 5]:
        torch.manual_seed(args.runseed)
        np.random.seed(args.runseed)
        # args.seed = args.runseed 
        args.experiment_name = 'lr'+'_'+str(args.lr)+'_'+'decay'+'_'+str(args.decay)+'_'+'bz'+'_'+str(args.batch_size)+'_'+'seed'+'_'+str(args.seed)

        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.runseed)

        if args.dataset == "4":
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
        model_param_group.append({'params': cl_model.gnn.parameters()})
        model_param_group.append({'params': t1_2_model.gnn.parameters(), "lr":args.lr * args.t1_2_model_scale})

        model_param_group.append({"params": para_loss})
        model_param_group.append({"params": para_function_cl_vdss_t1_2})
        model_param_group.append({'params': para_function_auc_cl})

        if args.graph_pooling == "attention":
            model_param_group.append({"params": vdss_model.pool.parameters(), "lr":args.lr*args.lr_scale})
            model_param_group.append({'params': cl_model.pool.parameters(), 'lr':args.lr*args.lr_scale})
            model_param_group.append({'params': t1_2_model.pool.parameters(), "lr":args.lr * args.t1_2_model_scale})
        model_param_group.append({"params": vdss_model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
        model_param_group.append({'params': cl_model.graph_pred_linear.parameters(), 'lr':args.lr*args.lr_scale })
        model_param_group.append({'params': t1_2_model.graph_pred_linear.parameters(), "lr":args.lr * args.t1_2_model_scale})
        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
        print(optimizer)

        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=45)
        else:
            scheduler = None

        epoch_list = np.arange(0, args.epochs, 1)
        print('epoch_list_len',len(epoch_list))
        train_mse_loss_list = []
        val_mse_loss_list = []
        
        best_val_mse_loss=float('inf')
        best_true_val_mse_loss = float('inf')
       
        
        times = 0
        for epoch in range(1, args.epochs+1):
            print("====epoch " + str(epoch))
            
            train_mae_loss = train(args, vdss_model, t1_2_model, cl_model, para_loss, para_function_cl_vdss_t1_2, para_function_auc_cl, device, train_loader, optimizer, epoch) # para_contrastive_loss,
            train_mse_loss_list.append(train_mae_loss)

            auc_val_loss, cl_val_loss, vdss_val_loss, t1_2_val_loss, all_loss = eval(args, vdss_model, t1_2_model, cl_model, para_loss, para_function_cl_vdss_t1_2, para_function_auc_cl, device, val_loader, epoch) # para_contrastive_loss,
            
            if scheduler is not None:
                scheduler.step(metrics=all_loss)
                
            if best_val_mse_loss > all_loss:
                best_val_mse_loss = all_loss
                best_epoch = epoch 
                best_auc_loss = auc_val_loss
                best_cl_loss = cl_val_loss 
                best_vdss_loss = vdss_val_loss
                best_t1_2_loss = t1_2_val_loss
             
            print('best epoch:', best_epoch)
            print('best_val_mse_loss', best_val_mse_loss)
            print('best_auc_loss:', best_auc_loss)
            print('best_cl_loss:', best_cl_loss)
            print('best_vdss_loss:', best_vdss_loss)
            print('best_t1_2_loss:', best_t1_2_loss)
         
            print('at the meanwhile, the true valid mse loss', best_true_val_mse_loss)
            print("train: %f val: %f " %(train_mae_loss, best_val_mse_loss))
            val_mse_loss_list.append(all_loss)
            
        all_best_val_mean_loss.append(best_val_mse_loss)
        all_best_auc_mean_loss.append(best_auc_loss)
        all_best_cl_mean_loss.append(best_cl_loss)
        all_best_vdss_mean_loss.append(best_vdss_loss)
        all_best_t1_2_mean_loss.append(best_t1_2_loss)
      
    mean_val_mse_loss = np.mean(np.array(all_best_val_mean_loss))
    mean_auc_loss = np.mean(np.array(all_best_auc_mean_loss))
    mean_cl_loss = np.mean(np.array(all_best_cl_mean_loss))
    mean_vdss_loss = np.mean(np.array(all_best_vdss_mean_loss))
    mean_t1_2_loss = np.mean(np.array(all_best_t1_2_mean_loss))
    
    std_auc_loss = np.std(np.array(all_best_auc_mean_loss))
    std_cl_loss = np.std(np.array(all_best_cl_mean_loss))
    std_vdss_loss = np.std(np.array(all_best_vdss_mean_loss))
    std_t1_2_loss = np.std(np.array(all_best_t1_2_mean_loss))
    
    print('mean_auc_loss', mean_auc_loss)
    print('mean_cl_loss', mean_cl_loss)
    print('mean_vdss_loss', mean_vdss_loss)
    print('mean_t1_2_loss', mean_t1_2_loss)
    
    print('std auc loss', std_auc_loss)
    print('std cl loss', std_cl_loss)
    print('std vdss loss', std_vdss_loss)
    print('std t1_2 loss', std_t1_2_loss)
    
    dataframe = pandas.DataFrame({'val_mse_loss':[mean_val_mse_loss], 'auc_loss':[mean_auc_loss], 'cl_loss':[mean_cl_loss] , 'vdss_loss':[mean_vdss_loss], 't1_2_loss':[mean_t1_2_loss]})
    dataframe.to_csv("results/"+args.experiment_name+"_"+"result.csv", index=False)
    
    

if __name__ == "__main__":
    main()


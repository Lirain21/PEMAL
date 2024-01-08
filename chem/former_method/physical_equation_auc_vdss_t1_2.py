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

def train(args, vdss_model, t1_2_model,  para_loss, para_function_auc_vdss_t1_2,  device, loader, optimizer, epoch): # para_contrastive_loss,
    vdss_model.train()
    t1_2_model.train()
   
    total_loss = 0 
    count = 0 
    
    train_pred_1 = []
    train_pred_2 = []
    train_label = []
    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
      
        count += 1 
        vdss_pred_log = vdss_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True)
        t1_2_pred_log = t1_2_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True)
       
        vdss_pred = torch.exp(vdss_pred_log)
        t1_2_pred = torch.exp(t1_2_pred_log)
    
        # auc_pred_1 = torch.log(para_function_auc_cl*16000.0/cl_pred)
        ## TODO: leverage the mean +- 3 * std to avaluate the k 
        k = para_function_auc_vdss_t1_2 * K_MEAN
        k = cutoff(k)
        auc_pred_1 = torch.log(k * t1_2_pred / vdss_pred)
        y = batch.y.reshape(batch.y.size(0), 1)
        
        loss = criterion(auc_pred_1, y)
       
        total_loss += loss.item()
        train_pred_1.append(auc_pred_1)
        train_label.append(y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
  
    # if epoch == 8:
    #     train_pred_1 = torch.cat(train_pred_1, dim=0).detach().cpu().numpy()
    #     # train_pred_2 = torch.cat(train_pred_2, dim=0).detach().cpu().numpy()
    #     train_label = torch.cat(train_label, dim=0).detach().cpu().numpy()
    #     # train_label = train_label.cpu().detach().numpy()
    #     # train_pred_1 = train_pred_1.cpu().detach().numpy()
        
    #     plt.plot(train_label, train_label)
    #     plt.scatter(train_label, train_pred_1)
    #     plt.savefig('imgs/physical_auc_scatter.png')
    
    return (total_loss)/count



def eval(args, vdss_model, t1_2_model, para_loss, para_function_auc_vdss_t1_2,  device, loader, epoch): # para_contrastive_loss,
    vdss_model.eval()
    t1_2_model.eval()
    y_scores_1 = []
    y_scores_2 = []
    
    y_true_value = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            vdss_pred_log = vdss_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True)
            t1_2_pred_log = t1_2_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, True)
           
            y_true_value.append(batch.y.reshape(batch.y.size(0), 1))
        
            vdss_pred = torch.exp(vdss_pred_log)
            t1_2_pred = torch.exp(t1_2_pred_log)
                
            k = para_function_auc_vdss_t1_2 * K_MEAN
            k = cutoff(k)
            auc_pred_1 = torch.log(k * t1_2_pred / vdss_pred)
            y_scores_1.append(auc_pred_1)
            
            # y_scores_2.append(auc_pred_2)
            
    y_scores_1 = torch.cat(y_scores_1, dim = 0)
    y_true_value = torch.cat(y_true_value, dim = 0)
    
    loss = criterion(y_scores_1, y_true_value).cpu().detach().item()
  
    # if epoch == 8:
    #     y_scores_1 = y_scores_1.cpu().detach().numpy()
    #     y_true_value = y_true_value.cpu().detach().numpy()
        
    #     plt.plot(y_true_value, y_true_value)
    #     plt.scatter(y_true_value, y_scores_1)
    #     plt.savefig('imgs/physical_auc_scatter.png')
   
    return loss
  

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128, ## 32: 1.009 # 64: 
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=30, # 1500
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
    parser.add_argument('--dataset', type=str, default = 'auc', help='root directory of dataset. For now, only classification.')
    
    ## load three datasets' pretrain model 
    parser.add_argument('--input_vdss_model_file', type=str, default = 'results/atoms_0.75_motifs_0.1/vdss/graphmae_lr_0.005_decay_1e-11_bz_64_seed_4_model.pt', help='filename to read the model (if there is any)')
    parser.add_argument('--input_t1_2_model_file', type=str, default = 'results/atoms_0.75_motifs_0.1/t1_2/graphmae_lr_0.005_decay_1e-11_bz_64_seed_4_model.pt', help='filename to read the model (if there is any)')
    
    
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
    para_function_auc_vdss_t1_2 = nn.Parameter(torch.tensor(1.0, device=args.device))
 
    print('para_loss',para_loss)
    print('para_function_auc_vdss_t1_2',para_function_auc_vdss_t1_2)
 
    all_best_val_mean_loss = []
    best_epoch = 0
    for args.runseed in [2, 3, 4, 5]:
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

        if not args.input_vdss_model_file == "":
            print("load pretrained model from:", args.input_vdss_model_file)
            vdss_model.load_state_dict(torch.load(args.input_vdss_model_file, map_location=device))
        
        if not args.input_t1_2_model_file == "":
            print("load pretrained model from:", args.input_t1_2_model_file)
            t1_2_model.load_state_dict(torch.load(args.input_t1_2_model_file, map_location=device))
            
        vdss_model.to(device)
        t1_2_model.to(device)

        #set up optimizer
        #different learning rate for different part of GNN
        model_param_group = []
        model_param_group.append({"params": vdss_model.gnn.parameters()})
        model_param_group.append({'params': t1_2_model.gnn.parameters()})

        model_param_group.append({"params": para_loss})
        model_param_group.append({"params": para_function_auc_vdss_t1_2})

        if args.graph_pooling == "attention":
            model_param_group.append({"params": vdss_model.pool.parameters(), "lr":args.lr*args.lr_scale})
            model_param_group.append({'params': t1_2_model.pool.parameters(), "lr":args.lr*args.lr_scale})
        model_param_group.append({"params": vdss_model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
        model_param_group.append({'params': t1_2_model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
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

        best_val_mse_loss=float('inf')
        best_true_val_mse_loss = float('inf')
        times = 0
        for epoch in range(1, args.epochs+1):
            print("====epoch " + str(epoch))
            
            train_mes_loss = train(args, vdss_model, t1_2_model, para_loss, para_function_auc_vdss_t1_2, device, train_loader, optimizer, epoch) # para_contrastive_loss,
            train_mse_loss_list.append(train_mes_loss)
            
            if scheduler is not None:
                scheduler.step()

            print("====Evaluation")
            if args.eval_train:
                train_mse_loss = eval(args, vdss_model, t1_2_model, para_loss, para_function_auc_vdss_t1_2, device, train_loader, epoch) # para_contrastive_loss,
            else:
                print("omit the training accuracy computation")
                train_mse_loss = 0
            
            val_mse_loss = eval(args, vdss_model, t1_2_model, para_loss, para_function_auc_vdss_t1_2,  device, val_loader, epoch) # para_contrastive_loss,
            if best_val_mse_loss > val_mse_loss:
                best_val_mse_loss = val_mse_loss 
                best_epoch = epoch 
                
            # if val_mse_loss > best_val_mse_loss:
            #     times += 1
            #     print('times', times)
            #     if times >= args.patience:
            #         print('Early stopping !!!')
                    
            #         break 
                
                # torch.save(cl_model.state_dict(), "results/auc_models/cl/"+args.experiment_name+"_"+"cl_model.pt")
                # torch.save(t1_2_model.state_dict(), "results/auc_models/t1_2/"+args.experiment_name+"_"+"t1_2_model.pt")
                # torch.save(vdss_model.state_dict(), "results/auc_models/vdss/"+args.experiment_name+"_"+"vdss_model.pt")
                
            print('best epoch:', best_epoch)
            print('best_val_mse_loss', best_val_mse_loss)
            print('at the meanwhile, the true valid mse loss', best_true_val_mse_loss)
            print("train: %f val: %f " %(train_mes_loss, val_mse_loss))
            val_mse_loss_list.append(val_mse_loss)
            
            # cl_val_mse_loss_list.append(cl_val_mse_loss)
            # t1_2_val_mse_loss_list.append(t1_2_val_mse_loss)
            # vdss_val_mse_loss_list.append(vdss_val_mse_loss)
            
            # test_acc_list.append(test_acc)
            # train_mse_loss_list.append(train_mse_loss)
            
            
            # dataframe_1 = pandas.DataFrame({'train_mse_loss':train_mse_loss_list,'auc_valid_mse_loss':auc_val_mse_loss_list,'cl_valid_mse_loss':cl_val_mse_loss,'t1_2_valid_mse_loss':cl_val_mse_loss_list,'t1_2_valid_mse_loss':t1_2_val_mse_loss_list, 'vdss_volid_mse_loss':vdss_val_mse_loss_list})
            # dataframe_1 = pandas.DataFrame({'train_valid_loss':train_mse_loss_list, 'auc_valid_mse_loss':val_mse_loss_list,})
            # dataframe_1.to_csv("results/physical_equation_result/11_6/auc/mae/"+args.experiment_name+"_"+str(args.runseed)+"_"+"auc_cl.csv", index=False)
            
            all_best_val_mean_loss.append(best_val_mse_loss)
            
        
        # plt.plot(epoch_list, train_mse_loss_list)
        # plt.plot(epoch_list, val_mse_loss_list)
        # # plt.x_label('epoch')
        # # plt.y_label('mae')
        # plt.savefig('imgs/physical_loss_curve_'+str(args.runseed)+'.png')
        
        exit()
        
  
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
    dataframe = pandas.DataFrame({'val_mse_loss':[mean_val_mse_loss]})
    dataframe.to_csv("results/physical_equation_result/11_6/auc/mae/"+args.experiment_name+"_"+"result.csv", index=False)
    
    

if __name__ == "__main__":
    main()

import argparse
from functools import partial

from loader import MoleculeDataset
from dataloader import DataLoaderMaskingPred #, DataListLoader
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNNDecoder
from loss import ContrastiveLoss 

from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

from util import MaskAtom

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from tensorboardX import SummaryWriter
import random
import matplotlib.pyplot as plt
import timeit
import time 

def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)

def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

def get_atom_level_mol_emb(batch, node_rep, batch_size):
    mol_rep = []
    count = 0
    for i in range(batch_size):
        num = sum(batch.batch == i)
        per_mol_rep = torch.sum(node_rep[count:count+num], dim=0)
        mol_rep.append(per_mol_rep.unsqueeze(0))
    
    mol_representation = torch.cat(mol_rep, dim=0)
    
    # print(mol_representation.size())
    return mol_representation
  
  
def get_motif_level_mol_emb(batch, node_rep, batch_size):
    mol_rep = []
    count = 0
    motif_count = 0
    for i in range(batch_size):
        num = sum(batch.batch == i)
        motif_num = torch.unique(batch.motifs[count:count+num]).size(0)
        per_mol_rep = torch.sum(node_rep[motif_count:motif_count+motif_num], dim=0)
        mol_rep.append(per_mol_rep.unsqueeze(0))
    
    mol_representation = torch.cat(mol_rep, dim=0)
    # print(mol_representation.size())
    return mol_representation

    
def get_per_motif_emb(batch, motifs_node_rep, batch_size):
    motifs_rep = []
    count = 0
    for i in range(batch_size):
        num = sum(batch.batch == i)
        motif_index = batch.motifs[count:count+num]
        all_node_rep = motifs_node_rep[count:count+num]
        for j in torch.unique(motif_index):
            per_motif_index = torch.nonzero(motif_index==j)
            motif_index = per_motif_index.squeeze(-1)
            pre_motif_emb = torch.sum(torch.index_select(all_node_rep, dim=0, index=motif_index), dim=0).unsqueeze(0)
            motifs_rep.append(pre_motif_emb)
    
    motifs_rep = torch.cat(motifs_rep, dim=0)
    return motifs_rep
    

def train_mae(args, model_list, loader,  contrastive_loss_function, optimizer_list, device, alpha_l=1.0, loss_fn="sce"):
    if loss_fn == "sce":
        criterion = partial(sce_loss, alpha=alpha_l)
    else:
        criterion = nn.CrossEntropyLoss()

    gnn_model, atom_pred_decoder, motif_pres_decoder, bond_pred_decoder = model_list
    optimizer_gnn_model, optimizer_dec_pred_atoms, optimizer_dec_pred_bonds, optimizer_dec_pred_motifs = optimizer_list
    
    gnn_model.train()
    atom_pred_decoder.train()
    motif_pres_decoder.train()

    if bond_pred_decoder is not None:
        bond_pred_decoder.train()

    loss_accum = 0
    acc_node_accum = 0
    acc_edge_accum = 0  

    epoch_iter = tqdm(loader, desc="Iteration")
 
    for step, batch in enumerate(epoch_iter):
        batch = batch.to(device)
        node_rep = gnn_model(batch.x, batch.edge_index, batch.edge_attr)
        motifs_node_rep = gnn_model(batch.original_x, batch.edge_index, batch.edge_attr)
        motifs_rep = get_per_motif_emb(batch, motifs_node_rep, args.batch_size)
        motifs_mask_index = torch.tensor(list(random.sample(range(motifs_rep.size(0)), int(motifs_rep.size(0)*0.25)))).to(motifs_rep.device)  # set the motif mask_radio as hyperparameter

        motifs_rep_label = torch.index_select(motifs_rep, 0, motifs_mask_index)
        motifs_edge_attr = torch.tensor([1, 0]).unsqueeze(0).repeat(batch.sub_adj.size(0), 1).to(motifs_rep.device)  # the motifs_edge_attr maybe has something wrong
        
        pred_motifs = motif_pres_decoder(motifs_rep, batch.sub_adj.reshape(motifs_edge_attr.size(1), motifs_edge_attr.size(0)).int(), motifs_edge_attr, motifs_mask_index)

        node_attr_label = batch.node_attr_label
        masked_node_indices = batch.masked_atom_indices
        pred_node = atom_pred_decoder(node_rep, batch.edge_index, batch.edge_attr, masked_node_indices)
        
        ## loss for nodes
        if loss_fn == "sce":
            atom_loss = criterion(node_attr_label, pred_node[masked_node_indices])
            motifs_loss = criterion(motifs_rep_label, pred_motifs[motifs_mask_index])
        else:
            atom_loss = criterion(pred_node.double()[masked_node_indices], batch.mask_node_label[:,0])
            motifs_loss = criterion(pred_motifs.double()[motifs_mask_index], motifs_rep_label[:,0])
            
        ## calculate the motif-level loss and atom-level loss 
        
        node_level_mol_rep = get_atom_level_mol_emb(batch, node_rep, args.batch_size)
        motif_level_mol_rep = get_motif_level_mol_emb(batch, pred_motifs, args.batch_size)
        
        contras_loss = contrastive_loss_function(node_level_mol_rep, motif_level_mol_rep)
            
        loss = atom_loss + motifs_loss + 0.5 * contras_loss

        if args.mask_edge:
            masked_edge_index = batch.edge_index[:, batch.connected_edge_indices]
            edge_rep = node_rep[masked_edge_index[0]] + node_rep[masked_edge_index[1]]
            pred_edge = bond_pred_decoder(edge_rep)
            loss += criterion(pred_edge.double(), batch.mask_edge_label[:,0])

        optimizer_gnn_model.zero_grad()
        optimizer_dec_pred_atoms.zero_grad()
        optimizer_dec_pred_motifs.zero_grad()

        if optimizer_dec_pred_bonds is not None:
            optimizer_dec_pred_bonds.zero_grad()

        loss.backward()

        optimizer_gnn_model.step()
        optimizer_dec_pred_atoms.step()
        optimizer_dec_pred_motifs.step()

        if optimizer_dec_pred_bonds is not None:
            optimizer_dec_pred_bonds.step()

        loss_accum += float(loss.cpu().item())
        epoch_iter.set_description(f"train_loss: {loss.item():.4f}")

    return loss_accum/step 



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=1,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--mask_rate', type=float, default=0.75,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--mask_edge', type=int, default=0,
                        help='whether to mask edges or not together with atoms')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'ic_50', help='root directory of dataset for pretraining')
    parser.add_argument('--output_model_file', type=str, default = 'checkpoints/contrastive_alpha_0.1_motif_0.25/', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--input_model_file', type=str, default=None)
    parser.add_argument("--alpha_l", type=float, default=1.0)
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--decoder", type=str, default="gin")
    parser.add_argument("--use_scheduler", action="store_true", default=False)
    
    parser.add_argument("--loss_computer", type=str, default="nce_softmax")
    parser.add_argument("--tau", type=float, default=0.7)
    
    args = parser.parse_args()
    print(args)

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print("num layer: %d mask rate: %f mask edge: %d" %(args.num_layer, args.mask_rate, args.mask_edge))


    dataset_name = args.dataset
    dataset = MoleculeDataset("dataset_reg/" + dataset_name, dataset=dataset_name)

    loader = DataLoaderMaskingPred(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, mask_rate=args.mask_rate, mask_edge=args.mask_edge)

    # set up models, one for pre-training and one for context embeddings
    gnn_model = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type).to(device)
    
    ## set the atom-level and motif-level contrastive loss 
    contrastive_loss = ContrastiveLoss(args.loss_computer, args.tau, args)
    
    
    if args.input_model_file is not None and args.input_model_file != "":
        gnn_model.load_state_dict(torch.load(args.input_model_file))
        print("Resume training from:", args.input_model_file)
        resume = True
    else:
        resume = False

    NUM_NODE_ATTR = 119 # + 3 
    atom_pred_decoder = GNNDecoder(args.emb_dim, NUM_NODE_ATTR, JK=args.JK, gnn_type=args.gnn_type).to(device)
    motif_pres_decoder = GNNDecoder(args.emb_dim, args.emb_dim, JK=args.JK, gnn_type=args.gnn_type).to(device)
    
    if args.mask_edge:
        NUM_BOND_ATTR = 5 + 3
        bond_pred_decoder = GNNDecoder(args.emb_dim, NUM_BOND_ATTR, JK=args.JK, gnn_type=args.gnn_type)
        optimizer_dec_pred_bonds = optim.Adam(bond_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)
    else:
        bond_pred_decoder = None
        optimizer_dec_pred_bonds = None

    model_list = [gnn_model,  atom_pred_decoder, motif_pres_decoder, bond_pred_decoder] 

    # set up optimizers
    optimizer_gnn_model = optim.Adam(gnn_model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_dec_pred_atoms = optim.Adam(atom_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_dec_pred_motifs = optim.Adam(motif_pres_decoder.parameters(), lr=args.lr, weight_decay=args.decay)

    if args.use_scheduler:
        print("--------- Use scheduler -----------")
        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / args.epochs) ) * 0.5
        scheduler_gnn_model = torch.optim.lr_scheduler.LambdaLR(optimizer_gnn_model, lr_lambda=scheduler)
        scheduler_atom_dec = torch.optim.lr_scheduler.LambdaLR(optimizer_dec_pred_atoms, lr_lambda=scheduler)
        scheduler_motif_pred = torch.optim.lr_scheduler.LambdaLR(optimizer_dec_pred_motifs, lr_lambda=scheduler)
        scheduler_list = [scheduler_gnn_model, scheduler_atom_dec, scheduler_motif_pred]
    else:
        scheduler_gnn_model = None
        scheduler_atom_dec= None
        scheduler_motif_pred = None 
        

    optimizer_list = [optimizer_gnn_model, optimizer_dec_pred_atoms, optimizer_dec_pred_bonds, optimizer_dec_pred_motifs]
    
    optim_loss = torch.tensor(float('inf')).to(device)
    
    epoch_list, loss_list = [], []
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
  
        train_loss = train_mae(args, model_list, loader, contrastive_loss, optimizer_list, device, alpha_l=args.alpha_l, loss_fn=args.loss_fn)
        if not resume:
            if epoch % 10 == 0 or epoch == 1 or epoch == 2:
                torch.save(gnn_model.state_dict(), args.output_model_file + f"_{epoch}.pth")
        if train_loss < optim_loss:
            optim_loss = train_loss
            torch.save(gnn_model.state_dict(), args.output_model_file + f"best_model.pth")
       
        if scheduler_gnn_model is not None:
            scheduler_gnn_model.step()
        if scheduler_atom_dec is not None:
            scheduler_atom_dec.step()
        if scheduler_motif_pred is not None:
            scheduler_motif_pred.step()
            
        epoch_list.append(epoch)
        loss_list.append(train_loss)
    
    x = np.array(epoch_list)
    y = np.array(loss_list)
    plt.figure(figsize=(6,4))
    plt.plot(x, y, color="red", linewidth=1 )
    plt.savefig('results/contrastive_alpha_0.5_motif_0.25_loss.png',dpi=120, bbox_inches='tight')  

if __name__ == "__main__":
    main()


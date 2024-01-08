#!/bin/bash
batch_size=256
decay=1e-11
lr=0.01
device=1
python newchem/physical_equation_auc_cl_fp_gnn.py \
--batch_size 256 \
--decay 1e-11 \
--lr 0.01 \
--device 1 \

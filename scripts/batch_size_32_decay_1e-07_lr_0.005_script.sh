#!/bin/bash
batch_size=32
seed=4
decay=1e-07
lr=0.005
device=0
python chem/pfc_nopretrain.py \
--batch_size 32 \
--seed 4 \
--decay 1e-07 \
--lr 0.005 \
--device 0 \

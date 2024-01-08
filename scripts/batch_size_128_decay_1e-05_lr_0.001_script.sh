#!/bin/bash
batch_size=128
seed=4
decay=1e-05
lr=0.001
device=2
python chem/pfc_nopretrain.py \
--batch_size 128 \
--seed 4 \
--decay 1e-05 \
--lr 0.001 \
--device 2 \

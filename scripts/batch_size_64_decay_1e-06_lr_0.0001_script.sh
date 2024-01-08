#!/bin/bash
batch_size=64
seed=4
decay=1e-06
lr=0.0001
device=1
python chem/pfc_nopretrain.py \
--batch_size 64 \
--seed 4 \
--decay 1e-06 \
--lr 0.0001 \
--device 1 \

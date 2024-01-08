#!/bin/bash
batch_size=64
seed=4
decay=0.0001
lr=0.0001
device=1
python chem/pfc_nopretrain.py \
--batch_size 64 \
--seed 4 \
--decay 0.0001 \
--lr 0.0001 \
--device 1 \

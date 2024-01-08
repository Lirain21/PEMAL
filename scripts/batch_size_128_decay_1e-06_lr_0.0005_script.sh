#!/bin/bash
batch_size=128
seed=4
decay=1e-06
lr=0.0005
device=2
python chem/pfc_nopretrain.py \
--batch_size 128 \
--seed 4 \
--decay 1e-06 \
--lr 0.0005 \
--device 2 \

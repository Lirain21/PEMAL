#!/bin/bash
batch_size=64
seed=4
decay=1e-09
lr=0.001
device=1
python chem/finetune_reg.py \
--batch_size 64 \
--seed 4 \
--decay 1e-09 \
--lr 0.001 \
--device 1 \

#!/bin/bash
batch_size=64
seed=4
decay=1e-08
lr=0.0001
device=2
python chem/finetune_reg.py \
--batch_size 64 \
--seed 4 \
--decay 1e-08 \
--lr 0.0001 \
--device 2 \

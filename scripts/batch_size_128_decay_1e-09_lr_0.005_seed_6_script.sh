#!/bin/bash
batch_size=128
seed=6
decay=1e-09
lr=0.005
device=3
python chem/finetune_reg.py \
--batch_size 128 \
--seed 6 \
--decay 1e-09 \
--lr 0.005 \
--device 3 \

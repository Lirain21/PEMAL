#!/bin/bash
batch_size=32
seed=4
decay=1e-11
lr=0.001
device=0
python chem/finetune_reg.py \
--batch_size 32 \
--seed 4 \
--decay 1e-11 \
--lr 0.001 \
--device 0 \

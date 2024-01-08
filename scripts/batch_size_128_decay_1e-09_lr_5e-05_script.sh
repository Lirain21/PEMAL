#!/bin/bash
batch_size=128
seed=4
decay=1e-09
lr=5e-05
device=1
python chem/finetune_reg.py \
--batch_size 128 \
--seed 4 \
--decay 1e-09 \
--lr 5e-05 \
--device 1 \

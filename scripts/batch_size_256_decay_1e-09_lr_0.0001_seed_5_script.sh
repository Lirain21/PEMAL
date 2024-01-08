#!/bin/bash
batch_size=256
seed=5
decay=1e-09
lr=0.0001
device=1
python chem/finetune_reg.py \
--batch_size 256 \
--seed 5 \
--decay 1e-09 \
--lr 0.0001 \
--device 1 \

#!/bin/bash
batch_size=256
seed=5
decay=1e-11
lr=0.0001
device=1
python chem/finetune_reg.py \
--batch_size 256 \
--seed 5 \
--decay 1e-11 \
--lr 0.0001 \
--device 1 \

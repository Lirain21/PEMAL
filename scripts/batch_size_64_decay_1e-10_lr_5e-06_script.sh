#!/bin/bash
batch_size=64
seed=4
decay=1e-10
lr=5e-06
device=2
python chem/finetune_reg.py \
--batch_size 64 \
--seed 4 \
--decay 1e-10 \
--lr 5e-06 \
--device 2 \

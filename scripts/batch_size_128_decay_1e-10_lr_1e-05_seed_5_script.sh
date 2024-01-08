#!/bin/bash
batch_size=128
seed=5
decay=1e-10
lr=1e-05
device=3
python chem/finetune_reg.py \
--batch_size 128 \
--seed 5 \
--decay 1e-10 \
--lr 1e-05 \
--device 3 \

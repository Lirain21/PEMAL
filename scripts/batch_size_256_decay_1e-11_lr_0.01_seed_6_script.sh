#!/bin/bash
batch_size=256
seed=6
decay=1e-11
lr=0.01
device=1
python chem/finetune_reg.py \
--batch_size 256 \
--seed 6 \
--decay 1e-11 \
--lr 0.01 \
--device 1 \

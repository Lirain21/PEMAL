#!/bin/bash
batch_size=256
seed=4
decay=1e-10
lr=1e-05
device=2
python chem/finetune_reg.py \
--batch_size 256 \
--seed 4 \
--decay 1e-10 \
--lr 1e-05 \
--device 2 \

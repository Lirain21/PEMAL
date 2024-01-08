#!/bin/bash
batch_size=256
seed=4
decay=1e-09
lr=5e-06
device=1
python chem/finetune_reg.py \
--batch_size 256 \
--seed 4 \
--decay 1e-09 \
--lr 5e-06 \
--device 1 \

#!/bin/bash
batch_size=32
seed=5
decay=1e-09
lr=0.01
device=0
python chem/finetune.py \
--batch_size 32 \
--seed 5 \
--decay 1e-09 \
--lr 0.01 \
--device 0 \

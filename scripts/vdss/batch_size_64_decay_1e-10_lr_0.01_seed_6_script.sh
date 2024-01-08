#!/bin/bash
batch_size=64
seed=6
decay=1e-10
lr=0.01
device=0
python chem/finetune.py \
--batch_size 64 \
--seed 6 \
--decay 1e-10 \
--lr 0.01 \
--device 0 \

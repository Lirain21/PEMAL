#!/bin/bash
batch_size=128
seed=6
decay=1e-10
lr=0.0001
device=2
python chem/finetune.py \
--batch_size 128 \
--seed 6 \
--decay 1e-10 \
--lr 0.0001 \
--device 2 \

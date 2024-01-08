#!/bin/bash
batch_size=128
seed=5
decay=1e-10
lr=0.01
device=3
python chem/finetune.py \
--batch_size 128 \
--seed 5 \
--decay 1e-10 \
--lr 0.01 \
--device 3 \

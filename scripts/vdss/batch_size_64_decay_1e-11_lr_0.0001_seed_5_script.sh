#!/bin/bash
batch_size=64
seed=5
decay=1e-11
lr=0.0001
device=0
python chem/finetune.py \
--batch_size 64 \
--seed 5 \
--decay 1e-11 \
--lr 0.0001 \
--device 0 \

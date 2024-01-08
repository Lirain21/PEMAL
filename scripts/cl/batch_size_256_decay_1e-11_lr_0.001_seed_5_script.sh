#!/bin/bash
batch_size=256
seed=5
decay=1e-11
lr=0.001
device=1
python chem/finetune.py \
--batch_size 256 \
--seed 5 \
--decay 1e-11 \
--lr 0.001 \
--device 1 \

#!/bin/bash
batch_size=256
seed=6
decay=1e-11
lr=0.0001
device=1
python chem/finetune.py \
--batch_size 256 \
--seed 6 \
--decay 1e-11 \
--lr 0.0001 \
--device 1 \

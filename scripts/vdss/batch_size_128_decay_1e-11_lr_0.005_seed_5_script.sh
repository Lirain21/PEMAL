#!/bin/bash
batch_size=128
seed=5
decay=1e-11
lr=0.005
device=2
python chem/finetune.py \
--batch_size 128 \
--seed 5 \
--decay 1e-11 \
--lr 0.005 \
--device 2 \

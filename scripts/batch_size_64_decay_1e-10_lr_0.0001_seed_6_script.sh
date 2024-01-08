#!/bin/bash
batch_size=64
seed=6
decay=1e-10
lr=0.0001
device=2
python chem/finetune_cls.py \
--batch_size 64 \
--seed 6 \
--decay 1e-10 \
--lr 0.0001 \
--device 2 \

#!/bin/bash
batch_size=32
seed=6
decay=1e-10
lr=0.001
device=0
python chem/finetune_cls.py \
--batch_size 32 \
--seed 6 \
--decay 1e-10 \
--lr 0.001 \
--device 0 \

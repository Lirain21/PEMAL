#!/bin/bash
batch_size=32
seed=5
decay=1e-10
lr=1e-05
device=0
python chem/finetune_cls.py \
--batch_size 32 \
--seed 5 \
--decay 1e-10 \
--lr 1e-05 \
--device 0 \

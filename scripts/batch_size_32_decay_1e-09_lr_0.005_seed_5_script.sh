#!/bin/bash
batch_size=32
seed=5
decay=1e-09
lr=0.005
device=0
python chem/finetune_cls.py \
--batch_size 32 \
--seed 5 \
--decay 1e-09 \
--lr 0.005 \
--device 0 \

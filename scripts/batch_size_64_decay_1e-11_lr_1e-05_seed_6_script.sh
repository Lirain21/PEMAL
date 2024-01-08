#!/bin/bash
batch_size=64
seed=6
decay=1e-11
lr=1e-05
device=2
python chem/finetune_cls.py \
--batch_size 64 \
--seed 6 \
--decay 1e-11 \
--lr 1e-05 \
--device 2 \

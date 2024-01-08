#!/bin/bash
batch_size=64
seed=4
decay=1e-11
lr=0.0001
device=0
python chem/physical_equation_4_tasks_con.py \
--batch_size 64 \
--seed 4 \
--decay 1e-11 \
--lr 0.0001 \
--device 0 \

#!/bin/bash
batch_size=64
seed=4
decay=1e-09
lr=1e-05
device=0
python chem/physical_equation_4_tasks_con.py \
--batch_size 64 \
--seed 4 \
--decay 1e-09 \
--lr 1e-05 \
--device 0 \

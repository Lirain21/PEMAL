#!/bin/bash
batch_size=128
seed=4
decay=1e-09
lr=0.0001
device=1
python chem/physical_equation_4_tasks_con.py \
--batch_size 128 \
--seed 4 \
--decay 1e-09 \
--lr 0.0001 \
--device 1 \
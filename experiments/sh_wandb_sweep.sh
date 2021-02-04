#!/usr/bin/env bash

rm IO/error_sweep.txt
rm IO/output_sweep.txt

for lr in 1 2 3 4; do
  qsub qsub_wandb_sweep.sh
done

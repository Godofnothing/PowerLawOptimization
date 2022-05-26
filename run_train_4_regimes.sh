#!/bin/bash

N=3000
DATASET=sgemm_product
MOMENTUM=0.0
LR=1.1

python train_4_regimes_serial.py \
    \
    --dataset ${DATASET} \
    --N ${N} \
    --nn_runs 100 \
    --linearized_runs 100 \
    \
    --lr ${LR} \
    --momentum ${MOMENTUM} \
    \
    --n_steps 100 \
    --batch_size 10 100 \
    \
    --aggr_type zip \
    \
    --save_dir output/compare_4_regimes_serial/N=${N}/${DATASET}/lr=${LR}/momentum=${MOMENTUM} \

#!/bin/bash

N=10000
DATASET='mnist'
LR=1.2
MOMENTUM=0.0
BATCH_SIZE="1 10 100"

python train_nn.py \
    \
    --dataset ${DATASET} \
    \
    --N ${N} \
    --nu 1.5 \
    --kappa 3.0 \
    \
    --sched constant \
    --lr ${LR} \
    --momentum ${MOMENTUM} \
    --lr_scale 0.85 \
    \
    --n_steps 10000 \
    --batch_size ${BATCH_SIZE} \
    \
    --aggr_type prod \
    --val_frequency 100 \
    \
    --save_dir output/mlp_nn/compare_4_regimes/N=${N}/${DATASET}/lr=${LR}/batch_size=${BATCH_SIZE} \

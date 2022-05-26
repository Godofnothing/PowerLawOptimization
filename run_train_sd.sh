#!/bin/bash

N=10000
NU=1.5
KAPPA=4.0

python train_spectral_diagonal.py \
    \
    --dataset synthetic \
    \
    --N ${N} \
    --nu ${NU} \
    --kappa ${KAPPA} \
    \
    --sched constant \
    --lr # set learning rates \
    --momentum # set momentum \
    \
    --n_steps # set number of steps \
    --batch_size # set batch size \
    \
    --aggr_type prod \
    \
    --save_dir output/spectral_diagonal/loss_optimum_beta/N=${N}/synthetic/nu=${NU}/kappa=${KAPPA} \

#!/bin/bash

EXP_DIR='./output'
EXP_NAME=''
DATASET=mnist

python train_linearized.py \
    \
    --dataset ${DATASET} \
    \
    --N # set N \
    --nu # set nu \
    --kappa # set kappa \
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
    --save_dir ${EXP_DIR}/linearized/${EXP_NAME}/N=1000/${DATASET} \


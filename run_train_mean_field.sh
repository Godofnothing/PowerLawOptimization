#!/bin/bash

python train_mean_field.py \
    \
    --dataset synthetic \
    \
    --N 1000 \
    --nu 1.5 \
    --kappa 3.0 \
    \
    --sched constant \
    --lr 0.001 0.003 0.01 0.03 0.1 0.3 1.0 \
    --momentum 0.0 0.5 0.7 0.9 0.95 0.99 \
    --a 1.0 \
    --b 1.0 \
    \
    --n_steps 10000 \
    --batch_size 1 3 10 30 100 300 1000 \
    \
    --save_dir output/mean_field/sgd/constant_lr/N=1000/nu=1.5/kappa=3.0 \
    
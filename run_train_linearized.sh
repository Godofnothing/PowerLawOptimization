#!/bin/bash

python train_linearized.py \
    \
    --dataset mnist \
    \
    --N 5000 \
    --nu 1.5 \
    --kappa 3.0 \
    \
    --opt sgd \
    --sched constant \
    --lr 0.0001 0.001 0.01 0.1 \
    --momentum 0.0 0.5 0.7 0.9 0.95 \
    --a 1.0 \
    --b 1.0 \
    \
    --n_steps 10000 \
    --batch_size 1 10 100 1000 \
    \
    --save_dir output/linearized/sgd/constant_lr/N=5000/mnist \

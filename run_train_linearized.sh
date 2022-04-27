#!/bin/bash

python train_linearized.py \
    \
    --dataset synthetic \
    \
    --N 1000 \
    --nu 1.5 \
    --kappa 3.0 \
    \
    --sched constant \
    --lr 0.250 0.325 0.400 0.475 0.550 0.625 0.700 0.775 0.850 0.925 1.000 \
    --momentum -0.5 -0.4 -0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5 \
    --a 1.0 \
    --b 1.0 \
    \
    --n_steps 30000 \
    --batch_size 1 \
    \
    --aggr_type prod \
    \
    --save_dir output/linearized/negative_momenta_2d/N=1000/synthetic/nu=1.5/kappa=3.0 \

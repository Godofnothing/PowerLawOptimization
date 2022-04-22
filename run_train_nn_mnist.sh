#!/bin/bash

python train_nn_mnist.py \
    --hidden_dim 1000 \
    \
    --sched constant \
    --lr 0.0001 0.001 0.01 0.1 \
    --momentum 0.0 0.5 0.7 0.9 0.95 \
    \
    --n_steps 10000 \
    --batch_size 1 10 100 1000 \
    --train_size 1000 \
    \
    --log_frequency 100 \
    --val_frequency 100 \
    \
    --save_dir output/two_layer_nn/mnist \
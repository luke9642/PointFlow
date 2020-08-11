#! /bin/bash

python triang.py \
    --cates airplane \
    --resume_checkpoint checkpoints/ae/pretrained-airplane/checkpoint-latest.pt \
    --dims 512-512-512 \
    --latent_dims 256-256 \
    --use_deterministic_encoder \
    --use_latent_flow \
    --num_sample_shapes 20 \
    --num_sample_points 2048 \
    --data_dir ../ShapeNetCore.v2.PC15k \
    --save_triangulation \
    --method edge \
    --depth 5 \
    --samples_num 3

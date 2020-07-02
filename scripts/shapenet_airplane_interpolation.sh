#! /bin/bash
#    --resume_checkpoint checkpoints/ae/shapenet15k-airplane-new-sampling-gauss/checkpoint-80.pt \

python interpolation.py \
    --cates airplane \
    --resume_checkpoint checkpoints/ae/pretrained-airplane/checkpoint-latest.pt \
    --dims 512-512-512 \
    --latent_dims 256-256 \
    --use_deterministic_encoder \
    --use_latent_flow \
    --num_sample_shapes 20 \
    --num_sample_points 2048 \
    --data_dir ../ShapeNetCore.v2.PC15k \

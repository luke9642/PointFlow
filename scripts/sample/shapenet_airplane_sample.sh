#! /bin/bash

cate="airplane"

python sample.py \
    --cates ${cate} \
    --log_name "ae/${cate}-gauss-schedule" \
    --resume_checkpoint "checkpoints/ae/shapenet15k-${cate}-gauss-most-tiny-m-sigma-schedule/checkpoint-latest.pt" \
    --dims 512-512-512 \
    --latent_dims 256-256 \
    --use_deterministic_encoder \
    --use_latent_flow \
    --num_sample_shapes 20 \
    --num_sample_points 2048 \
    --data_dir ../ShapeNetCore.v2.PC15k \
    --batch_size 32
#    --sigma 0.1 \
#    --m 0. \
#    --overwrite

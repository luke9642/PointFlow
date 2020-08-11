#! /bin/bash

cate='chair'

python triang.py \
    --cates ${cate} \
    --log_name "ae/${cate}-gauss-schedule" \
    --resume_checkpoint "checkpoints/ae/shapenet15k-${cate}-gauss-schedule/checkpoint-3139.pt" \
    --dims 512-512-512 \
    --latent_dims 256-256 \
    --use_deterministic_encoder \
    --use_latent_flow \
    --num_sample_shapes 20 \
    --num_sample_points 2048 \
    --data_dir ../ShapeNetCore.v2.PC15k \
    --batch_size 32 \
    --depth 4 \
    --samples_num 32

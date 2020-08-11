#! /bin/bash

cate="airplane"
dims="512-512-512"
latent_dims="256-256"
num_blocks=1
latent_num_blocks=1
zdim=128
batch_size=16
lr=2e-3
epochs=4000
ds=shapenet15k
log_name="ae/${ds}-${cate}-gauss-schedule"
data_dir="../ShapeNetCore.v2.PC15k"

python -u train.py \
    --log_name ${log_name} \
    --lr ${lr} \
    --dataset_type ${ds} \
    --data_dir ${data_dir} \
    --cates ${cate} \
    --dims ${dims} \
    --latent_dims ${latent_dims} \
    --num_blocks ${num_blocks} \
    --latent_num_blocks ${latent_num_blocks} \
    --batch_size ${batch_size} \
    --zdim ${zdim} \
    --epochs ${epochs} \
    --save_freq 10 \
    --viz_freq 5 \
    --log_freq 1 \
    --val_freq 1 \
    --use_deterministic_encoder \
    --prior_weight 0 \
    --entropy_weight 0 \
    --use_latent_flow \
    --decrease_m_sigma \
    --decrease_m_sigma_size 500 \
    --decrease_m_sigma_epochs_interval 5 \
    --decrease_m_sigma_initial_epoch 50
#    --overwrite


echo "Done"
exit 0

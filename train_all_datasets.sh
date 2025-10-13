#!/bin/bash

# GraphSSM TimeSeriesForecaster Training Script
# This script trains GraphSSM models for ETTm1, ETTh1, and Solar datasets

export CUDA_VISIBLE_DEVICES=0

echo "Starting GraphSSM TimeSeriesForecaster Training"
echo "=============================================="

# Change to workspace directory
cd /workspace

# Run training for all datasets
python train_graphssm.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id GraphSSM \
    --model GraphSSM \
    --train_epochs 200 \
    --patience 3 \
    --itr 1 \
    --des 'GraphSSM_Exp' \
    --loss MSE \
    --lradj cosine \
    --use_amp \
    --use_gpu True \
    --gpu 0 \
    --d_state 16 \
    --d_conv 4 \
    --expand 2 \
    --prune_ratio 0.15 \
    --optimizer adamw \
    --weight_decay 0.05 \
    --learning_rate 0.001 \
    --seq_len 96 \
    --pred_len 96

echo "Training completed!"
echo "Checkpoints saved in: ./checkpoints/"
echo "Test results saved in: ./test_results/"


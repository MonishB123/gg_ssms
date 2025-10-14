#!/bin/bash

# GraphSSM TimeSeriesForecaster Training Script
# This script trains GraphSSM models for ETTm1, ETTh1, and Solar datasets

export CUDA_VISIBLE_DEVICES=0

echo "🚀 Starting GraphSSM TimeSeriesForecaster Training"
echo "=================================================="
echo "📊 Training Configuration:"
echo "   • Lookback Length: 96"
echo "   • Forecast Length: 96" 
echo "   • Training Epochs: 5"
echo "   • Learning Rate: 0.001"
echo "   • Optimizer: AdamW"
echo "   • Pruning Ratio: 15%"
echo "=================================================="

# Change to workspace directory
cd /workspace

# Record start time
START_TIME=$(date +%s)

# Run training for all datasets with progress tracking
echo "🎯 Starting training process..."
python train_graphssm.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id GraphSSM \
    --model GraphSSM \
    --train_epochs 5 \
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
    --prune_ratio 0.0 \
    --optimizer adamw \
    --weight_decay 0.05 \
    --learning_rate 0.001 \
    --seq_len 96 \
    --pred_len 96 \
    --verbose

# Calculate total training time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "✅ Training completed successfully!"
echo "=================================================="
echo "📈 Training Summary:"
echo "   • Total Time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "   • Checkpoints: ./checkpoints/"
echo "   • Test Results: ./test_results/"
echo "=================================================="


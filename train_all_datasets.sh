#!/bin/bash

# GraphSSM TimeSeriesForecaster Training Script
# This script trains GraphSSM models for ETTm1, ETTh1, and Solar datasets
# Total: 9 models (3 datasets × 3 prediction lengths)

export CUDA_VISIBLE_DEVICES=0

# Define prediction lengths and datasets
PRED_LENGTHS=(96 192 336)
DATASETS=("ETTm1" "ETTh1" "Solar")

echo "🚀 Starting GraphSSM TimeSeriesForecaster Training"
echo "=================================================="
echo "📊 Training Configuration:"
echo "   • Datasets: ${DATASETS[*]}"
echo "   • Prediction Lengths: ${PRED_LENGTHS[*]}"
echo "   • Total Models: $((${#DATASETS[@]} * ${#PRED_LENGTHS[@]}))"
echo "   • Lookback Length: 96"
echo "   • Training Epochs: 5"
echo "   • Learning Rate: 0.001"
echo "   • Optimizer: AdamW"
echo "   • Pruning Ratio: 0%"
echo ""
echo "📝 File Naming Pattern:"
echo "   Format: {task}_{model_id}_{model}_{data}_{features}_{seq_len}_{label_len}_{pred_len}_{d_model}_{prune_ratio}_{description}"
echo "   Example: long_term_forecast_GraphSSM_GraphSSM_ETTm1_M_96_48_96_512_0.0_GraphSSM_Exp"
echo "=================================================="

# Change to workspace directory
cd /workspace

# Record overall start time
OVERALL_START_TIME=$(date +%s)
TOTAL_MODELS_TRAINED=0

# Train models for each prediction length
for pred_len in "${PRED_LENGTHS[@]}"; do
    echo ""
    echo "🎯 Training models for prediction length: ${pred_len}"
    echo "=================================================="
    
    # Record start time for this prediction length
    START_TIME=$(date +%s)
    
    # Run training for all datasets with this prediction length
    echo "📈 Training ETTm1, ETTh1, and Solar datasets..."
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
        --pred_len ${pred_len} \
        --verbose
    
    # Calculate training time for this prediction length
    END_TIME=$(date +%s)
    TOTAL_TIME=$((END_TIME - START_TIME))
    HOURS=$((TOTAL_TIME / 3600))
    MINUTES=$(((TOTAL_TIME % 3600) / 60))
    SECONDS=$((TOTAL_TIME % 60))
    
    # Update total models trained
    TOTAL_MODELS_TRAINED=$((TOTAL_MODELS_TRAINED + ${#DATASETS[@]}))
    
    echo ""
    echo "✅ Training completed for prediction length ${pred_len}!"
    echo "=================================================="
    echo "📈 Training Summary:"
    echo "   • Models Trained: ${#DATASETS[@]} (ETTm1, ETTh1, Solar)"
    echo "   • Prediction Length: ${pred_len}"
    echo "   • Time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo "   • Total Models Completed: ${TOTAL_MODELS_TRAINED}/9"
    echo "   • Checkpoints: ./checkpoints/"
    echo "   • Test Results: ./test_results/"
    echo "=================================================="
done

# Calculate overall training time
OVERALL_END_TIME=$(date +%s)
OVERALL_TOTAL_TIME=$((OVERALL_END_TIME - OVERALL_START_TIME))
OVERALL_HOURS=$((OVERALL_TOTAL_TIME / 3600))
OVERALL_MINUTES=$(((OVERALL_TOTAL_TIME % 3600) / 60))
OVERALL_SECONDS=$((OVERALL_TOTAL_TIME % 60))

echo ""
echo "🎉 ALL TRAINING COMPLETED SUCCESSFULLY!"
echo "=================================================="
echo "📊 Final Training Summary:"
echo "   • Total Models Trained: ${TOTAL_MODELS_TRAINED}"
echo "   • Datasets: ${DATASETS[*]}"
echo "   • Prediction Lengths: ${PRED_LENGTHS[*]}"
echo "   • Total Time: ${OVERALL_HOURS}h ${OVERALL_MINUTES}m ${OVERALL_SECONDS}s"
echo "   • Checkpoints: ./checkpoints/"
echo "   • Test Results: ./test_results/"
echo "=================================================="
echo "📁 Model Checkpoints Created:"
echo "   • ETTm1 models: long_term_forecast_GraphSSM_GraphSSM_ETTm1_M_96_48_[96|192|336]_512_0.0_GraphSSM_Exp"
echo "   • ETTh1 models: long_term_forecast_GraphSSM_GraphSSM_ETTh1_M_96_48_[96|192|336]_512_0.0_GraphSSM_Exp"  
echo "   • Solar models: long_term_forecast_GraphSSM_GraphSSM_Solar_M_96_48_[96|192|336]_32_0.0_GraphSSM_Exp"
echo "   • Checkpoint files: ./checkpoints/[setting_name]/checkpoint.pth"
echo "=================================================="
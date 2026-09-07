# ML worker commands

Common commands for development, testing, and linting the ML worker codebase.

## Virtual environment and dependencies

Per the repository standard, all worker tasks use `uv` and the root virtual environment (`../.venv`):

```bash
# Bootstrap once, then install or update dependencies from repository root
uv venv --python 3.13 .venv
uv pip install -r worker/requirements.txt --python ./.venv/bin/python
```

## Running the worker

From the `worker/` directory:

```bash
# Run the worker with health server and task listener
../.venv/bin/python app.py
```

## Testing

From the `worker/` directory:

```bash
# Run test suite
../.venv/bin/python -m pytest -q

# Run tests with coverage
../.venv/bin/python -m pytest --cov=. --cov-report=xml --cov-report=html
```

HTML coverage reports are generated at `htmlcov/index.html`.

## Linting and type-checking

From the `worker/` directory:

```bash
# Auto-fix lint issues and format
../.venv/bin/python -m ruff check --fix . && ../.venv/bin/python -m ruff format .

# Check lint without modifying
../.venv/bin/python -m ruff check .

# Check formatting without modifying
../.venv/bin/python -m ruff format --check .

# Type-check with Pyright
../.venv/bin/python -m pyright .
```

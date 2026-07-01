# ML Worker Commands Guide

This document lists the common commands used for development, testing, formatting, and linting the ML Worker (Python) codebase.

We use **pytest** for testing and **Ruff** (recommended) for linting and code formatting, or alternatively a combination of **Black** (formatting) and **Flake8** (linting).

## Setup & Virtual Environment

You should use the unified virtual environment located at the root of the project (`.venv`) for consistency across components.

```bash
# Activate the root virtual environment
source ../.venv/bin/activate
```

## Running the Worker

```bash
# Run the Python worker locally (requires Redis and other services)
python -m worker.main
```

## Testing & Coverage

```bash
# Run all unit tests
pytest

# Run tests and generate code coverage report (including HTML report)
pytest --cov=. --cov-report=xml --cov-report=html
```
*The HTML coverage report will be generated at `htmlcov/index.html`.*

## Linting & Formatting

[Ruff](https://github.com/astral-sh/ruff) is an extremely fast Python linter and formatter that replaces Flake8, Black, isort, and more. Run these commands from the `unified-workers/` directory:

```bash
# 1. Run lint checks
ruff check .

# 2. Run lint checks and auto-fix safe issues
ruff check . --fix

# 3. Format the code
ruff format .
```

### Alternative Tools: Black & Flake8

If you prefer standard tools, you can use **Black** for formatting and **Flake8** for linting.

```bash
# 1. Format code with Black
black .

# 2. Check style guidelines with Flake8
flake8 .
```

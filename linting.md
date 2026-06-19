# Linting & Formatting (ML Worker)

The ML Worker codebase is written in Python. We use **Ruff** (recommended) for linting and code formatting, or alternatively a combination of **Black** (formatting) and **Flake8** (linting).

## Recommended Tool: Ruff

[Ruff](https://github.com/astral-sh/ruff) is an extremely fast Python linter and formatter that replaces Flake8, Black, isort, and more.

Run these commands from the `unified-workers/` directory:

```bash
# Activate your virtual environment first
source ../.venv/bin/activate  # or wherever your venv is located

# 1. Run lint checks
ruff check .

# 2. Run lint checks and auto-fix safe issues
ruff check . --fix

# 3. Format the code
ruff format .
```

---

## Alternative Tools: Black & Flake8

If you prefer standard tools, you can use **Black** for formatting and **Flake8** for linting.

```bash
# 1. Format code with Black
black .

# 2. Check style guidelines with Flake8
flake8 .
```


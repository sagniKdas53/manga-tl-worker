<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **manga-tl-worker** (1307 symbols, 2505 relationships, 98 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/manga-tl-worker/context` | Codebase overview, check index freshness |
| `gitnexus://repo/manga-tl-worker/clusters` | All functional areas |
| `gitnexus://repo/manga-tl-worker/processes` | All execution flows |
| `gitnexus://repo/manga-tl-worker/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
## This repo is a submodule

`manga-tl-worker` is a git submodule of **manga-tl** (`manga-library`), mounted at `worker/`. It is
indexed **separately and on purpose**: the parent's `detect_changes()` runs `git diff` in the parent,
which sees this repo only as a pointer and reports `changed_count: 0` for any change made here.

- **For changes in this repo, run `detect_changes({repo: "manga-tl-worker"})`** — not the parent's.
- `impact()` resolves worker symbols from either index, but only this one carries the worker's
  execution flows.
- Commit here first and **push before** bumping the parent's pointer, or the parent references a
  commit nobody can fetch.

## Gates

Four, all required. Source lives at `src/worker/`, not the repo root (`app.py` is the exception).

```bash
cd worker
../.venv/bin/python -m ruff check --fix . && ../.venv/bin/python -m ruff format .   # before the final run
../.venv/bin/python -m ruff check .
../.venv/bin/python -m ruff format --check .
../.venv/bin/python -m pyright .
../.venv/bin/python -m pytest -q
```

The `.venv` is at the **parent** repo root (Python 3.13.12), not in here. Baseline: **315 passed**.

Tests load `tests/test_providers.json` via `conftest.py`'s `PROVIDERS_CONFIG`, **not**
`../config/providers.json` — check the fixture carries a key before assuming behaviour is broken.

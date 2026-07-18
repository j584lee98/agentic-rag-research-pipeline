# a-rag-research-pipeline

## Pre-commit hook (Ruff)

This repo uses `pre-commit` to run Ruff automatically before each commit.

### One-time setup

```bash
uv sync --group dev
uv run pre-commit install
```

### Run manually (optional)

```bash
uv run pre-commit run --all-files
```
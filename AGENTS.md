# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this is

An MCP server giving MCP-compatible AI agents (Claude Code, Cursor,
Windsurf, and others) persistent, structured, queryable memory across
sessions, backed by the [YantrikDB](https://github.com/yantrikos/yantrikdb)
Rust engine. MIT license; the engine dependency is Apache-2.0.

## Setup

Use Python 3.10 through 3.13 (the `yantrikdb` engine ships native wheels
for those versions; 3.14 may not install yet).

```bash
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -e ".[dev]"
```

This installs the server, its runtime dependencies, and pytest.

## Run tests

```bash
pytest
```

Only `tests/` is collected by default (see `pyproject.toml`). `test_v14.py`
at the repo root is a heavier integration check against the engine — run
it explicitly with `pip install -e ".[torch]"` first.

## Run the server locally

```bash
yantrikdb-mcp --help
yantrikdb-mcp --version
```

Stdio mode (the default, used by Cursor/Claude Code) needs no extra flags.

## Before opening a PR

1. Branch from `main`.
2. If you add behavior, add or extend a test in `tests/`. Doc-only changes
   don't need one.
3. Open the PR against
   [yantrikos/yantrikdb-mcp](https://github.com/yantrikos/yantrikdb-mcp)
   with a short description of the change.

Full contributor guide: [CONTRIBUTING.md](CONTRIBUTING.md).

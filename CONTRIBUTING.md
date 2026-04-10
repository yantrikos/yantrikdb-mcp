# Contributing

Thanks for helping improve YantrikDB MCP. Maintainer-tracked tasks may appear on [GitHub Issues](https://github.com/yantrikos/yantrikdb-mcp/issues); small PRs for tests, docs, or developer ergonomics are welcome anytime.

## Environment (recommended: new venv)

Use **Python 3.10 through 3.13**. The `yantrikdb` engine ships native wheels for those versions; **3.14** may fail to install until the engine’s PyO3 stack supports it.

From the repository root:

```bash
python3.12 -m venv .venv   # or python3.11 / python3.13 — avoid 3.14 for now
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -e ".[dev]"
```

This installs the MCP server and its runtime dependencies, plus **pytest** for the suite under `tests/`.

Optional: `pip install -e ".[dev,torch]"` if you work with SentenceTransformers-based flows (see root `test_v14.py`).

## Run tests

```bash
pytest
```

By default only `tests/` is collected (see `pyproject.toml`). The script `test_v14.py` at the repo root is a heavier integration check against the engine; run it explicitly if needed:

```bash
pip install -e ".[torch]"
pytest test_v14.py -v
```

## Run the server locally

```bash
yantrikdb-mcp --help
yantrikdb-mcp --version
```

Stdio mode (typical for Cursor / Claude Code) uses no extra flags. For SSE with auth:

```bash
export YANTRIKDB_API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
yantrikdb-mcp --transport sse --port 8420
```

## Submitting a PR

1. Fork [yantrikos/yantrikdb-mcp](https://github.com/yantrikos/yantrikdb-mcp) on GitHub.
2. Create a branch from `main`, push your fork, open a PR with a short description of the change.
3. If you add behavior, add or extend tests in `tests/`. Doc-only changes are fine without tests.

## License

This package is MIT. The `yantrikdb` engine dependency is AGPL-3.0; see the [README](README.md) for a short explanation of how that applies.

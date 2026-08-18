# syntax=docker/dockerfile:1

# Build stage: resolve and install into a self-contained virtualenv.
# Kept separate from the runtime so pip, build wheels and caches never
# reach the shipped image.
FROM python:3.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /src

# Copy only what the build backend reads, so a source-only edit does not
# invalidate the dependency layer.
COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install .

# Runtime stage.
FROM python:3.12-slim-bookworm

# The engine ships manylinux_2_17 wheels for x86_64 and aarch64; no
# compiler or system library is needed at runtime.
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    YANTRIKDB_DB_PATH=/mcp/memory.db

# The SQLite store lives here. Mount a volume to keep memories across
# container restarts; without one they are lost with the container.
RUN mkdir -p /mcp && useradd --create-home --uid 10001 yantrik && chown yantrik:yantrik /mcp
VOLUME ["/mcp"]

USER yantrik

# stdio transport: the MCP client talks over stdin/stdout.
ENTRYPOINT ["yantrikdb-mcp"]

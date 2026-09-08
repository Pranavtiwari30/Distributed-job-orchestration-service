# syntax=docker/dockerfile:1

# ---- build stage ------------------------------------------------------------
# Dependencies are compiled into a virtualenv here, and only the finished
# virtualenv is copied forward. The build toolchain -- compilers, headers, uv
# itself -- never reaches the runtime image, which keeps it small and shrinks
# the attack surface to code that is actually executed.
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Copy only the manifest first: this layer is cached and rebuilt only when
# dependencies change, not on every source edit.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN uv venv /opt/venv && VIRTUAL_ENV=/opt/venv uv pip install --no-cache .

# ---- runtime stage ----------------------------------------------------------
FROM python:3.11-slim AS runtime

# A non-root user: a container process that does not need root should not have
# it, and several Kubernetes admission policies reject images that run as root.
RUN groupadd --system --gid 1001 conductor \
 && useradd --system --uid 1001 --gid conductor --create-home conductor

COPY --from=builder --chown=conductor:conductor /opt/venv /opt/venv
WORKDIR /app
COPY --chown=conductor:conductor migrations/ ./migrations/
COPY --chown=conductor:conductor alembic.ini ./

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER conductor
EXPOSE 8000

# Talks to the app rather than merely checking the process exists: a hung
# uvicorn is still a running process.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/healthz',timeout=2).status_code==200 else 1)"

# Overridden with `conductor-worker` for worker replicas: one image, two roles,
# so the code running in both is provably identical.
CMD ["conductor-api"]

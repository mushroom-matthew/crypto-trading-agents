# syntax=docker/dockerfile:1.7

# ---- Base image with Python 3.11 ----
FROM python:3.11-slim AS base

# System deps (ssl, tz, build tools if needed for wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates tzdata build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ---- Install uv (fast Python package manager) ----
# Using the official installer script (static binary)
RUN curl -fsSL https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

# ---- Copy only dependency files first for layer caching ----
# If the project has a pyproject + uv.lock, this will cache nicely.
COPY pyproject.toml uv.lock* ./

# Create a minimal venv managed by uv and sync deps
# (If the repo uses extras like [worker], uv handles those via lock)
RUN uv sync --frozen --no-dev

# ---- Now copy source code ----
COPY . .

# Default port for the MCP/FastAPI server
EXPOSE 8080

# We won't set CMD here so docker-compose can run either server or worker.

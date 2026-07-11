# Shiny for Python app served with uv on Python 3.11
# (matches the Python runtime Posit Connect Cloud provisions).
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /electric_generation_planning

# Install dependencies first (cached layer) from the locked environment.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy the application code.
COPY app.py ./
COPY ./src ./src

EXPOSE 8501

# app.py is at the repo root; serve on all interfaces for the container.
ENTRYPOINT ["uv", "run", "--no-dev", "shiny", "run", "--host", "0.0.0.0", "--port", "8501", "app.py"]

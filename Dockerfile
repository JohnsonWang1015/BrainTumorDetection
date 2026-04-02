FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    git \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /workspace
USER appuser

ENV PATH="/workspace/.venv/bin:${PATH}"

CMD ["bash"]

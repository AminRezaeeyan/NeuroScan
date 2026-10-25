# syntax=docker/dockerfile:1.7
# NeuroScan prediction API and demo page.
# The image holds code and dependencies only; the model is loaded from the MLflow
# registry at startup (see docker-compose.yml), so releasing a model needs no rebuild.
#
# Packages come from a uv cache passed as the build context "uvcache". docker compose
# passes the host's ~/.cache/uv and builds offline (nothing is downloaded); elsewhere:
#   docker build --build-context uvcache=<empty dir> --build-arg UV_OFFLINE=0 .

FROM ghcr.io/astral-sh/uv:0.12.3 AS uv

FROM python:3.12-slim AS build
COPY --from=uv /uv /bin/uv
ARG UV_OFFLINE=1
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_CACHE_DIR=/uv-cache \
    UV_OFFLINE=$UV_OFFLINE \
    UV_HTTP_TIMEOUT=300
WORKDIR /src
# Dependencies first, so code changes don't reinstall TensorFlow.
COPY pyproject.toml uv.lock ./
RUN --mount=type=bind,from=uvcache,target=/uv-cache,rw \
    uv sync --frozen --no-dev --extra serving --extra tracking --no-install-project
COPY README.md LICENSE ./
COPY src ./src
RUN --mount=type=bind,from=uvcache,target=/uv-cache,rw \
    uv sync --frozen --no-dev --extra serving --extra tracking --no-editable

FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 10001 neuroscan
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MPLCONFIGDIR=/tmp/matplotlib \
    KERAS_HOME=/tmp/keras \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    WORKERS=1
USER neuroscan
WORKDIR /home/neuroscan
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4)"]
# Each worker loads its own copy of the model (~0.5 GB with TensorFlow).
CMD ["sh", "-c", "exec uvicorn neuroscan.serving.api:app --host 0.0.0.0 --port 8000 --workers \"$WORKERS\" --no-server-header"]

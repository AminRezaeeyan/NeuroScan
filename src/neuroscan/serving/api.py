"""Prediction API and the demo page.

    uvicorn neuroscan.serving.api:app --port 8000

Loads the packaged model (network + temperature + OOD detector) from the MLflow
registry, by default ``models:/neuroscan-classifier@champion``: moving the alias is
how a model is released or rolled back. Configuration comes from the environment:

- ``NEUROSCAN_MODEL_URI``: model to serve (default: the champion alias);
- ``MLFLOW_TRACKING_URI``: where the registry lives;
- ``NEUROSCAN_MAX_UPLOAD_MB``: upload limit (default 10);
- ``NEUROSCAN_PREDICTION_LOG``: optional JSON-lines file of prediction outcomes.

``/metrics`` exposes prediction counts, latency and confidence in Prometheus format.

Uploads are processed in memory and never stored; logs record the outcome
(status, class, confidence, latency, model version), never the image.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from neuroscan.serving.checks import check_bytes
from neuroscan.serving.monitoring import Metrics
from neuroscan.serving.schemas import BatchPrediction, Error, ModelInfo, Prediction

log = logging.getLogger("neuroscan.api")

DEFAULT_MODEL_URI = "models:/neuroscan-classifier@champion"
DOCS_PATHS = ("/docs", "/redoc", "/openapi.json")  # Swagger/ReDoc load their assets from a CDN
STATIC_DIR = Path(__file__).parent / "static"
MAX_BATCH = 16
INFO_METRICS = {
    "test_macro_f1": "test.grouped.macro_f1",
    "test_tumor_sensitivity": "test.grouped.tumor_sensitivity",
    "test_accuracy_when_answered": "ood.test.accuracy_when_answered",
    "test_uncertain_rate": "ood.test.uncertain",
    "ood_auroc": "ood.all.mahalanobis.auroc",
}
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; "
        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cache-Control": "no-store",
}


@dataclass
class ServedModel:
    """What the API needs from a packaged model; ``predictor`` is a NeuroScanModel."""

    predictor: Any
    uri: str
    version: str
    metrics: dict[str, float] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def config(self) -> dict:
        return dict(self.predictor.config)

    @property
    def image_size(self) -> int:
        return int(self.config["image_size"])

    @property
    def ood_threshold(self) -> float | None:
        det = getattr(self.predictor, "detector", None)
        return None if det is None else float(det.threshold)


def load_from_registry(uri: str) -> ServedModel:
    import mlflow

    from neuroscan.tracking import get_client

    client = get_client()
    # Copy the model out of the registry first: loading in place writes a metadata file
    # next to the artifacts, which fails on a read-only registry (as in Docker).
    local = mlflow.artifacts.download_artifacts(
        artifact_uri=uri, dst_path=tempfile.mkdtemp(prefix="neuroscan-model-")
    )
    pyfunc = mlflow.pyfunc.load_model(local)
    predictor = pyfunc.unwrap_python_model()
    version, run_id = "unknown", None
    if uri.startswith("models:/"):
        name, _, ref = uri.removeprefix("models:/").partition("@")
        if ref:
            mv = client.get_model_version_by_alias(name, ref)
        else:
            name, _, number = uri.removeprefix("models:/").partition("/")
            mv = client.get_model_version(name, number)
        version, run_id = str(mv.version), mv.run_id
    metrics: dict[str, float] = {}
    if run_id:
        run_metrics = client.get_run(run_id).data.metrics
        metrics = {k: run_metrics[m] for k, m in INFO_METRICS.items() if m in run_metrics}
    return ServedModel(predictor, uri, version, metrics)


def png_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def heatmap_png(cam: np.ndarray) -> str:
    """Grad-CAM as an RGBA PNG: color from the 'turbo' map, opacity grows with the value."""
    from matplotlib import colormaps

    rgba = colormaps["turbo"](cam)
    rgba[..., 3] = np.clip(cam, 0, 1) ** 1.5 * 0.9
    return png_base64(Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA"))


def predict_images(
    served: ServedModel, images: np.ndarray, with_heatmap: bool
) -> list[dict[str, Any]]:
    """Run the packaged model (and Grad-CAM for answered images) on preprocessed images."""
    with served.lock:
        out = served.predictor.predict(None, images)
        answered = out["status"].isin(["ok", "uncertain"]).to_numpy()
        cams = None
        if with_heatmap and answered.any():
            from neuroscan.explain import gradcam

            cams, _ = gradcam(served.predictor.model, images[answered])
    names = served.config["class_names"]
    results, k = [], 0
    for i, (_, row) in enumerate(out.iterrows()):
        rejected = row["status"] in ("rejected_ood", "rejected_input")
        item: dict[str, Any] = {
            "status": row["status"],
            "ood_score": None if np.isnan(row["ood_score"]) else float(row["ood_score"]),
            "input_png_base64": png_base64(
                Image.fromarray(np.clip(images[i], 0, 255).astype(np.uint8), mode="RGB")
            ),
        }
        if row["status"] == "rejected_input":
            item["reason"] = "not_grayscale"
        elif row["status"] == "rejected_ood":
            item["reason"] = "out_of_distribution"
        if not rejected:
            item["prediction"] = row["prediction"]
            item["confidence"] = float(row["confidence"])
            item["probabilities"] = {n: float(row[n]) for n in names}
            if cams is not None:
                item["heatmap_png_base64"] = heatmap_png(cams[k])
                k += 1
        results.append(item)
    return results


async def read_upload(file: UploadFile, max_bytes: int) -> bytes:
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, detail="file_too_large")
    return data


def create_app(loader: Callable[[str], ServedModel] = load_from_registry) -> FastAPI:
    model_uri = os.environ.get("NEUROSCAN_MODEL_URI", DEFAULT_MODEL_URI)
    max_bytes = int(float(os.environ.get("NEUROSCAN_MAX_UPLOAD_MB", "10")) * 1024 * 1024)
    state: dict[str, ServedModel | None] = {"model": None}
    metrics = Metrics(os.environ.get("NEUROSCAN_PREDICTION_LOG") or None)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            model = loader(model_uri)
            state["model"] = model
            log.info("Serving %s (version %s)", model_uri, model.version)
        except Exception:
            log.exception("Could not load %s; /ready will report 503", model_uri)
        yield

    app = FastAPI(
        title="NeuroScan API",
        description="Brain tumor MRI classification. Research demo, not a medical device.",
        version="2.0.0",
        lifespan=lifespan,
    )

    def served() -> ServedModel:
        model = state["model"]
        if model is None:
            raise HTTPException(503, detail="model_not_loaded")
        return model

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        is_docs = request.url.path.startswith(DOCS_PATHS)
        for key, value in SECURITY_HEADERS.items():
            if not (is_docs and key == "Content-Security-Policy"):
                response.headers.setdefault(key, value)
        return response

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(Error(error=str(exc.detail)).model_dump(), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def internal_error(_: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled error")  # details in the logs only
        return JSONResponse(Error(error="internal_error").model_dump(), status_code=500)

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> PlainTextResponse:
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", responses={503: {"model": Error}})
    def ready() -> dict[str, str]:
        return {"status": "ready", "model_version": served().version}

    @app.get("/v1/info", response_model=ModelInfo)
    def info() -> ModelInfo:
        model = served()
        cfg = model.config
        return ModelInfo(
            model_uri=model.uri,
            model_version=model.version,
            classes=list(cfg["class_names"]),
            temperature=cfg.get("temperature"),
            uncertain_threshold=cfg.get("uncertain_threshold"),
            ood_threshold=model.ood_threshold,
            metrics=model.metrics,
        )

    @app.post(
        "/v1/predict",
        response_model=Prediction,
        responses={400: {"model": Error}, 413: {"model": Error}, 503: {"model": Error}},
    )
    async def predict(
        file: Annotated[UploadFile, File()],
        heatmap: Annotated[bool, Query()] = True,
    ) -> Prediction:
        model = served()
        start = time.perf_counter()
        data = await read_upload(file, max_bytes)
        problem = check_bytes(data)
        if problem in ("not_an_image", "bad_size"):
            raise HTTPException(400, detail=problem)
        from neuroscan.data.preprocess import preprocess_bytes

        # CPU-bound work runs in a worker thread so the event loop keeps serving
        # (health checks, other requests) during inference.
        image = (await run_in_threadpool(preprocess_bytes, data, model.image_size))[None]
        result = (await run_in_threadpool(predict_images, model, image, heatmap))[0]
        latency = (time.perf_counter() - start) * 1000
        metrics.record(model.version, result, latency)
        log.info(
            "predict status=%s class=%s confidence=%s version=%s latency_ms=%.0f",
            result["status"],
            result.get("prediction"),
            None if result.get("confidence") is None else round(result["confidence"], 3),
            model.version,
            latency,
        )
        return Prediction(
            model_version=model.version,
            ood_threshold=model.ood_threshold,
            latency_ms=latency,
            **result,
        )

    @app.post(
        "/v1/predict/batch",
        response_model=BatchPrediction,
        responses={400: {"model": Error}, 413: {"model": Error}, 503: {"model": Error}},
    )
    async def predict_batch(files: Annotated[list[UploadFile], File()]) -> BatchPrediction:
        model = served()
        if len(files) > MAX_BATCH:
            raise HTTPException(400, detail="too_many_files")
        from neuroscan.data.preprocess import preprocess_bytes

        start = time.perf_counter()
        payloads = [await read_upload(f, max_bytes) for f in files]
        for data in payloads:
            if check_bytes(data) in ("not_an_image", "bad_size"):
                raise HTTPException(400, detail="not_an_image")
        images = await run_in_threadpool(
            lambda: np.stack([preprocess_bytes(d, model.image_size) for d in payloads])
        )
        results = await run_in_threadpool(predict_images, model, images, False)
        latency = (time.perf_counter() - start) * 1000
        for r in results:
            metrics.record(model.version, r, latency / len(results))
        for r in results:
            r.pop("input_png_base64", None)
        return BatchPrediction(
            model_version=model.version,
            results=[
                Prediction(
                    model_version=model.version,
                    ood_threshold=model.ood_threshold,
                    latency_ms=latency,
                    **r,
                )
                for r in results
            ],
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s"
)
app = create_app()

"""Prediction monitoring: Prometheus-style counters and an optional outcome log.

Only outcomes are recorded (status, class, confidence, latency, model version),
never images or file names.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

LATENCY_BUCKETS_MS = (50, 100, 250, 500, 1000, 2500, 5000)
CONFIDENCE_BUCKETS = (0.5, 0.6, 0.7, 0.8, 0.86, 0.9, 0.95, 0.99)


class Metrics:
    def __init__(self, log_path: str | Path | None = None) -> None:
        self.lock = threading.Lock()
        self.predictions: Counter[tuple[str, str, str]] = Counter()
        self.latency = Histogram(LATENCY_BUCKETS_MS)
        self.confidence = Histogram(CONFIDENCE_BUCKETS)
        self.log_path = Path(log_path) if log_path else None

    def record(self, version: str, result: dict[str, Any], latency_ms: float) -> None:
        status, prediction = result["status"], result.get("prediction") or "none"
        confidence = result.get("confidence")
        with self.lock:
            self.predictions[(version, status, prediction)] += 1
            self.latency.observe(latency_ms)
            if confidence is not None:
                self.confidence.observe(confidence)
            if self.log_path is not None:
                entry = {
                    "time": round(time.time(), 3),
                    "model_version": version,
                    "status": status,
                    "prediction": result.get("prediction"),
                    "confidence": None if confidence is None else round(confidence, 4),
                    "ood_score": result.get("ood_score"),
                    "latency_ms": round(latency_ms, 1),
                }
                with self.log_path.open("a") as f:
                    f.write(json.dumps(entry) + "\n")

    def render(self) -> str:
        """Prometheus text exposition format."""
        lines = [
            "# HELP neuroscan_predictions_total Predictions by model version, status and class.",
            "# TYPE neuroscan_predictions_total counter",
        ]
        with self.lock:
            for (version, status, prediction), n in sorted(self.predictions.items()):
                labels = f'model_version="{version}",status="{status}",prediction="{prediction}"'
                lines.append(f"neuroscan_predictions_total{{{labels}}} {n}")
            lines += self.latency.render(
                "neuroscan_request_latency_ms", "End-to-end /v1/predict latency in ms."
            )
            lines += self.confidence.render(
                "neuroscan_confidence", "Calibrated confidence of answered predictions."
            )
        return "\n".join(lines) + "\n"


class Histogram:
    def __init__(self, buckets: tuple[float, ...]) -> None:
        self.buckets = buckets
        self.counts = [0] * len(buckets)
        self.total = 0
        self.sum = 0.0

    def observe(self, value: float) -> None:
        self.total += 1
        self.sum += value
        for i, upper in enumerate(self.buckets):
            if value <= upper:
                self.counts[i] += 1

    def render(self, name: str, help_text: str) -> list[str]:
        lines = [f"# HELP {name} {help_text}", f"# TYPE {name} histogram"]
        for upper, n in zip(self.buckets, self.counts, strict=True):
            lines.append(f'{name}_bucket{{le="{upper}"}} {n}')
        lines += [
            f'{name}_bucket{{le="+Inf"}} {self.total}',
            f"{name}_sum {self.sum:.3f}",
            f"{name}_count {self.total}",
        ]
        return lines

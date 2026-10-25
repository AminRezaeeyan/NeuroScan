"""Monitoring: metrics rendering and the dashboard figure."""

import pandas as pd

from neuroscan.reports.monitoring import dashboard
from neuroscan.serving.monitoring import Metrics


def test_metrics_render_prometheus_text():
    m = Metrics()
    m.record("7", {"status": "ok", "prediction": "glioma", "confidence": 0.97}, 120.0)
    m.record("7", {"status": "rejected_ood", "prediction": None}, 300.0)
    text = m.render()
    assert (
        'neuroscan_predictions_total{model_version="7",status="ok",prediction="glioma"} 1' in text
    )
    assert 'neuroscan_request_latency_ms_bucket{le="250"} 1' in text
    assert "neuroscan_confidence_count 1" in text


def test_dashboard_figure():
    log = pd.DataFrame(
        {
            "time": range(40),
            "model_version": [7] * 40,
            "status": ["ok"] * 30 + ["uncertain"] * 5 + ["rejected_ood"] * 5,
            "confidence": [0.95] * 35 + [None] * 5,
            "latency_ms": [100.0] * 40,
        }
    )
    fig = dashboard(log)
    assert len(fig.axes) == 3

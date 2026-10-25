"""Small load test for a running API (no extra dependencies: httpx + asyncio).

    python -m neuroscan.serving.loadtest --url http://127.0.0.1:8000 --concurrency 1 2 4

For each concurrency level, sends ``--requests`` uploads cycling through the demo
samples and reports throughput, p50/p95 latency and errors.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import httpx
import numpy as np

SAMPLES = Path(__file__).parent / "static" / "samples"


async def run_level(
    url: str, payloads: list[bytes], concurrency: int, requests: int, heatmap: bool
) -> dict[str, float]:
    latencies: list[float] = []
    errors = 0
    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(requests):
        queue.put_nowait(i)

    async def worker(client: httpx.AsyncClient) -> None:
        nonlocal errors
        while not queue.empty():
            i = queue.get_nowait()
            files = {"file": ("scan.jpg", payloads[i % len(payloads)], "image/jpeg")}
            start = time.perf_counter()
            try:
                r = await client.post(
                    f"{url}/v1/predict", files=files, params={"heatmap": str(heatmap).lower()}
                )
                ok = r.status_code == 200
            except httpx.HTTPError:
                ok = False
            latencies.append((time.perf_counter() - start) * 1000)
            errors += not ok

    async with httpx.AsyncClient(timeout=60) as client:
        await client.post(  # warm-up
            f"{url}/v1/predict", files={"file": ("w.jpg", payloads[0], "image/jpeg")}
        )
        start = time.perf_counter()
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))
        elapsed = time.perf_counter() - start
    return {
        "concurrency": concurrency,
        "requests": requests,
        "rps": requests / elapsed,
        "p50_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--requests", type=int, default=24, help="per concurrency level")
    parser.add_argument("--heatmap", action="store_true", help="also compute Grad-CAM")
    args = parser.parse_args()

    payloads = [p.read_bytes() for p in sorted(SAMPLES.glob("*.jpg"))]
    print(f"heatmap={args.heatmap}")
    print("| Concurrency | Requests | Requests/s | p50 latency | p95 latency | Errors |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for level in args.concurrency:
        r = asyncio.run(run_level(args.url, payloads, level, args.requests, args.heatmap))
        print(
            f"| {r['concurrency']} | {r['requests']} | {r['rps']:.1f} | {r['p50_ms']:.0f} ms "
            f"| {r['p95_ms']:.0f} ms | {r['errors']} |"
        )


if __name__ == "__main__":
    main()

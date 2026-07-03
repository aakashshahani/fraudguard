"""
FraudGuard — Phase 7 Part D: /predict latency benchmark.

Reports the number that actually matters for a checkout budget — **single-request
processing latency** (SEQUENTIAL, one request at a time), plus an assemble-vs-model
breakdown. It deliberately does NOT report a high-concurrency in-process number:
the work is CPU-bound under Python's GIL, so blasting N concurrent requests at one
process measures GIL contention, not deployment latency (production scales
throughput with multiple uvicorn workers / replicas, each serving requests at the
sequential latency below). Results (json + histogram) go to reports/figures.

    python scripts/benchmark_latency.py
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("FRAUDGUARD_STORE", "snapshot")

import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
N_REQUESTS = 400  # sequential


def _pctile(x, q):
    x = sorted(x)
    return round(x[min(int(q * len(x)), len(x) - 1)], 2)


async def _end_to_end(payloads) -> tuple[list[float], float]:
    from src.serving.api import STATE, app, build_state

    STATE.update(build_state())  # raw ASGITransport doesn't run lifespan; init directly
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://bench") as client:
        for i in range(30):  # warm up
            await client.post("/predict", json=payloads[i % len(payloads)])
        lat = []
        t0 = time.perf_counter()
        for i in range(N_REQUESTS):  # SEQUENTIAL — true per-request latency
            s = time.perf_counter()
            r = await client.post("/predict", json=payloads[i % len(payloads)])
            lat.append((time.perf_counter() - s) * 1000.0)
            assert r.status_code == 200
        wall = time.perf_counter() - t0
    return lat, wall


def _component_breakdown(payloads) -> dict:
    import joblib
    from src.modeling import WINNER_ARTIFACT
    from src.serving.assemble import FeatureAssembler
    from src.serving.feature_store import FeatureStore

    store = FeatureStore.from_snapshot(ROOT / "tests" / "fixtures" / "feature_store_snapshot.json")
    asm = FeatureAssembler.load(store)
    model = joblib.load(WINNER_ARTIFACT)
    for _ in range(20):
        model.predict_proba(asm.assemble(payloads[0]))
    a, p = [], []
    for i in range(200):
        raw = payloads[i % len(payloads)]
        t0 = time.perf_counter(); v = asm.assemble(raw); t1 = time.perf_counter()
        model.predict_proba(v); t2 = time.perf_counter()
        a.append((t1 - t0) * 1000); p.append((t2 - t1) * 1000)
    return {"assemble_p50_ms": _pctile(a, 0.5), "assemble_p99_ms": _pctile(a, 0.99),
            "model_p50_ms": _pctile(p, 0.5), "model_p99_ms": _pctile(p, 0.99)}


def main() -> None:
    payloads = [fx["raw"] for fx in json.loads(
        (ROOT / "tests" / "fixtures" / "serving_fixture.json").read_text())]
    lat, wall = asyncio.run(_end_to_end(payloads))
    res = {
        "mode": "sequential (one request at a time)",
        "n_requests": N_REQUESTS,
        "e2e_p50_ms": _pctile(lat, 0.5),
        "e2e_p95_ms": _pctile(lat, 0.95),
        "e2e_p99_ms": _pctile(lat, 0.99),
        "e2e_mean_ms": round(statistics.mean(lat), 2),
        "single_worker_rps": round(N_REQUESTS / wall, 1),
        **_component_breakdown(payloads),
        "note": "CPU-bound under the GIL; scale throughput with more uvicorn workers/replicas.",
    }
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR / "phase7_latency.json").write_text(json.dumps(res, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lat, bins=50, color="steelblue", edgecolor="none")
    ax.axvline(res["e2e_p50_ms"], color="green", ls="--", label=f"p50={res['e2e_p50_ms']}ms")
    ax.axvline(res["e2e_p99_ms"], color="red", ls="--", label=f"p99={res['e2e_p99_ms']}ms")
    ax.set_xlabel("/predict latency (ms, sequential)"); ax.set_ylabel("requests")
    ax.set_title(f"Phase 7 — /predict latency (assemble {res['assemble_p50_ms']}ms + "
                 f"model {res['model_p50_ms']}ms)")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "phase7_latency.png", dpi=120)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

"""
CI Regression Gate.
Compares latest N requests vs previous N to detect:
  - Latency regression (p95 increase > threshold)
  - Quality regression (average score drop > threshold)
Exits 1 (fails CI) if regression detected.
"""

import sys
import json
import sqlite3
import numpy as np
from pathlib import Path


DB_PATH = Path("data/observability.db")
WINDOW = 50  # Compare last 50 vs previous 50

THRESHOLDS = {
    "p95_latency_increase_pct": 20.0,   # Allow up to 20% increase
    "quality_drop_absolute": 0.10,       # Allow up to 0.10 drop
    "cost_increase_pct": 30.0,           # Allow up to 30% cost increase
}


def load_traces(offset: int, limit: int):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM request_traces ORDER BY timestamp DESC LIMIT ? OFFSET ?",
        (limit, offset)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def p95(values):
    if not values:
        return 0
    return float(np.percentile(values, 95))


def mean(values):
    return float(np.mean(values)) if values else 0


def run_regression_gate():
    recent = load_traces(offset=0, limit=WINDOW)
    baseline = load_traces(offset=WINDOW, limit=WINDOW)

    if len(recent) < 10 or len(baseline) < 10:
        print("[Gate] Not enough data to run regression check (need 10+ requests).")
        sys.exit(0)

    checks = []
    passed = True

    # Latency p95
    recent_p95 = p95([r["latency_ms"] for r in recent if r["latency_ms"]])
    baseline_p95 = p95([r["latency_ms"] for r in baseline if r["latency_ms"]])
    if baseline_p95 > 0:
        delta_pct = (recent_p95 - baseline_p95) / baseline_p95 * 100
        ok = delta_pct <= THRESHOLDS["p95_latency_increase_pct"]
        if not ok:
            passed = False
        checks.append(("p95 latency", baseline_p95, recent_p95, f"{delta_pct:+.1f}%", ok))

    # Quality score
    recent_quality = [r["quality_score"] for r in recent if r["quality_score"] is not None]
    baseline_quality = [r["quality_score"] for r in baseline if r["quality_score"] is not None]
    if recent_quality and baseline_quality:
        delta_q = mean(recent_quality) - mean(baseline_quality)
        ok = delta_q >= -THRESHOLDS["quality_drop_absolute"]
        if not ok:
            passed = False
        checks.append(("quality score", mean(baseline_quality), mean(recent_quality), f"{delta_q:+.3f}", ok))

    # Cost per request
    recent_cost = mean([r["cost_usd"] for r in recent if r["cost_usd"]])
    baseline_cost = mean([r["cost_usd"] for r in baseline if r["cost_usd"]])
    if baseline_cost > 0:
        delta_cost = (recent_cost - baseline_cost) / baseline_cost * 100
        ok = delta_cost <= THRESHOLDS["cost_increase_pct"]
        if not ok:
            passed = False
        checks.append(("cost/request", baseline_cost, recent_cost, f"{delta_cost:+.1f}%", ok))

    print("\n=== Regression Gate Results ===")
    print(f"{'Metric':20s} {'Baseline':>10} {'Current':>10} {'Delta':>10}  Status")
    print("-" * 65)
    for metric, base, cur, delta, ok in checks:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {metric:18s} {base:>10.2f} {cur:>10.2f} {delta:>10}  {status}")

    if passed:
        print("\n✅ All regression checks passed.")
        sys.exit(0)
    else:
        print("\n❌ Regression detected — blocking deployment.")
        sys.exit(1)


if __name__ == "__main__":
    run_regression_gate()

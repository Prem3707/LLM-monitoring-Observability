"""
LLM Observability Layer.
Wraps any LLM call with:
  - Distributed tracing (OpenTelemetry)
  - Latency tracking (p50 / p95 / p99)
  - Cost-per-request estimation
  - Quality score logging
  - Persistent SQLite store for the dashboard
"""

import time
import uuid
import sqlite3
import statistics
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Any, Dict, List
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from functools import wraps

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource


# ---------- OpenTelemetry Setup ----------
resource = Resource.create({"service.name": "production-rag", "service.version": "1.0.0"})
provider = TracerProvider(resource=resource)
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("rag.tracer")

# ---------- Cost Table (USD per 1K tokens, as of 2025) ----------
COST_TABLE = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}

DB_PATH = Path("data/observability.db")


# ---------- Data Model ----------
@dataclass
class RequestTrace:
    trace_id: str
    timestamp: str
    model: str
    question: str
    latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    quality_score: Optional[float]
    num_sources: int
    error: Optional[str]


# ---------- SQLite Store ----------
def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS request_traces (
            trace_id TEXT PRIMARY KEY,
            timestamp TEXT,
            model TEXT,
            question TEXT,
            latency_ms REAL,
            retrieval_latency_ms REAL,
            generation_latency_ms REAL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cost_usd REAL,
            quality_score REAL,
            num_sources INTEGER,
            error TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_trace(t: RequestTrace):
    conn = sqlite3.connect(DB_PATH)
    d = asdict(t)
    conn.execute(
        """INSERT OR REPLACE INTO request_traces VALUES
        (:trace_id,:timestamp,:model,:question,:latency_ms,:retrieval_latency_ms,
         :generation_latency_ms,:input_tokens,:output_tokens,:cost_usd,
         :quality_score,:num_sources,:error)""",
        d,
    )
    conn.commit()
    conn.close()


def load_recent_traces(limit: int = 200) -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM request_traces ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------- Percentile Calculator ----------
def percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0, "p95": 0, "p99": 0}
    sorted_v = sorted(values)
    n = len(sorted_v)
    return {
        "p50": sorted_v[int(n * 0.50)],
        "p95": sorted_v[min(int(n * 0.95), n - 1)],
        "p99": sorted_v[min(int(n * 0.99), n - 1)],
    }


# ---------- Cost Calculator ----------
def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = COST_TABLE.get(model, {"input": 0.001, "output": 0.002})
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1000


# ---------- Observability Decorator ----------
def observe(model: str = "gpt-4o-mini"):
    """Decorator that wraps a RAG function with full observability."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            trace_id = str(uuid.uuid4())[:8]
            start = time.perf_counter()

            with tracer.start_as_current_span(func.__name__) as span:
                span.set_attribute("trace.id", trace_id)
                span.set_attribute("model", model)

                error = None
                result = None
                retrieval_ms = 0.0
                generation_ms = 0.0
                input_tokens = 0
                output_tokens = 0
                quality = None
                num_sources = 0

                try:
                    result = func(*args, **kwargs)
                    # Expect result dict with timing keys
                    retrieval_ms = result.get("retrieval_latency_ms", 0)
                    generation_ms = result.get("generation_latency_ms", 0)
                    input_tokens = result.get("input_tokens", 0)
                    output_tokens = result.get("output_tokens", 0)
                    quality = result.get("quality_score")
                    num_sources = len(result.get("sources", []))
                except Exception as e:
                    error = str(e)
                    span.set_attribute("error", error)

                total_ms = (time.perf_counter() - start) * 1000
                cost = estimate_cost(model, input_tokens, output_tokens)
                question = args[0] if args else kwargs.get("question", "")

                t = RequestTrace(
                    trace_id=trace_id,
                    timestamp=datetime.utcnow().isoformat(),
                    model=model,
                    question=str(question)[:500],
                    latency_ms=round(total_ms, 2),
                    retrieval_latency_ms=round(retrieval_ms, 2),
                    generation_latency_ms=round(generation_ms, 2),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=round(cost, 6),
                    quality_score=quality,
                    num_sources=num_sources,
                    error=error,
                )
                save_trace(t)

                span.set_attribute("latency_ms", total_ms)
                span.set_attribute("cost_usd", cost)

            return result
        return wrapper
    return decorator


# ---------- Stats Reporter ----------
def print_stats():
    """Print p50/p95/p99 latency and total cost summary."""
    traces = load_recent_traces(1000)
    if not traces:
        print("No traces found.")
        return

    latencies = [t["latency_ms"] for t in traces if t["latency_ms"]]
    costs = [t["cost_usd"] for t in traces if t["cost_usd"]]
    errors = [t for t in traces if t["error"]]

    p = percentiles(latencies)
    total_cost = sum(costs)
    error_rate = len(errors) / len(traces) * 100 if traces else 0

    print(f"\n=== Observability Summary ({len(traces)} requests) ===")
    print(f"  Latency p50:   {p['p50']:.0f}ms")
    print(f"  Latency p95:   {p['p95']:.0f}ms")
    print(f"  Latency p99:   {p['p99']:.0f}ms")
    print(f"  Total cost:    ${total_cost:.4f}")
    print(f"  Avg cost/req:  ${total_cost/len(traces):.6f}")
    print(f"  Error rate:    {error_rate:.1f}%")


if __name__ == "__main__":
    init_db()
    print_stats()

# 📈  Monitoring & Observability

## What Is It?
A production observability layer that wraps any LLM/RAG system with full tracing, latency percentiles (p50/p95/p99), cost tracking, and a live dashboard. Includes a CI regression gate that blocks deployment if latency or quality degrades.

> **This is what 70% of production AI work actually looks like** — most portfolio projects skip it entirely.

---

## Industrial Applications
| Use Case | Business Value |
|---|---|
| Production RAG systems | Catch latency spikes before users complain |
| Multi-model A/B testing | Cost-per-request comparison across models |
| SLA enforcement | Automated alerts when p95 exceeds contract |
| FinOps for AI | Track spend by team, feature, or user |
| Quality regression | Block bad model updates before they ship |

---

## Key Technical Concepts

### p50 / p95 / p99 Latency
- **p50 (median)**: Half of requests are faster than this. Your "typical" user experience.
- **p95**: 95% of requests are faster. This is your SLA target — your worst regular experience.
- **p99**: 99% are faster. Your outlier tail — often 10x p50 due to cold starts or retries.

### OpenTelemetry (OTel)
Industry-standard distributed tracing framework. Every request gets a trace ID that lets you follow it across services. Traces can be exported to Jaeger, Zipkin, Datadog, etc.

### Regression Gate
Compares the last 50 requests vs the 50 before them. If p95 latency increased >20% or quality dropped >0.10, the CI pipeline fails and blocks the PR merge. Prevents accidentally shipping a slower or worse model.

### Cost Per Request
Token usage × cost-per-1K-token rate. Tracked per request so you can see which query types are expensive and optimize prompts or model selection accordingly.

---

## File Structure
```
03-monitoring-observability/
├── src/
│   ├── tracing/
│   │   └── observer.py           # @observe decorator + SQLite store
│   ├── metrics/
│   │   └── regression_gate.py    # CI gate: p95 + quality regression check
│   └── dashboard/
│       └── app.py                # Streamlit live dashboard
└── requirements.txt
```

---

# ▶️ How to Run

## Setup
```bash
cd 03-monitoring-observability
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Using the @observe Decorator
```python
from src.tracing.observer import observe, init_db

init_db()  # Creates SQLite DB on first run

@observe(model="gpt-4o-mini")
def my_rag_query(question: str) -> dict:
    # Your RAG logic here
    return {
        "answer": "...",
        "sources": [...],
        "retrieval_latency_ms": 45.2,
        "generation_latency_ms": 820.1,
        "input_tokens": 512,
        "output_tokens": 180,
        "quality_score": 0.87,
    }
```

## Run the Dashboard
```bash
streamlit run src/dashboard/app.py
# Opens at http://localhost:8501
```

## Run Regression Gate (CI)
```bash
python -m src.metrics.regression_gate
# Exit 0 = pass, Exit 1 = regression detected
```

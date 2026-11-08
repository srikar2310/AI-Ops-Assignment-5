"""eta_app.py -- Part A: REST service for the SwiftParcel ETA model.

Run (from the project root, the folder that contains data/):
    ETA_MODEL=data/eta_v1.joblib uvicorn partA.eta_app:app --port 8000
or from inside partA/:
    ETA_MODEL=../data/eta_v1.joblib uvicorn eta_app:app --port 8000

Design notes
- The model bundle is loaded ONCE at import time into the module-level variable `model`.
  The endpoint looks `model` up at request time, so tests can monkeypatch it.
- One middleware does request-id handling, metrics and the catch-all generic 500.
- Metrics are hand-rolled (no prometheus_client dependency) and are per-process.
"""
import os
import threading
import time
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, model_validator

HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve(path):
    """Find the bundle whether we are run from the project root, partA/ or elsewhere."""
    if os.path.isabs(path):
        return path
    for base in (os.getcwd(), HERE, os.path.dirname(HERE)):
        cand = os.path.join(base, path)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"model bundle not found: {path} (set ETA_MODEL)")


# ----------------------------------------------------------------------------- model
_bundle = joblib.load(_resolve(os.environ.get("ETA_MODEL", "data/eta_v1.joblib")))
model = _bundle["model"]
FEATURES = list(_bundle["feature_columns"])
VERSION = _bundle["version"]


# ----------------------------------------------------------------------------- schema
class Delivery(BaseModel):
    distance_km: float = Field(ge=0.1, le=500)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1)
    rain: int = Field(ge=0, le=1)
    weight_kg: float = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def _feasible(self):
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible delivery: distance_km > 150 and n_stops > 20")
        return self


class PredictRequest(BaseModel):
    instances: list[Delivery] = Field(min_length=1, max_length=500)


# ----------------------------------------------------------------------------- metrics
BUCKETS = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
_lock = threading.Lock()
_counts: dict = {}                      # (path, status) -> n
_bucket_counts = [0] * len(BUCKETS)     # cumulative: bucket i counts observations <= BUCKETS[i]
_sum = 0.0
_count = 0


def _record(path, status, seconds):
    global _sum, _count
    with _lock:
        key = (path, str(status))
        _counts[key] = _counts.get(key, 0) + 1
        for i, b in enumerate(BUCKETS):
            if seconds <= b:
                _bucket_counts[i] += 1
        _sum += seconds
        _count += 1


def _render_metrics():
    lines = ["# HELP eta_requests_total Total HTTP requests (excluding /metrics).",
             "# TYPE eta_requests_total counter"]
    with _lock:
        for (path, status), n in sorted(_counts.items()):
            lines.append(f'eta_requests_total{{path="{path}",status="{status}"}} {n}')
        lines += ["# HELP eta_request_latency_seconds Request latency in seconds.",
                  "# TYPE eta_request_latency_seconds histogram"]
        for b, n in zip(BUCKETS, _bucket_counts):
            lines.append(f'eta_request_latency_seconds_bucket{{le="{b}"}} {n}')
        lines.append(f'eta_request_latency_seconds_bucket{{le="+Inf"}} {_count}')
        lines.append(f"eta_request_latency_seconds_sum {_sum}")
        lines.append(f"eta_request_latency_seconds_count {_count}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- app
app = FastAPI(title="SwiftParcel ETA service")


def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


@app.middleware("http")
async def observe(request: Request, call_next):
    rid = (request.headers.get("x-request-id") or str(uuid.uuid4()))[:128]
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:  # anything not handled below -> generic 500, no internals
        response = JSONResponse(status_code=500,
                                content={"error": "internal_error", "request_id": rid})
    response.headers["X-Request-ID"] = rid
    if request.url.path != "/metrics":
        route = request.scope.get("route")
        path = getattr(route, "path", None) or "unmatched"
        _record(path, response.status_code, time.perf_counter() - t0)
    return response


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    details = []
    for e in exc.errors():
        loc = [str(x) for x in e.get("loc", ()) if x != "body"]
        details.append({"field": ".".join(loc), "msg": str(e.get("msg", ""))})
    return JSONResponse(status_code=422,
                        content={"error": "validation_failed", "details": details,
                                 "request_id": _rid(request)})


@app.post("/v1/models/eta:predict")
def predict(req: PredictRequest, request: Request):
    rid = _rid(request)
    try:
        X = pd.DataFrame([d.model_dump() for d in req.instances])[FEATURES]
        preds = [round(float(p), 1) for p in model.predict(X)]
    except Exception:
        return JSONResponse(status_code=500,
                            content={"error": "internal_error", "request_id": rid})
    return {"predictions": preds, "model_version": VERSION, "request_id": rid}


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_version": VERSION}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(_render_metrics(), media_type="text/plain; version=0.0.4")

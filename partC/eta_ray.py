"""eta_ray.py -- Part C: Ray Serve app with a sticky, customer-level canary.

    build(weight=0.1, extra_work_ms=0.0) -> Serve Application
    POST /v1/models/eta:predict   {"customer_id": "...", "instances": [ {7 features}, ... ]}
        -> {"predictions": [...], "model_version": "v1"|"v2", "arm": "v1"|"v2"}
    Header  X-Beta: 1  forces the candidate for that request.

CPU budget (cluster started with num_cpus=2):
    router 0.25 + prod 2 x 0.25 + candidate (max 3) x 0.25  = at most 1.5 CPUs
"""
import hashlib
import os
import time

import joblib
import pandas as pd
from fastapi import FastAPI, Header
from pydantic import BaseModel, Field
from ray import serve

HERE = os.path.dirname(os.path.abspath(__file__))
SALT = "eta-canary-v1"
REPLICA_CPUS = 0.25
ROUTER_CPUS = 0.25


def _resolve(path):
    if os.path.isabs(path):
        return path
    for base in (os.getcwd(), HERE, os.path.dirname(HERE)):
        p = os.path.join(base, path)
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError(path)


def bucket(customer_id: str) -> float:
    """Stable, process-independent value in [0, 1) for a customer id (SHA-256, NOT hash())."""
    digest = hashlib.sha256(f"{SALT}:{customer_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def arm_for(customer_id: str, weight: float, force_beta: bool = False) -> str:
    return "v2" if force_beta or bucket(customer_id) < weight else "v1"


@serve.deployment
class EtaModel:
    """Loads a bundle once per replica; predicts a batch of rows."""

    def __init__(self, bundle_path: str, extra_work_ms: float = 0.0):
        b = joblib.load(bundle_path)
        self.model, self.cols, self.version = b["model"], list(b["feature_columns"]), b["version"]
        self.extra_work_ms = extra_work_ms

    def predict(self, rows):
        if self.extra_work_ms > 0:
            end = time.perf_counter() + self.extra_work_ms / 1000
            while time.perf_counter() < end:
                pass
        X = pd.DataFrame(rows)[self.cols]
        return {"predictions": [round(float(p), 1) for p in self.model.predict(X)],
                "model_version": self.version}


class PredictBody(BaseModel):
    customer_id: str = Field(min_length=1)
    instances: list[dict] = Field(min_length=1, max_length=500)


api = FastAPI()


@serve.deployment(ray_actor_options={"num_cpus": ROUTER_CPUS})
@serve.ingress(api)
class Router:
    def __init__(self, prod_handle, cand_handle, weight: float):
        self.prod, self.cand, self.weight = prod_handle, cand_handle, weight

    @api.post("/v1/models/eta:predict")
    async def predict(self, body: PredictBody, x_beta: str | None = Header(default=None)):
        arm = arm_for(body.customer_id, self.weight, force_beta=(x_beta == "1"))
        handle = self.cand if arm == "v2" else self.prod
        out = await handle.predict.remote(body.instances)
        return {**out, "arm": arm}


def build(weight: float = 0.1, extra_work_ms: float = 0.0):
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    v1 = _resolve(os.environ.get("ETA_PROD_MODEL", "data/eta_v1.joblib"))
    v2 = _resolve(os.environ.get("ETA_CAND_MODEL", "data/eta_v2.joblib"))
    prod = EtaModel.options(
        name="EtaProd", num_replicas=2,
        ray_actor_options={"num_cpus": REPLICA_CPUS},
    ).bind(v1, extra_work_ms)
    cand = EtaModel.options(
        name="EtaCandidate",
        autoscaling_config={"min_replicas": 1, "max_replicas": 3,
                            "target_ongoing_requests": 2,
                            "upscale_delay_s": 5, "downscale_delay_s": 60},
        ray_actor_options={"num_cpus": REPLICA_CPUS},
    ).bind(v2, extra_work_ms)
    return Router.bind(prod, cand, weight)

"""test_eta_app.py -- Part A2. Run from partA/ or the project root:  pytest partA -q
Needs data/eta_v1.joblib (python eta_data.py)."""
import os
import re
import sys
import uuid

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eta_app

client = TestClient(eta_app.app, raise_server_exceptions=False)
URL = "/v1/models/eta:predict"

ROW = dict(distance_km=10, n_stops=3, hour=9, weekday=2, traffic_index=0.5, rain=0, weight_kg=5)


def row(**kw):
    return {**ROW, **kw}


def post(instances, **kw):
    return client.post(URL, json={"instances": instances}, **kw)


def test_valid_request():
    r = post([ROW])
    assert r.status_code == 200
    b = r.json()
    assert len(b["predictions"]) == 1 and isinstance(b["predictions"][0], float)
    assert b["model_version"] == eta_app.VERSION
    assert b["request_id"] == r.headers["X-Request-ID"]
    assert round(b["predictions"][0], 1) == b["predictions"][0]


def test_batch_order_preserved():
    a, b = row(distance_km=5), row(distance_km=120, n_stops=5)
    fwd = post([a, b]).json()["predictions"]
    rev = post([b, a]).json()["predictions"]
    assert fwd[0] != fwd[1]
    assert fwd == rev[::-1]
    assert [post([a]).json()["predictions"][0], post([b]).json()["predictions"][0]] == fwd


@pytest.mark.parametrize("bad", [
    row(distance_km=999),
    row(rain=2),
    row(hour=24),
    row(weight_kg=0),
    row(traffic_index=1.5),
    row(distance_km="far"),
])
def test_invalid_values_rejected(bad):
    r = post([bad])
    assert r.status_code == 422
    assert r.json()["error"] == "validation_failed"


def test_missing_field_rejected_with_details():
    bad = dict(ROW)
    del bad["weight_kg"]
    r = post([bad])
    body = r.json()
    assert r.status_code == 422
    assert body["error"] == "validation_failed"
    assert body["request_id"] == r.headers["X-Request-ID"]
    assert any("weight_kg" in d["field"] for d in body["details"])
    assert all(set(d) == {"field", "msg"} for d in body["details"])


def test_malformed_json_rejected():
    r = client.post(URL, content="{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422 and r.json()["error"] == "validation_failed"


def test_cross_field_infeasible():
    r = post([row(distance_km=160, n_stops=25)])
    assert r.status_code == 422
    assert "infeasible" in r.text
    # each condition alone is fine
    assert post([row(distance_km=160, n_stops=20)]).status_code == 200
    assert post([row(distance_km=150, n_stops=25)]).status_code == 200


def test_batch_size_limits():
    assert post([]).status_code == 422
    assert post([ROW] * 501).status_code == 422
    assert post([ROW] * 500).status_code == 200
    assert len(post([ROW] * 500).json()["predictions"]) == 500


def test_request_id_echoed():
    r = post([ROW], headers={"X-Request-ID": "my-id-123"})
    assert r.headers["X-Request-ID"] == "my-id-123"
    assert r.json()["request_id"] == "my-id-123"


def test_request_id_generated():
    r = post([ROW])
    assert str(uuid.UUID(r.headers["X-Request-ID"])) == r.headers["X-Request-ID"]
    assert r.json()["request_id"] == r.headers["X-Request-ID"]


def test_request_id_on_error_response():
    r = post([row(rain=5)], headers={"X-Request-ID": "err-77"})
    assert r.status_code == 422
    assert r.headers["X-Request-ID"] == "err-77"
    assert r.json()["request_id"] == "err-77"


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": eta_app.VERSION}


def test_metrics_output_and_metrics_not_counted():
    post([ROW])
    post([row(rain=9)])
    client.get("/metrics")
    text = client.get("/metrics").text
    assert re.search(r'eta_requests_total\{path="/v1/models/eta:predict",status="200"\} \d+', text)
    assert re.search(r'eta_requests_total\{path="/v1/models/eta:predict",status="422"\} \d+', text)
    assert text.count("eta_request_latency_seconds_bucket{le=") >= 5
    assert 'le="+Inf"' in text
    assert "eta_request_latency_seconds_sum" in text
    assert "eta_request_latency_seconds_count" in text
    assert 'path="/metrics"' not in text


def test_metrics_counter_increments():
    def n():
        m = re.search(r'path="/v1/models/eta:predict",status="200"\} (\d+)', client.get("/metrics").text)
        return int(m.group(1)) if m else 0
    before = n()
    post([ROW])
    assert n() == before + 1


def test_inference_failure_returns_generic_500(monkeypatch):
    def boom(X):
        raise RuntimeError("secret internal detail /home/me/model.pkl")
    monkeypatch.setattr(eta_app.model, "predict", boom)
    r = post([ROW], headers={"X-Request-ID": "boom-1"})
    assert r.status_code == 500
    assert "secret" not in r.text and "RuntimeError" not in r.text and "Traceback" not in r.text
    assert r.headers["X-Request-ID"] == "boom-1"
    assert r.json()["request_id"] == "boom-1"

"""smoke_eta.py -- Part D1 release gate. Exit 0 if every check passes, 1 otherwise.

    python smoke_eta.py --url http://localhost:8000/v1/models/eta:predict --expect-version v1 --max-p95-ms 80
    python smoke_eta.py --url http://localhost:5101/invocations --format mlflow --expect-version v1
(Part C: use --url .../v1/models/eta:predict; --expect-version should match the version
 the smoke customer_id lands on, or pass --header "X-Beta: 1" with --expect-version v2.)
"""
import argparse
import sys
import time

import numpy as np
import requests

BASE = dict(distance_km=30.0, n_stops=5, hour=11, weekday=2, traffic_index=0.4, rain=0, weight_kg=8.0)


def body(fmt, rows):
    if fmt == "mlflow":
        return {"dataframe_records": rows}
    return {"customer_id": "smoke-test", "instances": rows}


def parse(fmt, j):
    """-> (list of float predictions, version or None)"""
    p = j["predictions"]
    if isinstance(p, dict):
        return [float(x) for x in p["eta_minutes"]], (p.get("model_version") or [None])[0]
    if p and isinstance(p[0], dict):
        return [float(x["eta_minutes"]) for x in p], p[0].get("model_version")
    return [float(x) for x in p], j.get("model_version")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--expect-version", required=True)
    ap.add_argument("--max-p95-ms", type=float, default=80.0)
    ap.add_argument("--format", choices=["rest", "mlflow"], default="rest")
    ap.add_argument("--header", action="append", default=[], help='extra header, e.g. "X-Beta: 1"')
    a = ap.parse_args()
    headers = dict(h.split(":", 1) for h in a.header)
    headers = {k.strip(): v.strip() for k, v in headers.items()}
    sess = requests.Session()

    def post(rows, raw=None):
        return sess.post(a.url, json=raw if raw is not None else body(a.format, rows),
                         headers=headers, timeout=30)

    def eta(**kw):
        r = post([{**BASE, **kw}])
        r.raise_for_status()
        return parse(a.format, r.json())[0][0]

    results = []

    def check(name):
        def deco(fn):
            try:
                msg = fn()
                results.append((name, True, msg or ""))
            except Exception as e:  # noqa: BLE001
                results.append((name, False, f"{type(e).__name__}: {e}"))
            return fn
        return deco

    @check("valid request returns 200 and one finite positive prediction")
    def _():
        r = post([BASE])
        assert r.status_code == 200, f"status {r.status_code}"
        preds, _v = parse(a.format, r.json())
        assert len(preds) == 1 and np.isfinite(preds[0]) and 0 < preds[0] < 2000, preds
        return f"eta={preds[0]}"

    @check(f"model_version == {a.expect_version}")
    def _():
        r = post([BASE])
        r.raise_for_status()
        v = parse(a.format, r.json())[1]
        assert v == a.expect_version, f"got {v!r}"
        return f"version={v}"

    @check("batch of 3 returns 3 predictions in input order")
    def _():
        rows = [{**BASE, "distance_km": d} for d in (5.0, 40.0, 90.0)]
        r = post(rows)
        r.raise_for_status()
        preds, _v = parse(a.format, r.json())
        singles = [eta(distance_km=d) for d in (5.0, 40.0, 90.0)]
        assert len(preds) == 3
        assert np.allclose(preds, singles, atol=0.11), (preds, singles)
        return f"{preds}"

    @check("sanity: longer distance never gives a shorter ETA")
    def _():
        e = [eta(distance_km=d) for d in (5.0, 40.0, 120.0)]
        assert e[0] <= e[1] + 0.5 and e[1] <= e[2] + 0.5, e
        return f"5/40/120 km -> {e}"

    @check("sanity: heavier traffic never shortens the ETA")
    def _():
        e = [eta(traffic_index=t) for t in (0.05, 0.5, 0.95)]
        assert e[0] <= e[1] + 0.5 and e[1] <= e[2] + 0.5, e
        return f"traffic 0.05/0.5/0.95 -> {e}"

    @check("sanity: more stops never shortens the ETA")
    def _():
        e = [eta(n_stops=n) for n in (1, 10, 20)]
        assert e[0] <= e[1] + 0.5 and e[1] <= e[2] + 0.5, e
        return f"stops 1/10/20 -> {e}"

    @check("invalid input (rain=2) is rejected with 422" if a.format == "rest"
           else "invalid input (rain=2) is rejected with 4xx")
    def _():
        r = post([{**BASE, "rain": 2}])
        if a.format == "rest":
            assert r.status_code == 422, f"status {r.status_code}"
        else:
            assert 400 <= r.status_code < 500, f"status {r.status_code}"
        return f"status {r.status_code}"

    if a.format == "rest":
        @check("infeasible delivery (160 km, 25 stops) is rejected with 422")
        def _():
            r = post([{**BASE, "distance_km": 160.0, "n_stops": 25}])
            assert r.status_code == 422, f"status {r.status_code}"

    @check(f"latency: p95 of 40 sequential requests <= {a.max_p95_ms} ms")
    def _():
        post([BASE])
        t = []
        for _i in range(40):
            t0 = time.perf_counter()
            post([BASE])
            t.append((time.perf_counter() - t0) * 1000)
        p95 = float(np.percentile(t, 95))
        assert p95 <= a.max_p95_ms, f"p95={p95:.1f} ms"
        return f"p95={p95:.1f} ms"

    for name, ok, msg in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  {msg}")
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.RequestException as e:
        print(f"[FAIL] cannot reach service: {e}")
        sys.exit(1)
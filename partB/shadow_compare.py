"""shadow_compare.py -- Part B3. Send the SAME 200 single-row requests to prod (5101) and candidate (5102).

Start the servers first (project root, same MLFLOW_TRACKING_URI as used for register):
    export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
    mlflow models serve -m "models:/eta-model@prod"      -p 5101 --env-manager local --no-conda &
    mlflow models serve -m "models:/eta-model@candidate" -p 5102 --env-manager local --no-conda &
(if your mlflow rejects --no-conda, drop it.)  Then:
    python partB/shadow_compare.py | tee partB/shadow_output.txt
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import requests

FEATURES = ["distance_km", "n_stops", "hour", "weekday", "traffic_index", "rain", "weight_kg"]


def find_data(path):
    for base in (os.getcwd(), os.path.dirname(os.path.abspath(__file__)),
                 os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        p = os.path.join(base, path)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(path)


def extract(j):
    p = j["predictions"]
    if isinstance(p, dict):
        return float(p["eta_minutes"][0])
    first = p[0]
    return float(first["eta_minutes"] if isinstance(first, dict) else first)


def call(sess, port, rec):
    t0 = time.perf_counter()
    r = sess.post(f"http://localhost:{port}/invocations", json={"dataframe_records": [rec]}, timeout=30)
    ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return extract(r.json()), ms


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/eta_fresh.csv")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--prod-port", type=int, default=5101)
    ap.add_argument("--cand-port", type=int, default=5102)
    a = ap.parse_args()

    rows = pd.read_csv(find_data(a.data))[FEATURES].head(a.n).to_dict(orient="records")
    s = requests.Session()
    for port in (a.prod_port, a.cand_port):
        call(s, port, rows[0])

    pp, pl, cp, cl = [], [], [], []
    for rec in rows:
        p, t1 = call(s, a.prod_port, rec)
        c, t2 = call(s, a.cand_port, rec)
        pp.append(p); pl.append(t1); cp.append(c); cl.append(t2)

    pp, cp = np.array(pp), np.array(cp)
    diff = np.abs(pp - cp)
    print(f"requests compared                    : {len(rows)}")
    print(f"(i)   mean |prod - candidate|        : {diff.mean():.2f} min")
    print(f"(ii)  % differing by > 10 min        : {100 * (diff > 10).mean():.1f} %")
    print(f"(iii) p95 latency prod (:{a.prod_port})       : {np.percentile(pl, 95):.1f} ms")
    print(f"      p95 latency candidate (:{a.cand_port})  : {np.percentile(cl, 95):.1f} ms")
    print(f"      mean prediction prod / candidate: {pp.mean():.2f} / {cp.mean():.2f} min")

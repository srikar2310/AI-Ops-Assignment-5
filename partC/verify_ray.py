"""verify_ray.py -- Part C2. Deploys eta_ray on a 2-CPU Ray and checks (a)-(e) with assertions.

    python partC/verify_ray.py            # writes partC/verify_output.txt next to this file
Run from the project root so data/ is found (or set ETA_PROD_MODEL / ETA_CAND_MODEL).
"""
import concurrent.futures as cf
import os
import sys
import time

import ray
import requests
from ray import serve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eta_ray 

PORT = int(os.environ.get("SERVE_PORT", "8100"))
URL = f"http://127.0.0.1:{PORT}/v1/models/eta:predict"
ROW = dict(distance_km=12.0, n_stops=4, hour=10, weekday=2, traffic_index=0.4, rain=0, weight_kg=6.0)
LOG = []


def say(msg):
    print(msg, flush=True)
    LOG.append(msg)


def call(sess, cid, beta=False):
    r = sess.post(URL, json={"customer_id": cid, "instances": [ROW]},
                  headers={"X-Beta": "1"} if beta else {}, timeout=30)
    r.raise_for_status()
    return r.json()["arm"]


def arms(ids, beta=False):
    local = __import__("threading").local()

    def one(cid):
        if not hasattr(local, "s"):
            local.s = requests.Session()
        return cid, call(local.s, cid, beta)
    with cf.ThreadPoolExecutor(8) as ex:
        return dict(ex.map(one, ids))


def deploy(weight):
    serve.run(eta_ray.build(weight=weight), name="eta", route_prefix="/")
    s = requests.Session()
    for _ in range(60):
        try:
            call(s, "probe")
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("service did not become ready")


def share(a):
    return sum(v == "v2" for v in a.values()) / len(a)


def main():
    ray.init(num_cpus=2, ignore_reinit_error=True, logging_level="ERROR")
    serve.start(http_options={"host": "127.0.0.1", "port": PORT})
    say(f"cluster resources: {ray.cluster_resources()}")
    ids = [f"customer-{i:04d}" for i in range(1000)]

    deploy(0.2)
    say(f"CPU available while app is running: {ray.available_resources().get('CPU')} of 2 (app fits on 2 CPUs)")
    first = arms(ids)

    s = share(first)
    say(f"(a) share routed to candidate at weight 0.2: {s:.3f} (target 0.2 +/- 0.04)")
    assert abs(s - 0.2) <= 0.04, f"share {s} outside tolerance"

    sub = ids[:200]
    again = arms(sub)
    changed = [c for c in sub if again[c] != first[c]]
    say(f"(b) 200 customers, second call: {len(changed)} changed arm")
    assert not changed

    serve.delete("eta")
    deploy(0.2)
    after = arms(sub)
    changed = [c for c in sub if after[c] != first[c]]
    say(f"(c) after delete + redeploy of identical app: {len(changed)} of 200 changed arm")
    assert not changed

    prod_ids = [c for c in ids if first[c] == "v1"][:50]
    forced = arms(prod_ids, beta=True)
    n_forced = sum(v == "v2" for v in forced.values())
    say(f"(e) X-Beta: 1 on 50 prod-arm customers: {n_forced}/50 routed to v2")
    assert len(prod_ids) == 50 and n_forced == 50

    deploy(0.5)
    raised = arms(ids)
    was_v2 = [c for c in ids if first[c] == "v2"]
    moved_back = [c for c in was_v2 if raised[c] == "v1"]
    say(f"(d) weight 0.2 -> 0.5: share now {share(raised):.3f}; "
        f"{len(moved_back)} of {len(was_v2)} former candidate customers moved back to prod")
    assert not moved_back

    say("ALL CHECKS PASSED")
    serve.shutdown()
    ray.shutdown()


if __name__ == "__main__":
    try:
        main()
    finally:
        with open(os.path.join(HERE, "verify_output.txt"), "w") as f:
            f.write("\n".join(LOG) + "\n")
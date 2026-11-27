#!/usr/bin/env python3
"""Run cgf680.layer_shadow for stored CPU cases (host side, stdlib only). One fresh container per case, the torch
thread count of the original CPU run (the request stream depends on it), N cases in parallel.
  python3 tools/run_layer_labels.py --models resnet-18,resnet-50,mobilenet-v2-1.0-224,vit-large-patch16-224 --jobs 4
Output: runs/layer_labels/<case_id>.npz + .log, runs/layer_labels/summary.jsonl (resumable: existing npz skipped)
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

E = Path(__file__).resolve().parents[1]


def one(cid, root, out, mem, cases="manifest/cases395_l256.jsonl", models_file=None):
    cdir = root / cid
    res = json.loads((cdir / "result.json").read_text())
    th = int(res["environment"]["threads"])
    npz, log = out / f"{cid}.npz", out / f"{cid}.log"
    cmd = ["bash", str(E / "docs/run_in_image.sh"), "-m", "cgf680.layer_shadow", cid, cases,
           str(cdir.relative_to(E)), str(th), str(npz.relative_to(E))] + ([models_file] if models_file else [])
    env = {"MEM": mem, "CPUS": str(th), "PATH": "/usr/bin:/bin"}
    p = subprocess.run(cmd, cwd=E, env=env, capture_output=True, text=True, timeout=3600)
    log.write_text(p.stdout + p.stderr)
    line = next((x for x in p.stdout.splitlines() if x.startswith("LAYER_LABELS ")), None)
    row = json.loads(line[len("LAYER_LABELS "):]) if line else {"case_id": cid, "error": p.returncode}
    row["threads"] = th
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--runs-root", default="runs/main395_l256/cases")
    ap.add_argument("--out", default="runs/layer_labels")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--mem", default="20g")
    ap.add_argument("--cases", default="manifest/cases395_l256.jsonl")      # 2026-11-23: new-model cases
    ap.add_argument("--models-file", default=None, help="models manifest inside the image, e.g. /exp/manifest/models_newmodels_20261122.json")
    a = ap.parse_args()
    root, out = E / a.runs_root, E / a.out
    out.mkdir(parents=True, exist_ok=True)
    models = set(a.models.split(","))
    cases = sorted(d.name for d in root.iterdir() if d.name.split("__")[0] in models
                   and (d / "result.json").exists()
                   and json.loads((d / "result.json").read_text()).get("status") == "ok"
                   and not (out / f"{d.name}.npz").exists())
    print(f"todo {len(cases)}", flush=True)
    with ThreadPoolExecutor(a.jobs) as ex, (out / "summary.jsonl").open("a") as f:
        for row in ex.map(lambda c: one(c, root, out, a.mem, a.cases, a.models_file), cases):
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(json.dumps(row)[:300], flush=True)
    print("LAYER_LABELS_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Did the g4dn env comparison (A -> C -> B) finish as planned? Read-only. Prints problems and the commands to redo them.

  python3 -I envcmp/check_envcmp.py [label_prefix=aws-g4dn.xlarge__envcmp_]      (run inside cgf-fp16-vram-20261111)
"""
import glob
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PFX = sys.argv[1] if len(sys.argv) > 1 else "aws-g4dn.xlarge__envcmp_"
IMG_A = "sha256:457da255e6e2de7ee05ff7814674c90abcf6d1ed04e73df3cc4eebcdb31fb032"
IMG_C = "sha256:0b22fba14104ce1fe5f087908d74f93793f63a1f961cac2b6dafa5bed225efcc"
EXP = {"A": {"driver": "595.112.07", "torch": "2.6.0+cu124", "image": IMG_A},
       "C": {"driver": "595.112.07", "torch": "2.5.1+cu124", "image": IMG_C},
       "B": {"driver": "580.196.07", "torch": "2.6.0+cu124", "image": IMG_A}}
C_BLOCKED = ("opt-125m__", "vit-large-patch16-224__")   # .bin weights: transformers blocks torch < 2.6 (expected in C)
ids = [c["condition_id"] for c in json.load(open("config/conditions_envcmp39.json"))["conditions"]]
cur = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
problems, todo, info = [], {"A": [], "C": [], "B": []}, []
P = lambda s: problems.append(s)  # noqa: E731


def j(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def ts(s):
    return datetime.fromisoformat(s).timestamp()


a_facts = sorted(glob.glob("envcmp/records/A_before_*/DRIVER_FACTS.txt"))
a_kernel = dict(l.strip().split("=", 1) for l in open(a_facts[0]) if "=" in l).get("kernel") if a_facts else None
intervals, uuids = [], {}
print(f"current driver: {cur}   A kernel: {a_kernel}   labels: {sorted(os.path.basename(d) for d in glob.glob('runs/' + PFX + '*'))}")
for env in ("A", "C", "B"):
    lab = PFX + env
    n_ok = n_exp_fail = 0
    for cid in ids:
        d = Path("runs") / lab / cid
        r, o, la = j(d / "measure.json"), j(d / "measure.outside.json"), j(d / "measure.launch.json")
        if la and la.get("start") and la.get("end"):
            intervals.append((ts(la["start"]), ts(la["end"]), lab, cid))
        if r is None:
            P(f"{env} {cid}: measure.json missing")
            todo[env].append(("missing", cid))
            continue
        e, st = r.get("environment") or {}, r.get("status")
        uuids.setdefault(e.get("nvml_uuid") or e.get("gpu_uuid"), set()).add(env)
        for k in ("driver", "torch"):
            if e.get(k) and e.get(k) != EXP[env][k]:
                P(f"{env} {cid}: {k}={e.get(k)} (expected {EXP[env][k]})")
        if e.get("image_id") and e.get("image_id") != EXP[env]["image"]:
            P(f"{env} {cid}: image {e.get('image_id')[:19]} (expected {EXP[env]['image'][:19]})")
        if a_kernel and e.get("kernel") and e.get("kernel") != a_kernel:
            P(f"{env} {cid}: kernel {e.get('kernel')} != A {a_kernel}")
        if st == "ok":
            n_ok += 1
            pts = (o or {}).get("points") or {}
            miss = [p for p in ("cuda_init", "after_load", "after_warmup") if (pts.get(p) or {}).get("nvml_process_used") is None]
            if miss:
                P(f"{env} {cid}: outside points missing {miss}")
                todo[env].append(("ok", cid))
        elif env == "C" and cid.startswith(C_BLOCKED) and "at least v2.6" in (r.get("error") or ""):
            n_exp_fail += 1
        else:
            P(f"{env} {cid}: status={st} stage={r.get('failed_stage')} err={(r.get('error') or '')[:100]}")
            todo[env].append((st, cid))
    exp_ok = 33 if env == "C" else 39
    info.append(f"{env}: ok {n_ok}/{exp_ok}" + (f", expected C load errors {n_exp_fail}/6" if env == "C" else ""))
    dg = Path("runs") / lab / "_diag"
    want = ["attention_kernels.json"] if env != "C" else ["attention_kernels_no_bin.json"]
    for w in want:
        if not (dg / w).exists():
            P(f"{env}: attention diag {w} missing (diagnostic only)")
            todo[env].append(("diag", w))
# other labels (reruns): which driver/torch did they really use?
for d in sorted(glob.glob(f"runs/{PFX}*__*")):
    lab = os.path.basename(d)
    drv = {(j(p) or {}).get("environment", {}).get("driver") for p in glob.glob(d + "/*/measure.json")}
    trc = {(j(p) or {}).get("environment", {}).get("torch") for p in glob.glob(d + "/*/measure.json")}
    st = [(j(p) or {}).get("status") for p in glob.glob(d + "/*/measure.json")]
    base = lab[len(PFX):].split("__")[0]
    valid = base in EXP and drv <= {EXP[base]["driver"]} and trc <= {EXP[base]["torch"]}
    info.append(f"rerun label {lab}: n={len(st)} status={sorted(set(map(str, st)))} driver={sorted(map(str, drv))} torch={sorted(map(str, trc))} -> "
                + ("valid as " + base if valid else "NOT a valid " + base + " run (different driver/torch) - keep, exclude from analysis"))
    for p in glob.glob(d + "/*/measure.launch.json"):
        la = j(p)
        if la and la.get("start") and la.get("end"):
            intervals.append((ts(la["start"]), ts(la["end"]), lab, p.split("/")[-2]))
# overlaps: two GPU jobs at the same time, or a host_record GPU check during a measurement
intervals.sort()
for i in range(len(intervals) - 1):
    a, b = intervals[i], intervals[i + 1]
    if b[0] < a[1]:
        P(f"OVERLAP: {a[2]}/{a[3]} and {b[2]}/{b[3]} ran at the same time")
for rd in [x for x in glob.glob("envcmp/records/*_before_2*") if os.path.isdir(x)]:
    t = datetime.strptime(rd.rsplit("_", 1)[-1], "%Y%m%dT%H%M%S").timestamp()
    for s, e, lab, cid in intervals:
        if s - 1 <= t <= e + 30:
            P(f"HOST_RECORD {os.path.basename(rd)} (GPU check containers) during {lab}/{cid} -> re-run that condition to confirm")
            env = lab[len(PFX):].split("__")[0]
            if env in todo and lab == PFX + env:
                todo[env].append(("overlap", cid))
if len([u for u in uuids if u]) > 1:
    info.append(f"note: more than one GPU UUID across envs: {uuids}")
print("\n".join(info))
print(f"\nPROBLEMS ({len(problems)}):" if problems else "\nPROBLEMS: none")
for p in problems:
    print("  -", p)
print("\nREDO:")
any_todo = False
for env, items in todo.items():
    items = list(dict.fromkeys(items))
    if not items:
        continue
    any_todo = True
    need = EXP[env]["driver"]
    print(f"  [{env}] needs driver {need} (now {cur}) {'OK' if cur == need else '-> switch driver first (see GUIDE / driver swap commands)'}")
    miss = [c for k, c in items if k == "missing"]
    if miss:
        print(f"    missing results -> resume:  bash envcmp/run_env.sh {env} ~/hf-cache")
    for k in sorted({k for k, _ in items if k not in ("missing", "diag", "overlap", "ok")}):
        cs = [c for kk, c in items if kk == k]
        print(f"    status {k} ({len(cs)}) -> same label, old result kept in attempts/:")
        print(f"      python3 -I tools/run_conditions_env.py --conditions conditions_envcmp39.json --list {','.join(cs)} --mode measure "
              f"--gpu-label {PFX}{env} --attn default --hf-cache ~/hf-cache --image {'cloudgpufit-bench:pt251-cu124-v1' if env == 'C' else 'cloudgpufit-bench:cu124-v1'} "
              f"--env-name {env} --retry-status {k}")
    ov = [c for k, c in items if k in ("overlap", "ok")]
    if ov:
        print(f"    confirm (separate label, originals untouched):")
        print(f"      echo '{json.dumps({'condition_ids': ov})}' > /tmp/redo_{env}.json && bash envcmp/rerun_flagged.sh {env} /tmp/redo_{env}.json confirm ~/hf-cache")
    if any(k == "diag" for k, _ in items):
        print(f"    attention diag: see GUIDE (run_env.sh {env} re-runs it if attention_kernels.json is missing; for C use the no_bin command)")
if not any_todo:
    print("  nothing to redo")

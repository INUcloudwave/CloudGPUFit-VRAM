#!/usr/bin/env python3
"""Collect and compare the env-comparison runs (A baseline, C PyTorch env, B driver) by condition_id. CPU only.

  python3 -I compare_envs.py <exp root> <conditions json (list or manifest)> <out dir> \
      A=runs/<labelA> C=runs/<labelC> B=runs/<labelB> [REF=runs/<previous g4dn L256 label>] \
      [--res-mib 20] [--res-rel 0.02] [--alloc-mib 1] [--outside-mib 16]

Per env and condition it reads <cid>/measure.json (worker, unchanged), <cid>/measure.outside.json (wrapper points),
<cid>/measure.launch.json (wall), and writes:
  envcmp_long.csv     one row per (env, condition): status, first-measure peak alloc/reserved, NVML process peak,
                      torch alloc/reserved + NVML process used + outside at cuda_init / after_load / after_warmup,
                      allocator retries/OOMs, driver, torch, cuDNN, requested/actual attention, wall
  envcmp_wide.csv     one row per condition: values per env and the differences B-A and C-A (bytes and MiB)
  envcmp_summary.md   status changes, difference statistics, flagged conditions
  flagged_<X>_vs_A.json  {"condition_ids": [...]} for re-running only the conditions that stood out (runner --list)
Flag rule (any): status differs; |d peak reserved| >= res-mib MiB and >= res-rel relative; |d peak allocated| >= alloc-mib
MiB; |d outside after_warmup| >= outside-mib MiB. Missing values never count as a difference (they are listed separately).
"""
import csv
import json
import statistics
import sys
from pathlib import Path

pos = [a for a in sys.argv[1:] if not a.startswith("--") and "=" not in a]
envs = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a and not a.startswith("--"))
opt = {sys.argv[i][2:]: float(sys.argv[i + 1]) for i in range(1, len(sys.argv) - 1) if sys.argv[i].startswith("--")}
root, cond_path, outdir = Path(pos[0]), Path(pos[1]), Path(pos[2])
RES_MIB, RES_REL = opt.get("res-mib", 20), opt.get("res-rel", 0.02)
ALLOC_MIB, OUT_MIB = opt.get("alloc-mib", 1), opt.get("outside-mib", 16)
MiB = 2 ** 20
cj = json.loads(cond_path.read_text())
ids = cj["condition_ids"] if "condition_ids" in cj else [c["condition_id"] for c in cj["conditions"]]
outdir.mkdir(parents=True, exist_ok=True)
POINTS = ("cuda_init", "after_load", "after_warmup")


def load(p):
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def row(env, cid):
    d = root / envs[env] / cid
    r, o, la = load(d / "measure.json"), load(d / "measure.outside.json"), load(d / "measure.launch.json")
    s, e, mdl = r.get("summary") or {}, r.get("environment") or {}, r.get("model") or {}
    ms = r.get("measures") or []
    last = (ms[-1].get("stats_end") if ms else None) or r.get("stats_at_failure") or {}
    x = {"env": env, "label": envs[env], "condition_id": cid, "status": r.get("status", "missing"),
         "failed_stage": r.get("failed_stage"), "n_measures_ok": sum(1 for m in ms if m.get("ok")),
         "first_peak_allocated": s.get("first_measure_peak_allocated"), "first_peak_reserved": s.get("first_measure_peak_reserved"),
         "max_peak_reserved": s.get("max_peak_reserved"), "max_nvml_process_peak": s.get("max_nvml_process_peak"),
         "num_alloc_retries": last.get("num_alloc_retries"), "num_ooms": last.get("num_ooms"),
         "driver": e.get("driver") or (o.get("env") or {}).get("driver"), "torch": e.get("torch"), "cudnn": e.get("cudnn"),
         "gpu": e.get("gpu_name"), "kernel": e.get("kernel"), "image_id": e.get("image_id"),
         "cuda_module_loading": (e.get("cuda_module_loading_driver") or {}).get("mode"),
         "alloc_conf": (e.get("env") or {}).get("PYTORCH_CUDA_ALLOC_CONF"), "torch_threads": e.get("torch_threads"),
         "attn_requested": mdl.get("attn_requested"), "attn_actual_config": mdl.get("attn_implementation"),
         "wall_s": la.get("wall_s"), "error_head": (r.get("error") or "")[:160]}
    for p in POINTS:
        q = (o.get("points") or {}).get(p) or {}
        for k in ("allocated", "reserved", "nvml_process_used", "outside", "num_alloc_retries"):
            x[f"{p}_{k}"] = q.get(k)
    return x


long_rows = [row(env, cid) for env in envs for cid in ids]
with (outdir / "envcmp_long.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
    w.writeheader()
    w.writerows(long_rows)

L = {(x["env"], x["condition_id"]): x for x in long_rows}
METRICS = ["first_peak_allocated", "first_peak_reserved", "max_nvml_process_peak",
           "cuda_init_outside", "after_load_outside", "after_warmup_outside", "after_warmup_reserved",
           "after_warmup_nvml_process_used", "num_alloc_retries"]
wide, flagged, missing = [], {e: [] for e in envs if e != "A"}, []
stats = {e: {m: [] for m in METRICS} for e in envs if e != "A"}
for cid in ids:
    a = L.get(("A", cid), {})
    w = {"condition_id": cid}
    for env in envs:
        x = L[(env, cid)]
        w[f"{env}_status"] = x["status"]
        for m in METRICS:
            w[f"{env}_{m}"] = x[m]
    for env in envs:
        if env == "A":
            continue
        b, why = L[(env, cid)], []
        if a.get("status") != b["status"]:
            why.append(f"status {a.get('status')}->{b['status']}")
        for m in METRICS:
            va, vb = a.get(m), b.get(m)
            dv = (vb - va) if isinstance(va, int) and isinstance(vb, int) else None
            w[f"d_{env}_minus_A_{m}"] = dv
            if dv is not None:
                stats[env][m].append(dv)
            elif m in ("first_peak_reserved", "after_warmup_outside") and a.get("status") == b["status"] == "ok" \
                    and not (m == "after_warmup_outside" and env == "REF"):  # REF = earlier run without the wrapper
                missing.append(f"{env} {cid} {m}")
        d_res, d_al = w[f"d_{env}_minus_A_first_peak_reserved"], w[f"d_{env}_minus_A_first_peak_allocated"]
        d_out = w[f"d_{env}_minus_A_after_warmup_outside"]
        if d_res is not None and abs(d_res) >= RES_MIB * MiB and abs(d_res) >= RES_REL * (a.get("first_peak_reserved") or 1):
            why.append(f"peak_reserved {d_res / MiB:+.1f} MiB")
        if d_al is not None and abs(d_al) >= ALLOC_MIB * MiB:
            why.append(f"peak_allocated {d_al / MiB:+.1f} MiB")
        if d_out is not None and abs(d_out) >= OUT_MIB * MiB:
            why.append(f"outside_after_warmup {d_out / MiB:+.1f} MiB")
        w[f"flag_{env}_vs_A"] = "; ".join(why)
        if why:
            flagged[env].append(cid)
    wide.append(w)
with (outdir / "envcmp_wide.csv").open("w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(wide[0].keys()))
    wr.writeheader()
    wr.writerows(wide)
for env, lst in flagged.items():
    (outdir / f"flagged_{env}_vs_A.json").write_text(json.dumps({"condition_ids": lst, "rule": {
        "res_mib": RES_MIB, "res_rel": RES_REL, "alloc_mib": ALLOC_MIB, "outside_mib": OUT_MIB}}, indent=1))

md = [f"# 환경 비교 요약 ({len(ids)}개 조건)", "", "| env | label | driver | torch | cuDNN | ok | gpu_oom | 기타 |", "|---|---|---|---|---|---|---|---|"]
for env in envs:
    xs = [L[(env, c)] for c in ids]
    st = [x["status"] for x in xs]
    v = lambda k: sorted({str(x[k]) for x in xs if x[k] is not None})  # noqa: E731
    md.append(f"| {env} | {envs[env]} | {', '.join(v('driver'))} | {', '.join(v('torch'))} | {', '.join(v('cudnn'))} | "
              f"{st.count('ok')} | {st.count('gpu_oom')} | {len(st) - st.count('ok') - st.count('gpu_oom')} |")
for env in stats:
    md += ["", f"## {env} − A (MiB, 양쪽 값이 있는 조건만)", "", "| 지표 | n | 0인 조건 | 최소 | 중앙값 | 최대 |", "|---|---|---|---|---|---|"]
    for m in METRICS:
        v = stats[env][m]
        if not v:
            md.append(f"| {m} | 0 | | | | |")
            continue
        sc = 1 if m == "num_alloc_retries" else MiB
        md.append(f"| {m} | {len(v)} | {sum(1 for x in v if x == 0)} | {min(v) / sc:+.1f} | {statistics.median(v) / sc:+.1f} | {max(v) / sc:+.1f} |")
    md += ["", f"눈에 띈 조건 ({len(flagged[env])}개, `flagged_{env}_vs_A.json`):", ""]
    md += [f"- `{c}`: {next(w[f'flag_{env}_vs_A'] for w in wide if w['condition_id'] == c)}" for c in flagged[env]] or ["- 없음"]
if missing:
    md += ["", "## 값 누락 (둘 다 ok인데 지표가 없음)", ""] + [f"- {m}" for m in missing]
(outdir / "envcmp_summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

"""New LLMs (48 conditions): AWS r7i fp16_real (real FP16 CPU execution) vs INUOS (FP32 + FP16 size shadow), both v3
R_at(RTX 3090 effective capacity) reserved, against RTX 3090 m1_peak_reserved. usage: python3 compare.py <aws extract dir>"""
import csv, json, sys, collections
from pathlib import Path
E = Path(__file__).resolve().parents[4]
A = Path(sys.argv[1])
M = 1 << 20
meas = {r["condition_id"]: int(r["m1_peak_reserved"]) for r in csv.DictReader(open(E / "results/reserved_v3/newmodels_eval/gpu_rtx3090_newmodels_20261122_per_condition.csv"))}
def load(files):
    d = {}
    for f in files:
        for l in open(f):
            r = json.loads(l)
            if r.get("status") == "ok":
                d[r["case_id"]] = r["main"]["R_at"]["effective"]["reserved"]
    return d
ino = load([E / "results/reserved_v3/newmodels_rtx3090/predictions.jsonl", E / "results/reserved_v3/newmodels_rtx3090_retry2/predictions.jsonl"])
aws = load([A / "results/reserved_v3/aws_fp16_newmodels/predictions.jsonl"])
rows = []
for c in sorted(aws):
    rows.append({"case_id": c, "measured": meas[c], "inuos": ino[c], "aws": aws[c],
                 "err_inuos_MiB": round((ino[c] - meas[c]) / M, 1), "err_aws_MiB": round((aws[c] - meas[c]) / M, 1)})
out = {}
for mdl in sorted({r["case_id"].split("__")[0] for r in rows}) + ["all"]:
    rr = [r for r in rows if mdl == "all" or r["case_id"].startswith(mdl + "__")]
    s = {}
    for k in ("inuos", "aws"):
        e = [r[f"err_{k}_MiB"] for r in rr]
        s[k] = {"n": len(e), "within100MB": sum(abs(x) * M <= 1e8 for x in e), "exact": sum(x == 0 for x in e),
                "max_over": max(e), "max_under": min(e)}
    out[mdl] = s
    print(mdl, s)
print("conditions where they differ:")
for r in rows:
    if r["err_inuos_MiB"] != r["err_aws_MiB"]:
        print(" ", r["case_id"], "INUOS", r["err_inuos_MiB"], "AWS", r["err_aws_MiB"])
json.dump({"summary": out, "rows": rows}, open(Path(__file__).parent / "compare.json", "w"), indent=1)

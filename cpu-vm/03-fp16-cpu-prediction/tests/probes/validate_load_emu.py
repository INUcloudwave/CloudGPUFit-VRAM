"""Validate cgf680.load_emu against (a) GPU load-phase traces (exact sequence) and (b) GPU after-load reserved
/ load peak allocated of all 13 pilot models (RTX 3090, default attention)."""
import sys, json, csv, glob
sys.path.insert(0, "/diag/src"); sys.path.insert(0, "/exp/src")
import torch, transformers
from cgf_diag.events import load_snapshot
from cgf680 import load_emu
from cgf_diag.replay import Replayer
M = json.load(open("/handoff/config/models.json"))["models"]
ref = {r["condition_id"]: r for r in csv.DictReader(open("/handoff/reference/pilot_rtx3090_eager_vs_default.csv"))}
def gseq(cid):
    d = f"/handoff/reference/traces_attn_default/{cid}"
    t = json.load(open(f"{d}/trace.json")); end = next(x for x in t["history"]["boundaries"] if x["name"] == "model_load_end")
    ev = load_snapshot(f"{d}/trace.snapshot.pickle.gz")["device_traces"][0]; g = []; a = f = 0
    for e in ev:
        if a >= end["allocation.all.allocated"] and f >= end["allocation.all.freed"]: break
        if e["action"] == "alloc": a += 1; g.append(e["size"])
        elif e["action"] == "free_completed": f += 1; g.append(-e["size"])
    return g
traced = {p.split("/")[-1] for p in glob.glob("/handoff/reference/traces_attn_default/*")}
out = []
for mid, m in M.items():
    snap = glob.glob(f"/exp/hf_cache/hub/models--{mid.replace('/', '--')}/snapshots/{m['revision']}")[0]
    model = getattr(transformers, m["loader"]).from_pretrained(mid, revision=m["revision"], torch_dtype=torch.float16,
            low_cpu_mem_usage=True, use_safetensors=m["use_safetensors"], local_files_only=True)
    seq, st = load_emu.synthesize(model, snap, m["use_safetensors"])
    evs, live = load_emu.to_events(seq)
    rp = Replayer(1 << 40, keep_decisions=False)
    peak_a = 0
    for i, (a, s, aid) in enumerate(evs):
        rp.apply({"seq": i, "action": "alloc" if a else "free", "size": s, "stream": 0, "alloc_id": aid, "phase": "load", "invocation": None})
        peak_a = max(peak_a, rp.allocated)
    cid = next(c for c in ref if c.startswith(m["short"] + "__"))
    r = {"model": m["short"], "syn_events": len(evs), "load_R_MiB_emu": rp.reserved / 2**20, "load_R_MiB_gpu": float(ref[cid]["default_load_reserved_MiB"]),
         "load_pkA_MiB_emu": round(peak_a / 2**20, 1), "load_pkA_MiB_gpu": float(ref[cid]["default_load_peak_allocated_MiB"]),
         "alloc_end_MiB_emu": round(rp.allocated / 2**20, 1), "warmup": st["warmup_bytes"], "skipped": st["ckpt_skipped_unexpected"], "tied_dup": st["tied_dup_loaded_then_freed"]}
    tc = [c for c in traced if c.startswith(m["short"] + "__")]
    if tc:
        g = gseq(tc[0]); s = load_emu.signed(seq)
        r["trace_exact_sequence"] = g == s
        if g != s:
            k = next((i for i, (x, y) in enumerate(zip(g, s)) if x != y), min(len(g), len(s)))
            r["first_diff"] = {"i": k, "gpu": g[max(0, k - 3):k + 4], "syn": s[max(0, k - 3):k + 4], "len": [len(g), len(s)]}
    print(json.dumps(r), flush=True)
    del model

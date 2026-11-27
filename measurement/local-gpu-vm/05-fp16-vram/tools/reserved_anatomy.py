#!/usr/bin/env python3
"""Reserved-memory anatomy of one GPU allocator trace (trace mode output of vram_worker.py). CPU only, no torch needed.

  python3 -I reserved_anatomy.py <trace.json> <hf_hub_dir> <models.json> <out.json>

What it reports (all from the GPU trace, nothing simulated):
  - reserved / allocated at every worker boundary (load, warmup_0..2, measure_0)
  - every segment_alloc (= the moment reserved grows): window, segment size, and the request that triggered it,
    classified by matching the request size against shape formulas from the model config
    (weights, KV-cache tensor of length s, full prefill logits, last-token logits, other)
  - reserved growth per window x category, and the slack (reserved - allocated) at each boundary
Classification is by exact byte size only; a size that matches no formula is reported as "other:<size>".
"""
import gzip
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

trace_json, hub, models_json, out_json = sys.argv[1:5]
T = json.loads(Path(trace_json).read_text())
cond = T["condition"]
models = json.loads(Path(models_json).read_text())["models"]
ent = models[cond["model"]]
cfgp = next(Path(hub).glob(f"models--{cond['model'].replace('/', '--')}/snapshots/{ent['revision']}/config.json"))
C = json.loads(cfgp.read_text())
hist = T["history"]
ev = pickle.load(gzip.open(Path(trace_json).with_name(hist["snapshot_file"]), "rb"))["device_traces"][0]
bounds = hist["boundaries"]
MiB = 2 ** 20

# ---- size formulas (bytes, fp16 = 2 bytes) -------------------------------------------------------------------
kind, B = cond["workload_type"], cond["batch_size"]
sig = {}
if kind in ("decoder_llm", "text_encdec"):
    L, N = cond["input_len"], cond["max_new_tokens"]
    V = C["vocab_size"]
    if kind == "decoder_llm":
        nh = C.get("num_attention_heads") or C.get("n_head")
        h = C.get("hidden_size") or C.get("n_embd")
        kvh = C.get("num_key_value_heads", nh)
        hd = C.get("head_dim") or h // nh
        ffn = C.get("intermediate_size") or C.get("n_inner") or C.get("ffn_dim") or 4 * h
        for s in range(1, L + N + 1):
            sig.setdefault(B * kvh * hd * s * 2, f"kv_len{s}" if s != L else "kv_prefill")
        decode_marker = B * kvh * hd * (L + 1) * 2
        if kvh != nh:  # GQA: sdpa path expands K/V to all heads (repeat_kv) each step
            for s_ in range(1, L + N + 1):
                sig.setdefault(B * nh * hd * s_ * 2, f"kvrep_len{s_}")
        for nm, sz in (("act_hidden", B * L * h * 2), ("act_ffn", B * L * ffn * 2), ("act_qkv", B * L * 3 * h * 2),
                       ("act_ffn_fp32", B * L * ffn * 4), ("act_hidden_fp32", B * L * h * 4)):
            sig.setdefault(sz, nm)
    else:  # T5: decoder self-attn grows 1..N, cross-attn fixed at L
        nh, hd = C["num_heads"], C["d_kv"]
        for s in range(1, N + 2):
            sig.setdefault(B * nh * hd * s * 2, f"self_kv_len{s}")
        sig[B * nh * hd * L * 2] = "cross_kv"
        decode_marker = B * nh * hd * 2 * 2
        dm, dff = C["d_model"], C["d_ff"]
        for nm, sz in (("enc_act_hidden", B * L * dm * 2), ("enc_act_ffn", B * L * dff * 2),
                       ("enc_attn_scores", B * nh * L * L * 2), ("enc_act_hidden_fp32", B * L * dm * 4)):
            sig.setdefault(sz, nm)
    sig[B * L * V * 2] = "logits_prefill_full_fp16"
    sig[B * L * V * 4] = "logits_prefill_full_fp32"
    sig[B * V * 2] = "logits_last_fp16"
    sig[B * V * 4] = "logits_last_fp32"


decode_marker = locals().get("decode_marker")


def cat(size, window):
    if window == "model_load":
        return "weights_or_load"
    lab = sig.get(size)
    if lab is None:
        return "other"
    if lab.startswith("kv_len") or lab.startswith("self_kv_len"):
        return "kv_growth"
    if lab.startswith("kvrep_len"):
        return "kv_repeat_growth"
    return lab


# ---- walk events, map boundaries ------------------------------------------------------------------------------
n_alloc = n_free = 0
phase = ["prefill"]
bi = 0
win = "pre_load"
res = alloc = 0
bstate = []
seg_rows = []
pending_seg = None
growth = defaultdict(lambda: defaultdict(int))
seg_free = defaultdict(int)
other_sizes = Counter()


def at_boundary():
    global bi, win
    while bi < len(bounds) and n_alloc == bounds[bi]["allocation.all.allocated"] and \
            n_free == bounds[bi]["allocation.all.freed"]:
        nm = bounds[bi]["name"]
        bstate.append({"name": nm, "reserved_MiB": round(res / MiB, 1), "allocated_MiB": round(alloc / MiB, 1),
                       "slack_MiB": round((res - alloc) / MiB, 1),
                       "gpu_counter_reserved_MiB": round(bounds[bi]["reserved_bytes.all.current"] / MiB, 1)})
        win = nm[: -len("_begin")] if nm.endswith("_begin") else "between:" + nm[: -len("_end")]
        phase[0] = "prefill"
        bi += 1


at_boundary()
live = {}
for e in ev:
    a = e["action"]
    if a == "segment_alloc":
        res += e["size"]
        pending_seg = e
    elif a == "segment_free":
        res -= e["size"]
        seg_free[win] += e["size"]
    elif a == "alloc":
        sz = int(e["size"])
        alloc += sz
        live[e["addr"]] = sz
        n_alloc += 1
        if decode_marker is not None and sz == decode_marker and win.startswith(("warmup", "measure")):
            phase[0] = "decode"
        if pending_seg is not None:
            c = cat(sz, win)
            key = win if not win.startswith(("warmup", "measure")) or decode_marker is None else f"{win}/{phase[0]}"
            growth[key][c] += pending_seg["size"]
            if c == "other":
                other_sizes[sz] += 1
            seg_rows.append({"window": win, "phase": phase[0], "segment_MiB": round(pending_seg["size"] / MiB, 2),
                             "request_bytes": sz, "category": c, "label": sig.get(sz)})
            pending_seg = None
    elif a == "free_completed":
        alloc -= live.pop(e["addr"], 0)
        n_free += 1
    at_boundary()

out = {"condition_id": cond["condition_id"], "model_config": str(cfgp.name),
       "boundaries": bstate,
       "reserved_growth_MiB_by_window": {w: {c: round(v / MiB, 1) for c, v in d.items()} for w, d in growth.items()},
       "segment_allocs": len(seg_rows),
       "segment_free_MiB_by_window": {w: round(v / MiB, 1) for w, v in seg_free.items()},
       "top_other_trigger_sizes": [{"bytes": s, "count": n} for s, n in other_sizes.most_common(8)],
       "segments": seg_rows}
Path(out_json).write_text(json.dumps(out, indent=1))
print(cond["condition_id"])
for b in bstate:
    print(f"  {b['name']:<18} reserved {b['reserved_MiB']:>9} allocated {b['allocated_MiB']:>9} slack {b['slack_MiB']:>9}"
          f"  (gpu counter {b['gpu_counter_reserved_MiB']})")
for w, d in out["reserved_growth_MiB_by_window"].items():
    print(f"  growth[{w}]: {d}")
print(f"  segment_free (cache released, e.g. OOM-retry): {out['segment_free_MiB_by_window']}")
print(f"  segment_allocs {len(seg_rows)}; other trigger sizes: {out['top_other_trigger_sizes'][:4]}")

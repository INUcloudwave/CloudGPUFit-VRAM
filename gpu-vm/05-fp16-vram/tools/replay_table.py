#!/usr/bin/env python3
"""Markdown table: GPU trace vs xMem replay, eager vs default attention (observed values)."""
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); M = 1 << 20
ids = json.loads((root / "config/trace_pilot_v2.json").read_text())["condition_ids"]
print("| 조건 | 실행 설정(실제 attention) | 경계 위치 확인 | 짝 없는 free | GPU/xMem segment 수 | 로드 후 reserved GPU/xMem | m1 시작 reserved GPU/xMem | m1 피크 reserved GPU/xMem | m1 피크 allocated GPU/xMem | m1 구간 alloc 요청 수 | 전체 alloc 이벤트 |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for cid in ids:
    for tag, lab in (("eager", "rtx3090"), ("default", "rtx3090__attn-default")):
        rp = root / f"results/xmem_replay_rtx3090/{tag}/{cid}.json"
        tp = root / f"runs/{lab}/{cid}/trace.json"
        if not rp.exists():
            print(f"| {cid} | {tag} | 없음 |||||||||"); continue
        r, t = json.loads(rp.read_text()), json.loads(tp.read_text())
        b = {x["name"]: x for x in t["history"]["boundaries"]}
        n_m1 = b["measure_0_end"]["allocation.all.allocated"] - b["measure_0_begin"]["allocation.all.allocated"]
        w = {x["window"]: x for x in r["windows"]}["measure_0"]
        f = lambda x: f"{x / M:.1f}"
        print(f"| {cid} | {'eager 고정' if tag == 'eager' else '기본값'}({t['model']['attn_implementation']}) | {r['boundaries_located']}/{r['boundaries_total']} | {r['missing_frees']} | "
              f"{r['gpu_segment_alloc_events']}/{r['xmem_segment_alloc_count']} | {f(r['after_load']['gpu_reserved'])}/{f(r['after_load']['xmem_reserved'])} | "
              f"{f(w['gpu_start_reserved'])}/{f(w['xmem_start_reserved'])} | {f(w['gpu_peak_reserved'])}/{f(w['xmem_peak_reserved'])} | "
              f"{f(w['gpu_peak_allocated'])}/{f(w['xmem_peak_allocated'])} | {n_m1} | {r['gpu_alloc_events']} |")

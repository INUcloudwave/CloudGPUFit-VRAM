# observed values only: per run status / checks / phase_end current & peak (MiB)
import json, sys, glob, os
root = sys.argv[1]
M = 1 << 20
for meta_p in sorted(glob.glob(f"{root}/*/*/run_meta.json")):
    d = os.path.dirname(meta_p); m = json.load(open(meta_p))
    tc = m.get("trace_check", {})
    print(f"## {m['case_id']} {m['run_id']} status={m.get('status')} start_empty={m['record_start_check']['empty']} "
          f"truncated={m.get('truncated')} alloc_ev={tc.get('action_counts',{}).get('alloc')} ctr={tc.get('counter_delta_allocation_all_allocated')} "
          f"seg_ev={tc.get('action_counts',{}).get('segment_alloc')} segctr={tc.get('counter_delta_segment_all_allocated')} entries={m.get('snapshot_trace_entries')}")
    inv = [json.loads(l) for l in open(f"{d}/invocations.jsonl")]
    toks = {(i['phase'], i['invocation']): i.get('actual_new_tokens_per_sample') for i in inv}
    for l in open(f"{d}/boundaries.jsonl"):
        b = json.loads(l)
        if b["kind"] not in ("phase_end", "record_end", "empty_cache_before"): continue
        s = b["memory_stats"]
        print(f"| {b['kind']} | {b['phase']} | {b['invocation']} | {s['reserved_bytes.all.current']/M:.1f} | {s['reserved_bytes.all.peak']/M:.1f} | "
              f"{s['allocated_bytes.all.current']/M:.1f} | {s['allocated_bytes.all.peak']/M:.1f} | {s['segment.all.current']} | "
              f"{s['num_alloc_retries']} | {s['num_ooms']} | {toks.get((b['phase'], b['invocation']), '')} |")

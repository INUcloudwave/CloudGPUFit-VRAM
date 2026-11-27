"""CPU tensor-request trace of a GPU-equivalent execution history (diagnostic, not production).

Mirrors the per-case GPU worker (bench_worker_v3.py, profile local_core, hybrid isolation) step by
step: finite-output check forward -> warmup generate(s) -> measured generate(s), optionally followed
by the memory-mode block (between-mode gc.collect, finite check, warmup, measured). Each step is a
separate torch.profiler window so RAM stays bounded; one Pairer spans all windows so allocations are
matched to their frees across windows. Allocation inside a step's record_function region = request.

GPU-equivalence choices (each recorded in meta.json):
  * inputs are generated exactly like the GPU worker (seed 20261006, randint(3, vocab), special
    token ids replaced by 3) and kept host-side; each step clones them (GPU: value.to("cuda")).
  * output.cpu() is a no-op on CPU (GPU: device->host copy, no device allocation).
  * automatic Python GC is disabled; gc.collect() is run only where the GPU worker runs it (between
    modes), inside a profiled window. Objects collected are reported.
  * model weights: from_config (production CPU path) or pretrained (local HF cache) -- selectable.
The model-loading allocation history of the GPU (from_pretrained to cuda) is NOT reproduced here; the
analysis places resident storages synthetically (production method) or uses a GPU-recorded load phase.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import sys
import tempfile
import time
from pathlib import Path

import torch

from cgf_diag.events import cpu_memory_events_from_chrome, normalize_cpu, order_report, write_jsonl, Pairer, open_text

SEQUENCES = {
    # name: list of (mode, step, count)
    "aws_replica_local_core": [("performance", "finite_check", 1), ("performance", "warmup", 1), ("performance", "measure", 5),
                               ("memory", "between_modes", 1), ("memory", "finite_check", 1), ("memory", "warmup", 1),
                               ("memory", "measure", 3)],
    "controlled_f1_w1_m3": [("controlled", "finite_check", 1), ("controlled", "warmup", 1), ("controlled", "measure", 3)],
    "single_generate": [("single", "measure", 1)],
}


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def gpu_worker_inputs(config, batch, length):
    g = torch.Generator().manual_seed(20261006)
    ids = torch.randint(3, config.vocab_size, (batch, length), generator=g)
    for name in ("pad_token_id", "eos_token_id", "bos_token_id"):
        tok = getattr(config, name, None)
        if isinstance(tok, int):
            ids[ids == tok] = 3
    return {"input_ids": ids, "attention_mask": torch.ones(batch, length, dtype=torch.long)}


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def current_rss():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-json", required=True, help="production <case>.input.json (request + model_config)")
    p.add_argument("--sequence", choices=sorted(SEQUENCES), required=True)
    p.add_argument("--weights", choices=["from_config", "pretrained"], default="from_config")
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--keep-raw-traces", action="store_true")
    p.add_argument("--expected-input-sha", default=None, help="input_ids sha256 recorded by the GPU run")
    p.add_argument("--max-steps", type=int, default=None, help="stop after N steps (smoke)")
    p.add_argument("--step-index", type=int, default=None,
                   help="run ONLY this step of the sequence in this process (one profiler session per process); "
                        "output goes to <out>/steps/<KK>/ and is combined by cgf_diag.merge_steps")
    p.add_argument("--mem-log", default=None, help="sample VmRSS every 0.5 s with the current step label")
    a = p.parse_args()
    label = {"step": "init"}
    if a.mem_log:
        import threading

        def sampler():
            with open(a.mem_log, "a") as f:
                while True:
                    f.write(json.dumps({"t": time.time(), "step": label["step"], "rss": current_rss()}) + "\n")
                    f.flush()
                    time.sleep(0.5)
        threading.Thread(target=sampler, daemon=True).start()

    from inference_trace_runtime import iter_trace_events
    import predict_vram
    import transformers

    out = Path(a.out)
    if a.step_index is not None:
        out = out / "steps" / f"{a.step_index:02d}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw_traces").mkdir(exist_ok=True)
    spec = json.loads(Path(a.input_json).read_text())
    req = spec["request"]
    torch.manual_seed(20261006)
    torch.set_num_threads(a.threads)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[req["dtype"]]

    t0 = time.time()
    if a.weights == "from_config":
        model, kind, cfg = predict_vram.load_model(req["model"], req.get("model_config"), req["workload_type"], req["dtype"])
    else:
        cls = getattr(transformers, predict_vram.WORKLOAD_MODEL_CLASS[req["workload_type"]][0])
        model = cls.from_pretrained(req["model"], revision=spec["revision"], torch_dtype=dtype, low_cpu_mem_usage=True,
                                    use_safetensors=True, local_files_only=True, attn_implementation="eager").to(dtype=dtype).eval()
        cfg = model.config
    model.eval()
    model.config.output_attentions = False
    model.config.output_hidden_states = False
    load_seconds = time.time() - t0

    if req["workload_type"] != "decoder_llm":
        raise SystemExit("this diagnostic currently covers decoder_llm cases only")
    batch, length, new_tokens = req["batch_size"], req["seq_len"], req["max_new_tokens"]
    inputs = gpu_worker_inputs(model.config, batch, length)
    input_sha = {k: sha(v) for k, v in inputs.items()}
    if a.expected_input_sha and input_sha["input_ids"] != a.expected_input_sha:
        raise SystemExit(f"input_ids sha mismatch: {input_sha['input_ids']} != {a.expected_input_sha}")
    pad = model.config.pad_token_id if model.config.pad_token_id is not None else model.config.eos_token_id

    resident_sizes = predict_vram.storage_sizes(list(model.parameters()) + list(model.buffers()))
    gc.disable()
    gc.collect()

    def step_fn(step):
        def finite_check():
            with torch.inference_mode():
                probe = model(**{k: v.clone() for k, v in inputs.items()})
                finite = bool(torch.isfinite(probe.logits).all().item())
                del probe
            return {"finite": finite}

        def generate():
            with torch.inference_mode():
                gpu_inputs = {k: v.clone() for k, v in inputs.items()}
                output = model.generate(**gpu_inputs, do_sample=False, num_beams=1, use_cache=True,
                                        min_new_tokens=new_tokens, max_new_tokens=new_tokens, pad_token_id=pad,
                                        return_dict_in_generate=False, output_scores=False, output_attentions=False,
                                        output_hidden_states=False, streamer=None)
                output.cpu()
                produced = output.shape[1] - length
            del output, gpu_inputs
            if produced != new_tokens:
                raise RuntimeError(f"actual new tokens {produced} != {new_tokens}")
            return {"actual_new_tokens_per_sample": produced}

        def between_modes():
            return {"gc_collected_objects": gc.collect()}

        return {"finite_check": finite_check, "warmup": generate, "measure": generate, "between_modes": between_modes}[step]

    pairer = Pairer()
    seq0 = 0
    windows = []
    events_path = out / "events.jsonl.gz"
    ev_out = open_text(events_path, "wt")
    steps = [(m, s, i) for (m, s, n) in SEQUENCES[a.sequence] for i in range(n)]
    if a.max_steps:
        steps = steps[:a.max_steps]
    all_steps = steps
    selected = list(enumerate(steps))
    if a.step_index is not None:
        selected = [selected[a.step_index]]
    for index, (mode, step, inv) in selected:
        region = f"cgf::{index:02d}::{mode}::{step}::{inv}"
        fn = step_fn(step)
        label["step"] = f"{index}:{mode}/{step}/{inv}:run"
        tw = time.time()
        prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True,
                                      record_shapes=False, with_stack=False)
        prof.start()
        try:
            with torch.profiler.record_function(region):
                info = fn()
        finally:
            prof.stop()
        run_s = time.time() - tw
        label["step"] = f"{index}:{mode}/{step}/{inv}:export_parse"
        raw = out / "raw_traces" / f"{index:02d}_{mode}_{step}_{inv}.json"
        prof.export_chrome_trace(str(raw))
        del prof
        regions, mem = cpu_memory_events_from_chrome(raw, "cgf::", iter_trace_events=iter_trace_events)
        rows, stats, pairer = normalize_cpu(mem, regions, start_seq=seq0, pairer=pairer,
                                            phase_of=lambda name, m=mode, s=step, i=inv: (f"{m}/{s}", i))
        for r in rows:
            ev_out.write(json.dumps(r, separators=(",", ":")) + "\n")
        seq0 += len(rows)
        orep = order_report(mem)
        if a.keep_raw_traces:
            import gzip, shutil
            with raw.open("rb") as src, gzip.open(str(raw) + ".gz", "wb", compresslevel=1) as dst:
                shutil.copyfileobj(src, dst, 1 << 22)
        raw.unlink()
        w = {"index": index, "region": region, "mode": mode, "step": step, "invocation": inv, "seconds": run_s,
             "requests": sum(1 for r in rows if r["action"] == "alloc"), "frees": sum(1 for r in rows if r["action"] == "free"),
             "request_bytes": sum(r["size"] for r in rows if r["action"] == "alloc"), "order": orep, "pairing": stats,
             "live_requests_after_step": len(pairer.live), "max_rss_bytes": rss(), **info}
        del mem, rows
        # Profiler result objects form reference cycles; with automatic GC disabled they leaked one
        # profiler session per window (medium fp32 run killed at 36 GiB, 2026-11-10). Collect them
        # OUTSIDE the profiled window, but only when no traced request is live, so no request free can
        # be lost; otherwise skip and record it.
        if not pairer.live:
            w["gc_outside_window_collected"] = gc.collect()
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
        else:
            w["gc_outside_window_collected"] = "skipped_live_requests"
        w["rss_after_cleanup_bytes"] = current_rss()
        windows.append(w)
        print("CGF_STEP " + json.dumps(w), flush=True)
    ev_out.close()
    gc.enable()

    meta = {"case_id": spec["case_id"], "revision": spec["revision"], "request": {k: v for k, v in req.items() if k != "model_config"},
            "sequence": a.sequence, "steps": [f"{m}/{s}/{i}" for (m, s, i) in all_steps], "weights": a.weights,
            "step_index": a.step_index,
            "process_policy": "single process, one profiler window per step" if a.step_index is None
                              else "one fresh process per step (only this step executed in the process)",
            "model_config_sha256": hashlib.sha256(json.dumps(req.get("model_config"), sort_keys=True).encode()).hexdigest(),
            "input_sha256": input_sha, "input_shapes": {k: list(v.shape) for k, v in inputs.items()},
            "input_policy": "gpu_worker_equivalent: seed 20261006 randint(3,vocab) special->3; host-resident; cloned per step",
            "resident_storage_sizes": resident_sizes, "resident_storage_bytes": sum(resident_sizes),
            "gc_policy": "automatic GC disabled; gc.collect inside the between_modes window (GPU worker point) and "
                         "outside profiled windows only when no traced request is live (profiler cleanup)",
            "region_rule": "alloc inside cgf:: region = request; frees matched anywhere in any window",
            "environment": {"torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
                            "threads": torch.get_num_threads(), "attention": "eager", "device": "cpu"},
            "model_load_seconds": load_seconds, "total_seconds": time.time() - t0, "max_rss_bytes": rss(),
            "unfreed_requests_at_end": len(pairer.live), "events_file": events_path.name}
    (out / "windows.json").write_text(json.dumps(windows, indent=1))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("CGF_DONE " + json.dumps({"out": str(out), "events": seq0, "seconds": meta["total_seconds"]}), flush=True)


if __name__ == "__main__":
    sys.exit(main())

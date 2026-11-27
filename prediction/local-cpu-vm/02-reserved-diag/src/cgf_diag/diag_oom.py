"""Diagnostic for the RSS blow-up seen in the 3rd profiler window of the medium fp32 history trace.
Runs N generates with or without a per-generate profiler window and logs VmRSS every 0.5 s."""
import argparse
import gc
import json
import threading
import time
from pathlib import Path

import torch

from cgf_diag.cpu_history_trace import gpu_worker_inputs, current_rss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--profile", choices=["none", "memory", "nomemory"], default="none")
    ap.add_argument("--gc", choices=["disabled", "enabled"], default="disabled")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import predict_vram
    spec = json.loads(Path(a.input_json).read_text())
    req = spec["request"]
    torch.manual_seed(20261006)
    torch.set_num_threads(4)
    model, _, _ = predict_vram.load_model(req["model"], req.get("model_config"), req["workload_type"], req["dtype"])
    inputs = gpu_worker_inputs(model.config, req["batch_size"], req["seq_len"])
    n_new = req["max_new_tokens"]
    label = {"s": "init"}
    log = open(a.out, "w")

    def sampler():
        while True:
            log.write(json.dumps({"t": time.time(), "s": label["s"], "rss": current_rss()}) + "\n")
            log.flush()
            time.sleep(0.5)
    threading.Thread(target=sampler, daemon=True).start()
    if a.gc == "disabled":
        gc.disable()
    for i in range(a.n):
        label["s"] = f"gen{i}"
        prof = None
        if a.profile != "none":
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                          profile_memory=(a.profile == "memory"), record_shapes=False, with_stack=False)
            prof.start()
        with torch.inference_mode():
            x = {k: v.clone() for k, v in inputs.items()}
            out = model.generate(**x, do_sample=False, num_beams=1, use_cache=True, min_new_tokens=n_new,
                                 max_new_tokens=n_new, pad_token_id=model.config.eos_token_id)
        del out, x
        if prof is not None:
            prof.stop()
            del prof
        label["s"] = f"after{i}"
        log.write(json.dumps({"event": "gen_done", "i": i, "gc_count": gc.get_count(), "rss": current_rss(),
                              "objects": len(gc.get_objects())}) + "\n")
        gc.collect()
        time.sleep(1)
    label["s"] = "end"
    time.sleep(1)


if __name__ == "__main__":
    main()

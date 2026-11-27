"""Build manifest/common_manifest.json from the production inputs and the previous L4 results (host python)."""
import glob
import hashlib
import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]
CASES = ["core__decoder_llm__small__fp16__01", "core__decoder_llm__small__fp16__12", "core__decoder_llm__medium__fp32__12"]
ROLES = {CASES[0]: "basic comparison", CASES[1]: "large reserved error", CASES[2]: "workspace and large error candidate"}
L4 = EXP / "inputs/aws_g6_2xlarge/results"
PKG = Path("/home/inuos/CloudGPUFit_CPU_predictions_20261110/source")


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def l4(case, mode):
    return json.loads(Path(glob.glob(str(L4 / f"*/{case}__{mode}__r0.json"))[0]).read_text())


def main():
    revisions = json.loads((PKG / "revisions.json").read_text())
    cases = []
    for c in CASES:
        inp = json.loads((EXP / f"inputs/cpu_production/{c}.input.json").read_text())
        pred = json.loads((EXP / f"inputs/cpu_production/{c}.prediction.json").read_text())
        req = inp["request"]
        perf, mem = l4(c, "performance"), l4(c, "memory")
        cases.append({
            "case_id": c, "role": ROLES[c], "workload_type": req["workload_type"], "model": req["model"],
            "revision": inp["revision"], "use_safetensors": revisions[req["model"]]["safetensors"],
            "model_config_sha256_cpu_input": hashlib.sha256(json.dumps(req["model_config"], sort_keys=True).encode()).hexdigest(),
            "dtype": req["dtype"], "batch_size": req["batch_size"], "seq_len": req["seq_len"], "max_new_tokens": req["max_new_tokens"],
            "inputs": {"recipe": "g=torch.Generator().manual_seed(20261006); input_ids=torch.randint(3, config.vocab_size, (batch, seq_len), generator=g); "
                                 "input_ids[input_ids==tok]=3 for tok in (pad,eos,bos) if int; attention_mask=torch.ones(batch, seq_len, dtype=torch.long). "
                                 "Created on CPU, moved with .to('cuda') inside every invocation.",
                       "expected_input_ids_sha256": mem["input_sha256"]["input_ids"],
                       "expected_attention_mask_sha256": mem["input_sha256"]["attention_mask"],
                       "sha_definition": "sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()) of the CPU tensor"},
            "generate_kwargs": {"do_sample": False, "num_beams": 1, "use_cache": True, "min_new_tokens": req["max_new_tokens"],
                                "max_new_tokens": req["max_new_tokens"], "pad_token_id": "config.pad_token_id or config.eos_token_id (50256)",
                                "return_dict_in_generate": False, "output_scores": False, "output_attentions": False, "output_hidden_states": False},
            "model_load": {"class": "AutoModelForCausalLM", "kwargs": {"revision": inp["revision"], "torch_dtype": req["dtype"], "low_cpu_mem_usage": True,
                           "device_map": {"": "cuda:0"}, "use_safetensors": revisions[req["model"]]["safetensors"], "local_files_only": True,
                           "attn_implementation": "eager"}, "then": ".to(device='cuda', dtype=dtype).eval(); config.output_attentions=False; config.output_hidden_states=False"},
            "cpu_production_baseline": {"source": "CloudGPUFit_CPU_predictions_20261110 (prediction_version 4-native-inference-single-request)",
                                        "predicted_allocated_bytes": pred["predicted_allocated_bytes"], "predicted_reserved_bytes": pred["predicted_reserved_bytes"],
                                        "conditions": "from_config random weights, warmup=0, measured=1, empty allocator + synthetic resident weights+inputs, xMem 1ee4bd0, gpu_memory_gb=80"},
            "previous_L4_reference_only": {
                "note": "g6.2xlarge 2026-11-02 run, profile local_core, hybrid isolation. Reference values, NOT targets for a new GPU.",
                "performance": {"after_load_allocated": perf["memory_baseline"]["resident_allocated_bytes"], "after_load_reserved": perf["memory_baseline"]["resident_reserved_bytes"],
                                "startup_peak_allocated": perf["startup_peak_allocated_bytes"], "startup_peak_reserved": perf["startup_peak_reserved_bytes"],
                                "inference_peak_allocated": perf["inference_peak_allocated_bytes"], "inference_peak_reserved": perf["inference_peak_reserved_bytes"]},
                "memory": {"at_mode_start_allocated": mem["memory_baseline"]["resident_allocated_bytes"], "at_mode_start_reserved": mem["memory_baseline"]["resident_reserved_bytes"],
                           "inference_peak_allocated": mem["inference_peak_allocated_bytes"], "inference_peak_reserved": mem["inference_peak_reserved_bytes"]},
                "environment": mem["environment"]},
        })
    runs = [
        {"run_id": "R0_aws_replica", "group": "reference", "purpose": "Reproduce the previous AWS per-case structure (bench_worker_v3, local_core, hybrid, allocator cache retained).",
         "steps": [["model_load", 1], ["performance/finite_check", 1], ["performance/warmup", 1], ["performance/measure", 5],
                   ["memory/between_modes", 1], ["memory/finite_check", 1], ["memory/warmup", 1], ["memory/measure", 3]],
         "notes": ["model_load = from_pretrained(..., device_map cuda:0).to(cuda,dtype).eval() in the same process",
                   "between_modes = torch.cuda.synchronize(); gc.collect(); NO empty_cache (keep_allocator_cache=True); model reused; config/generation_config deep-copied back; CPU/CUDA RNG restored",
                   "torch.cuda.reset_peak_memory_stats() before model load and before each measurement block (peak reset only, not a cache release)",
                   "tokens/components modes ran after memory in the original and cannot affect memory-mode values: omitted"]},
        {"run_id": "R1_controlled", "group": "reference", "purpose": "CPU-matching controlled history. CPU trace = prefix of R0 (identical first 5 inference steps).",
         "steps": [["model_load", 1], ["controlled/finite_check", 1], ["controlled/warmup", 1], ["controlled/measure", 3]]},
        {"run_id": "R2_cold_single", "group": "reference", "purpose": "One generate after model load (no finite check, no warmup): separates load history from repeated-request history.",
         "steps": [["model_load", 1], ["single/measure", 1]]},
        {"run_id": "D1_cold_single_empty_cache", "group": "diagnostic_only", "purpose": "As R2 but torch.cuda.empty_cache() after model load. Closest GPU analogue of the CPU baseline initial state. NOT the original condition.",
         "steps": [["model_load", 1], ["empty_cache", 1], ["single/measure", 1]]},
    ]
    manifest = {
        "schema": "cgf-reserved-diag-manifest/1", "created": "2026-11-10", "experiment_root_on_inuos": str(EXP),
        "question": ["Q1 how much does differing initial cache / execution history change reserved",
                     "Q2 do xMem and PyTorch 2.6 allocator make the same decisions for the same requests",
                     "Q3 after matching history, do residual differences come from GPU-only requests (workspace etc.), intermediate tensors or free timing"],
        "reference_software": {"torch": "2.6.0+cu124", "transformers": "4.52.4", "python": "3.11.11", "cuda_runtime": "12.4",
                               "xmem": {"repo": "Stone-ResearchLife/xMem", "commit": "1ee4bd08dda277a6248abee5b6d40e0d6067efaa",
                                        "allocator_lock_sha256": sha_file(EXP / "provenance/xmem-allocator-lock.json")},
                               "previous_gpu_image_id": cases[0]["previous_L4_reference_only"]["environment"]["image_id"],
                               "previous_gpu_worker_sha256": sha_file(EXP / "inputs/aws_g6_2xlarge/bench_worker_v3.py")},
        "required_process_settings": {"CUDA_MODULE_LOADING": "LAZY (set explicitly before importing torch; record the value and, if possible, cuModuleGetLoadingMode)",
                                      "PYTORCH_CUDA_ALLOC_CONF": "unset (record actual)", "CUBLAS_WORKSPACE_CONFIG": "unset (record actual)",
                                      "CUBLASLT_WORKSPACE_SIZE": "unset (record actual)", "PYTORCH_NO_CUDA_MEMORY_CACHING": "unset",
                                      "torch.manual_seed": 20261006, "allow_tf32": False, "cudnn.benchmark": False, "cudnn.allow_tf32": False,
                                      "torch.set_num_threads": 8, "attention_implementation": "eager (HF attn_implementation; unrelated to CUDA module loading EAGER/LAZY)",
                                      "isolation": "single GPU, no other GPU process, fresh process per (case, run_id)"},
        "cases": cases, "runs": runs,
        "phase_naming": "phase = '<mode>/<step>' exactly as in runs[].steps; invocation = 0-based index inside that phase",
        "uncontrolled_or_differing_factors": [
            "CPU trace uses from_config random weights by default; GPU uses pretrained weights. Request sizes are not data dependent for these greedy, min_new_tokens=max_new_tokens runs; checked on CPU for gpt2 fp16_01 (pretrained vs from_config) -- see results.",
            "CPU kernels (ATen CPU/oneDNN) vs CUDA kernels may request different intermediates; cuBLAS/cuBLASLt workspaces exist only on GPU.",
            "CPU cannot reproduce the GPU model-load allocation history; it is taken from the GPU trace (hybrid replay) or placed synthetically.",
        ],
    }
    out = EXP / "manifest/common_manifest.json"
    out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(out, sha_file(out))


if __name__ == "__main__":
    main()

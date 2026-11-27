# Qwen outside extension — 2026-11-05

Purpose: extend scale and model architecture on the existing RTX 3090. No prediction model training.

Model: Qwen/Qwen2.5-7B, pinned revision 272aa6d30271028cdf113345b1e82a660437edaa.
Official file listing: https://huggingface.co/Qwen/Qwen2.5-7B/tree/272aa6d30271028cdf113345b1e82a660437edaa
About 15.2 GB repository; download selects JSON and safetensors only. Random token inputs need no tokenizer.
Existing GPT-2 revision is the anchor. FP16 B1 T256 N64, eager, four CPU threads, baseline image and existing probe.

Files are additive: do not overwrite original probe, plans, revisions or criteria.
Requires existing outside-lab toolkit. Extract ZIP into ~/cloudgpufit-outside.
The launcher verifies image ID and existing probe hash. Separate pins reach both cache preparation and GPU workers.
Before cache-download, minimum available cache filesystem space is 25 GiB. CPU preparation has no GPU access.
A partly completed download can be resumed; do not delete cache shards manually. If the guard stops a resume, inspect space before changing it.

## Current preparation

Inside tmux, from ~/cloudgpufit-outside:

```bash
set -o pipefail
python3 outside-lab/container_run_qwen.py cache-download --cache /home/inugpu/cloudgpufit-bench/hf-cache 2>&1 | tee qwen-download.log
```

Expected two JSON rows with status ok, for GPT-2 and Qwen. This checks file presence and indexed shard coverage, not independent byte checksums.

## Subsequent steps (only after preparation output is checked)

```bash
python3 outside-lab/container_run_qwen.py cache-check --cache /home/inugpu/cloudgpufit-bench/hf-cache
python3 outside-lab/container_run_qwen.py plan > qwen-schedule.json
```

Plan is 12 fresh processes: two models × direct/performance_first × three repeats, randomized within repeat blocks. The GPU environment remains one independent environment. Size and architecture both change.

Model smoke uses the full direct probe on Qwen, not merely CUDA initialization. Run inside tmux with a new output path:

```bash
python3 outside-lab/container_run_qwen.py smoke --cache /home/inugpu/cloudgpufit-bench/hf-cache --gpu-uuid GPU-17eb7299-8b39-914d-7c5b-26908502904f --env CUDA_MODULE_LOADING=LAZY --output results/qwen-smoke-01
```

Inspect smoke manifest for status ok and finite logits / generation length success before pilot:

```bash
set -o pipefail
python3 outside-lab/container_run_qwen.py pilot --cache /home/inugpu/cloudgpufit-bench/hf-cache --gpu-uuid GPU-17eb7299-8b39-914d-7c5b-26908502904f --env CUDA_MODULE_LOADING=LAZY --output results/qwen-01 2>&1 | tee qwen-01.log
python3 outside-lab/analyse_probe.py --input results/qwen-01/session --out results/qwen-01-analysis
```

Pilot timeout is 900 seconds per process; a failure stops the suite. Do not switch to BF16, quantization, offload or another backend to hide a failure. Preserve the original failure.

Analysis uses unchanged criteria.json. Keep raw NVML/reserved pairs and bracket-change flags. Compare checkpoint medians, process repeat ranges, history contrasts and GPT-2 anchor against large-01. Do not subtract independent maxima. Model loading has checkpoint endpoints, no continuous load sampler.

CPU validation: eight launcher tests passed (four existing large extension, four Qwen tests). No Docker or GPU execution was possible in this workspace; actual compatibility is established by the user-side smoke.

#!/usr/bin/env python3
"""
xMem 추론 워크로드 검증용 공용 벤치마크.

--device cpu  : PyTorch Profiler 트레이스(JSON) 생성 -> xMem 입력
--device cuda : NVML 1ms 샘플링으로 ground truth peak 측정

Phase 1 정책: prefill only, use_cache=False, FP32 고정.
"""
import argparse
import json
import os
import threading
import time

import torch

# ---------------------------------------------------------------- 모델 정의
LLM_MODELS = {
    "distilgpt2": "distilgpt2",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "deepseek-r1-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
}
VISION_MODELS = ["resnet152", "convnext_tiny", "mobilenet_v2"]

WARMUP_ITERS = 2
PROFILE_ITERS = 3          # 논문 3.1: 기본 3 iteration
SEQ_LEN = 128


def build_model(name, device):
    """모델과 더미 입력을 만든다. 가중치는 랜덤 초기화(구조만 필요)."""
    if name in VISION_MODELS:
        import torchvision.models as tvm
        model = getattr(tvm, name)(weights=None)
        family = "vision"
    elif name in LLM_MODELS:
        from transformers import AutoConfig, AutoModelForCausalLM
        cfg = AutoConfig.from_pretrained(LLM_MODELS[name])
        cfg.use_cache = False
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32)
        family = "llm"
    else:
        raise ValueError(f"unknown model: {name}")
    return model.to(device).eval().to(torch.float32), family


def make_inputs(name, family, batch, device):
    g = torch.Generator(device="cpu").manual_seed(0)
    if family == "vision":
        x = torch.randn(batch, 3, 224, 224, generator=g)
        return {"x": x.to(device)}
    vocab = 1000
    ids = torch.randint(0, vocab, (batch, SEQ_LEN), generator=g)
    return {"input_ids": ids.to(device),
            "attention_mask": torch.ones_like(ids).to(device)}


def forward(model, family, inputs):
    if family == "vision":
        return model(inputs["x"])
    return model(input_ids=inputs["input_ids"],
                 attention_mask=inputs["attention_mask"],
                 use_cache=False)


# ---------------------------------------------------------------- NVML 샘플러
class NvmlSampler(threading.Thread):
    """논문 4.1.1과 동일하게 1ms 간격으로 프로세스 GPU 사용량을 샘플링."""

    def __init__(self, interval=0.001):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self._evt = threading.Event()
        import pynvml
        self.nvml = pynvml
        self.nvml.nvmlInit()
        self.handle = self.nvml.nvmlDeviceGetHandleByIndex(0)
        self.pid = os.getpid()

    def _used(self):
        try:
            procs = self.nvml.nvmlDeviceGetComputeRunningProcesses_v3(self.handle)
        except Exception:
            procs = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        del procs
        return self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used

    def run(self):
        while not self._evt.is_set():
            self.peak = max(self.peak, self._used())
            time.sleep(self.interval)

    def stop(self):
        self._evt.set()
        self.join(timeout=2)
        self.peak = max(self.peak, self._used())
        return self.peak


# ---------------------------------------------------------------- 실행 경로
def run_cuda(model_name, batch, outdir):
    dev = torch.device("cuda")
    torch.cuda.init()
    torch.zeros(1, device=dev)          # CUDA 컨텍스트 강제 생성

    sampler = NvmlSampler()
    baseline_ctx = sampler._used()      # M_init + M_fm (컨텍스트 오버헤드)

    model, family = build_model(model_name, dev)
    inputs = make_inputs(model_name, family, batch, dev)

    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            forward(model, family, inputs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    sampler.start()
    with torch.no_grad():
        for _ in range(PROFILE_ITERS):
            forward(model, family, inputs)
            torch.cuda.synchronize()
    nvml_peak = sampler.stop()

    res = {
        "model": model_name, "family": family, "batch": batch,
        "device": "cuda", "dtype": "fp32", "seq_len": SEQ_LEN,
        "ctx_overhead_bytes": baseline_ctx,
        "nvml_peak_bytes": nvml_peak,
        "nvml_peak_minus_ctx_bytes": max(0, nvml_peak - baseline_ctx),
        "torch_max_reserved_bytes": torch.cuda.max_memory_reserved(),
        "torch_max_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    path = os.path.join(outdir, f"gt_{model_name}_bs{batch}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print(f"[saved] {path}")


def run_cpu(model_name, batch, outdir):
    from torch.profiler import ProfilerActivity, profile, record_function

    dev = torch.device("cpu")
    model, family = build_model(model_name, dev)
    inputs = make_inputs(model_name, family, batch, dev)

    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            forward(model, family, inputs)

    trace = os.path.join(outdir, f"trace_{model_name}_bs{batch}.json")
    with profile(
        activities=[ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
        with_modules=True,
    ) as prof:
        with torch.no_grad():
            for i in range(PROFILE_ITERS):
                with record_function(f"profiler.step#{i}"):
                    forward(model, family, inputs)
                prof.step()
    prof.export_chrome_trace(trace)

    size_mb = os.path.getsize(trace) / 2**20
    print(f"[saved] {trace}  ({size_mb:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--outdir", default="/work/results")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    if a.device == "cuda":
        run_cuda(a.model, a.batch, a.outdir)
    else:
        run_cpu(a.model, a.batch, a.outdir)


if __name__ == "__main__":
    main()

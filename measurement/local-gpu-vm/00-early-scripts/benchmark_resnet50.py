import torch
import torch.nn as nn
import time
import json
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, record_function

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_gpu_info():
    if not torch.cuda.is_available():
        return {"error": "CUDA 사용 불가 — CPU 모드로 실행 중"}
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "total_memory_gb": round(props.total_memory / 1024**3, 2),
        "multi_processor_count": props.multi_processor_count,
        "cuda_version": torch.version.cuda,
    }

def measure_memory(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
        "peak_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
    }

def measure_time(fn, warmup=3, repeat=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    return {
        "mean_ms": round(sum(times) / len(times), 3),
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
    }

def measure_flops(model, x):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_flops=True,
        record_shapes=True,
    ) as prof:
        with record_function("resnet50_forward"):
            with torch.no_grad():
                model(x)
    total = sum(e.flops for e in prof.key_averages() if e.flops)
    return total

def benchmark_resnet50():
    print("=" * 55)
    print("ResNet-50 GPU 벤치마크 (RTX 3090 / PyTorch)")
    print("=" * 55)

    gpu_info = get_gpu_info()
    print(f"GPU : {gpu_info.get('name')}")
    print(f"VRAM: {gpu_info.get('total_memory_gb')} GB")
    print(f"CUDA: {gpu_info.get('cuda_version')}")
    print()

    model = models.resnet50(weights=None).to(DEVICE).eval()
    results = {"gpu_info": gpu_info, "inference": {}, "training": {}}

    print(f"{'batch':>6} | {'time(ms)':>9} | {'peak VRAM(MB)':>13} | {'img/s':>8}")
    print("-" * 46)

    batch_sizes = [1, 8, 16, 32, 64]
    for bs in batch_sizes:
        try:
            x = torch.randn(bs, 3, 224, 224, device=DEVICE)

            flops = measure_flops(model, x)
            timing = measure_time(lambda: model(x))
            mem = measure_memory(lambda: model(x))
            img_per_sec = round(bs / (timing["mean_ms"] / 1000), 1)

            results["inference"][bs] = {
                "flops": flops,
                "gflops": round(flops / 1e9, 2),
                "timing_ms": timing,
                "memory": mem,
                "throughput_img_per_sec": img_per_sec,
            }
            print(f"{bs:>6} | {timing['mean_ms']:>9.2f} | {mem['peak_mb']:>13.1f} | {img_per_sec:>8.1f}")

        except torch.cuda.OutOfMemoryError:
            results["inference"][bs] = {"error": "OOM"}
            print(f"{bs:>6} | {'OOM':>9} | {'—':>13} | {'—':>8}")

    print("\n[학습 스텝 측정 — batch=32, forward+backward]")
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x_tr = torch.randn(32, 3, 224, 224, device=DEVICE)
    y_tr = torch.randint(0, 1000, (32,), device=DEVICE)

    def train_step():
        optimizer.zero_grad()
        loss = criterion(model(x_tr), y_tr)
        loss.backward()
        optimizer.step()

    train_timing = measure_time(train_step, warmup=2, repeat=5)
    train_mem = measure_memory(train_step)

    results["training"]["batch32"] = {
        "timing_ms": train_timing,
        "memory": train_mem,
    }
    print(f"  평균 스텝 시간 : {train_timing['mean_ms']:.2f} ms")
    print(f"  피크 VRAM      : {train_mem['peak_mb']:.1f} MB")

    with open("resnet50_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n결과 저장 완료 → resnet50_benchmark.json")
    print("\n[CloudSim GPUCloudlet 파라미터 요약]")
    b32 = results["inference"].get(32, {})
    if "error" not in b32:
        print(f"  GFLOPS (batch=32) : {b32['gflops']}")
        print(f"  Peak VRAM (MB)    : {b32['memory']['peak_mb']}")
        print(f"  Throughput        : {b32['throughput_img_per_sec']} img/s")
        print(f"  Train step (ms)   : {train_timing['mean_ms']}")

if __name__ == "__main__":
    benchmark_resnet50()

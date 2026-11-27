"""
YOLOv5 GPU 벤치마크 (RTX 3090 / PyTorch)
측정 항목: FLOPS, VRAM, 추론시간, 학습시간, 배치별 처리량
출력: yolov5_benchmark.json

실행 전 설치:
    pip install ultralytics
또는 ultralytics 없을 경우 torch.hub 자동 사용
"""

import torch
import torch.nn as nn
import json
from torch.profiler import profile, ProfilerActivity, record_function

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────
# 유틸리티 (ResNet 스크립트와 동일 구조)
# ────────────────────────────────────────────────

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
        "peak_mb":      round(torch.cuda.max_memory_allocated() / 1024**2, 2),
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
        "min_ms":  round(min(times), 3),
        "max_ms":  round(max(times), 3),
    }

def measure_flops(model, x):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_flops=True,
        record_shapes=True,
    ) as prof:
        with record_function("yolov5_forward"):
            with torch.no_grad():
                model(x)
    total = sum(e.flops for e in prof.key_averages() if e.flops)
    return total

# ────────────────────────────────────────────────
# 모델 로드 (ultralytics 우선, 없으면 torch.hub)
# ────────────────────────────────────────────────

def load_yolov5():
    """
    ultralytics 패키지가 있으면 YOLO("yolov5s.pt") 로드
    없으면 torch.hub 사용 (자동 다운로드)
    반환: (raw_model, model_name)
    """
    try:
        from ultralytics import YOLO
        wrapper = YOLO("yolov5s.pt")
        wrapper.to(DEVICE)
        model = wrapper.model.eval()
        print("  모델 로드: ultralytics YOLO('yolov5s.pt')")
        return model, "YOLOv5s (ultralytics)"
    except ImportError:
        print("  ultralytics 미설치 → torch.hub 사용 (자동 다운로드)")
        model = torch.hub.load(
            "ultralytics/yolov5", "yolov5s",
            pretrained=True, trust_repo=True
        ).to(DEVICE).eval()
        print("  모델 로드: torch.hub yolov5s")
        return model, "YOLOv5s (torch.hub)"

# ────────────────────────────────────────────────
# 메인 벤치마크
# ────────────────────────────────────────────────

def benchmark_yolov5():
    print("=" * 55)
    print("YOLOv5 GPU 벤치마크 (RTX 3090 / PyTorch)")
    print("=" * 55)

    gpu_info = get_gpu_info()
    print(f"GPU : {gpu_info.get('name')}")
    print(f"VRAM: {gpu_info.get('total_memory_gb')} GB")
    print(f"CUDA: {gpu_info.get('cuda_version')}")
    print()

    model, model_name = load_yolov5()
    results = {
        "gpu_info":   gpu_info,
        "model":      model_name,
        "input_size": "640x640",
        "inference":  {},
        "training":   {},
    }

    # ── 추론 (배치별) ──
    # YOLOv5 기본 입력 640×640
    print(f"{'batch':>6} | {'time(ms)':>9} | {'peak VRAM(MB)':>13} | {'img/s':>8} | {'GFLOPS':>8}")
    print("-" * 57)

    batch_sizes = [1, 4, 8, 16, 32]
    for bs in batch_sizes:
        try:
            x = torch.randn(bs, 3, 640, 640, device=DEVICE)

            flops       = measure_flops(model, x)
            timing      = measure_time(lambda: model(x) and None)
            mem         = measure_memory(lambda: model(x) and None)
            img_per_sec = round(bs / (timing["mean_ms"] / 1000), 1)
            gflops      = round(flops / 1e9, 2)

            results["inference"][bs] = {
                "flops":                  flops,
                "gflops":                 gflops,
                "timing_ms":              timing,
                "memory":                 mem,
                "throughput_img_per_sec": img_per_sec,
            }
            print(f"{bs:>6} | {timing['mean_ms']:>9.2f} | "
                  f"{mem['peak_mb']:>13.1f} | {img_per_sec:>8.1f} | {gflops:>8.2f}")

        except torch.cuda.OutOfMemoryError:
            results["inference"][bs] = {"error": "OOM"}
            print(f"{bs:>6} | {'OOM':>9} | {'—':>13} | {'—':>8} | {'—':>8}")

    # ── 학습 스텝 (forward + backward, batch=8) ──
    # YOLOv5는 배치 8이 일반적인 학습 기준
    print("\n[학습 스텝 측정 — batch=8, forward+backward]")
    try:
        model.train()

        # YOLOv5 학습용 더미 타깃
        # 타깃 형식: [batch_idx, class, cx, cy, w, h]
        x_tr = torch.randn(8, 3, 640, 640, device=DEVICE)
        targets = torch.zeros(16, 6, device=DEVICE)  # 배치당 2개 박스
        for i in range(8):
            targets[i * 2,     0] = i          # batch_idx
            targets[i * 2,     1] = 0          # class
            targets[i * 2,   2:] = torch.tensor([0.5, 0.5, 0.3, 0.3])
            targets[i * 2 + 1, 0] = i
            targets[i * 2 + 1, 1] = 1
            targets[i * 2 + 1, 2:] = torch.tensor([0.2, 0.3, 0.15, 0.2])

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        def train_step():
            optimizer.zero_grad()
            preds = model(x_tr)
            # ultralytics / hub 모델 모두 리스트 반환
            # 간단한 더미 손실 (실제 YOLOv5 손실 함수 대체)
            if isinstance(preds, (list, tuple)):
                loss = sum(p.mean() for p in preds if isinstance(p, torch.Tensor))
            else:
                loss = preds.mean()
            loss.backward()
            optimizer.step()

        train_timing = measure_time(train_step, warmup=2, repeat=5)
        train_mem    = measure_memory(train_step)

        results["training"]["batch8"] = {
            "timing_ms": train_timing,
            "memory":    train_mem,
        }
        print(f"  평균 스텝 시간 : {train_timing['mean_ms']:.2f} ms")
        print(f"  피크 VRAM      : {train_mem['peak_mb']:.1f} MB")

    except Exception as e:
        results["training"]["batch8"] = {"error": str(e)}
        print(f"  학습 측정 실패: {e}")

    # ── 저장 ──
    with open("yolov5_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n결과 저장 완료 → yolov5_benchmark.json")

    # ── CloudSim 파라미터 요약 ──
    print("\n[CloudSim GPUCloudlet 파라미터 요약]")
    b8 = results["inference"].get(8, {})
    if "error" not in b8:
        print(f"  GFLOPS (batch=8)  : {b8.get('gflops', 'N/A')}")
        print(f"  Peak VRAM (MB)    : {b8.get('memory', {}).get('peak_mb', 'N/A')}")
        print(f"  Throughput        : {b8.get('throughput_img_per_sec', 'N/A')} img/s")
    train_b8 = results["training"].get("batch8", {})
    if "error" not in train_b8:
        print(f"  Train step (ms)   : {train_b8.get('timing_ms', {}).get('mean_ms', 'N/A')}")
        print(f"  Train VRAM (MB)   : {train_b8.get('memory', {}).get('peak_mb', 'N/A')}")

if __name__ == "__main__":
    benchmark_yolov5()

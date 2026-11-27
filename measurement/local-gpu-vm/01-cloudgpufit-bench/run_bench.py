import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path

MODES = [
    "components", "memory",
    "performance", "first_logits", "tokens",
]


def write(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=["local_smoke", "local_core", "aws_core"],
        default="local_smoke",
    )
    parser.add_argument(
        "--plan", default="benchmark_plan.json",
    )
    parser.add_argument(
        "--image", default="cloudgpufit-bench:cu124-v1",
    )
    parser.add_argument(
        "--instance-label", default="inugpu-rtx3090",
    )
    parser.add_argument(
        "--modes", nargs="+", choices=MODES, default=MODES,
    )
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--max-start-temp", type=float)
    parser.add_argument(
        "--idle-timeout", type=int, default=180,
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    plan = json.loads(
        Path(args.plan).resolve().read_text()
    )
    profile = plan["execution_profiles"][args.profile]
    cases = [
        case for case in plan["cases"]
        if case["suite"] == profile["suite"]
    ]

    if args.model:
        cases = [
            case for case in cases
            if case["model"] == args.model
        ]
    if args.limit:
        assert args.limit > 0
        cases = cases[:args.limit]

    assert cases
    assert all(
        re.fullmatch(
            r"[A-Za-z0-9_-]+", case["case_id"]
        ) for case in cases
    )
    assert len({
        case["case_id"] for case in cases
    }) == len(cases)

    worker = root / "bench_worker.py"
    assert worker.is_file()

    threads = (
        args.threads or len(os.sched_getaffinity(0))
    )
    assert threads > 0

    cache = root / "hf-cache"
    cache.mkdir(exist_ok=True)

    stamp = datetime.datetime.now(
        datetime.timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    folder = root / "results" / (
        stamp + "_" + uuid.uuid4().hex[:8]
    )
    folder.mkdir(parents=True)

    selected = folder / "selected-plan.json"
    plan["cases"] = cases
    plan["execution_profiles"][
        args.profile
    ]["modes"] = args.modes
    write(selected, plan)

    print(
        "Cases:", len(cases),
        "Repeats:", profile["repeats"],
        "Modes:", args.modes,
        flush=True,
    )

    image_id = subprocess.check_output([
        "sudo", "docker", "image", "inspect",
        args.image, "--format", "{{.Id}}",
    ], text=True).strip()

    def gpu_state():
        line = subprocess.check_output([
            "nvidia-smi", "--id=0",
            "--query-gpu="
            "memory.used,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], text=True).strip()
        return [
            float(value.strip())
            for value in line.split(",")
        ]

    baseline = gpu_state()
    max_temperature = (
        args.max_start_temp
        if args.max_start_temp is not None
        else min(70, baseline[1] + 10)
    )

    def wait_idle():
        deadline = time.monotonic() + args.idle_timeout
        stable = 0
        while time.monotonic() < deadline:
            used, temperature, utilization = gpu_state()
            good = (
                used <= baseline[0] + 64
                and temperature <= max_temperature
                and utilization <= 5
            )
            stable = stable + 1 if good else 0
            if stable >= 3:
                return [used, temperature, utilization]
            time.sleep(1)
        raise RuntimeError(
            "GPU idle/temperature check timed out; "
            "stop and inspect GPU state"
        )

    base = [
        "sudo", "docker", "run", "--rm",
        "--gpus", "device=0",
        "--pid=host",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", f"BENCH_IMAGE_ID={image_id}",
        "-e", f"BENCH_INSTANCE_LABEL={args.instance_label}",
        "-e", "HF_HUB_DISABLE_PROGRESS_BARS=1",
        "-v", f"{worker}:/work/bench_worker.py:ro",
        "-v", f"{selected}:/plan.json:ro",
        "-v", f"{cache}:/cache/huggingface",
        "-v", f"{folder}:/results",
    ]

    # 모델 다운로드와 revision 고정은 측정 전에 완료합니다.
    subprocess.run(
        base + [
            image_id,
            "python", "/work/bench_worker.py",
            "--prepare", "/plan.json",
        ],
        check=True,
    )

    lock = subprocess.check_output([
        "sudo", "docker", "run", "--rm",
        image_id, "cat", "/opt/environment.lock.txt",
    ], text=True)
    (folder / "environment.lock.txt").write_text(lock)

    write(folder / "run-manifest.json", {
        "profile": args.profile,
        "profile_settings": profile,
        "modes": args.modes,
        "threads": threads,
        "image_id": image_id,
        "instance_label": args.instance_label,
        "worker_sha256": hashlib.sha256(
            worker.read_bytes()
        ).hexdigest(),
        "host_os": Path("/etc/os-release").read_text(),
        "initial_gpu_state": baseline,
        "maximum_start_temperature_c": max_temperature,
        "case_order": [
            case["case_id"] for case in cases
        ],
        "components_iterations": 1,
        "note": (
            "Fresh containers; one sample is one request; "
            "token/component runs are diagnostic."
        ),
    })

    columns = [
        "case_id", "model", "dtype", "params",
        "mode", "repeat", "status",
        "latency_p90_ms", "req_s", "output_tokens_s",
        "ttft_p90_ms", "tpot_p90_ms",
        "torch_peak_reserved_mib",
        "nvml_sampled_peak_mib",
        "weight_storage_mib",
        "self_kv_snapshot_peak_mib",
        "cross_kv_snapshot_peak_mib",
        "nvml_status",
    ]

    def append_summary(data):
        memory = data.get("memory_baseline") or {}
        snapshots = (
            data.get("component_snapshots") or []
        )

        def mib(value):
            return (
                "" if value is None
                else value / (1024 ** 2)
            )

        row = {
            "case_id": data["case"]["case_id"],
            "model": data["case"]["model"],
            "dtype": data["case"]["dtype"],
            "params": json.dumps(
                data["case"]["params"]
            ),
            "mode": data["mode"],
            "repeat": data["repeat"],
            "status": data["status"],
            "latency_p90_ms": (
                data.get("latency_ms") or {}
            ).get("p90", ""),
            "req_s": (
                data.get("throughput") or {}
            ).get("requests_per_second", ""),
            "output_tokens_s": (
                data.get("throughput") or {}
            ).get("output_tokens_per_second", ""),
            "ttft_p90_ms": (
                data.get("ttft_ms") or {}
            ).get("p90", ""),
            "tpot_p90_ms": (
                data.get("tpot_ms") or {}
            ).get("p90", ""),
            "torch_peak_reserved_mib": mib(
                data.get("inference_peak_reserved_bytes")
            ),
            "nvml_sampled_peak_mib": mib(
                data.get(
                    "nvml_process_sampled_peak_bytes"
                )
            ),
            "weight_storage_mib": mib(
                memory.get("parameter_storage_bytes")
            ),
            "self_kv_snapshot_peak_mib": mib(max(
                (
                    snapshot["self_kv_storage_bytes"]
                    for snapshot in snapshots
                ),
                default=None,
            )),
            "cross_kv_snapshot_peak_mib": mib(max(
                (
                    snapshot["cross_kv_storage_bytes"]
                    for snapshot in snapshots
                ),
                default=None,
            )),
            "nvml_status": data.get("nvml_status", ""),
        }

        path = folder / "summary.csv"
        new_file = not path.exists()
        with path.open(
            "a",
            encoding=(
                "utf-8-sig" if new_file else "utf-8"
            ),
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=columns,
            )
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    guard = {}
    unsupported_dtypes = set()
    index = []
    fraction = plan.get(
        "runner_settings", {}
    ).get("memory_stop_fraction", 0.85)

    for case in cases:
        params = case["params"]
        key = (
            case["model"],
            case["dtype"],
            tuple(sorted(
                (name, value)
                for name, value in params.items()
                if name != "batch_size"
            )),
        )

        for repeat in range(profile["repeats"]):
            invalid = None

            for mode in args.modes:
                name = (
                    f'{case["case_id"]}__'
                    f'{mode}__r{repeat}.json'
                )
                output = folder / name

                reason = (
                    "unsupported_dtype"
                    if case["dtype"] in unsupported_dtypes
                    else "skipped_memory_guard"
                    if params["batch_size"]
                    >= guard.get(key, float("inf"))
                    else invalid
                    if invalid
                    else "not_applicable"
                    if case["workload_type"] == "vision_cls"
                    and mode in ("tokens", "first_logits")
                    else None
                )

                if reason:
                    write(output, {
                        "case": case,
                        "mode": mode,
                        "repeat": repeat,
                        "status": reason,
                    })
                else:
                    state = wait_idle()
                    command = base + [
                        "--network", "none",
                        "-e", "HF_HUB_OFFLINE=1",
                        image_id,
                        "python", "/work/bench_worker.py",
                        "--case", case["case_id"],
                        "--mode", mode,
                        "--repeat", str(repeat),
                        "--threads", str(threads),
                        "--warmup", str(profile.get(
                            "warmup",
                            profile.get(
                                "warmup_invocations", 3
                            ),
                        )),
                        "--iterations", str(profile.get(
                            "iterations",
                            profile.get(
                                "measured_invocations", 20
                            ),
                        )),
                        "--memory-fraction", str(fraction),
                        "--output", f"/results/{name}",
                    ]
                    completed = subprocess.run(command)

                    if not output.exists():
                        write(output, {
                            "case": case,
                            "mode": mode,
                            "repeat": repeat,
                            "status": "worker_failed",
                            "exit_code": completed.returncode,
                        })

                    data = json.loads(output.read_text())
                    data["start_gpu_state"] = state
                    data["worker_exit_code"] = (
                        completed.returncode
                    )
                    write(output, data)

                data = json.loads(output.read_text())
                status = data["status"]

                if status == "unsupported_dtype":
                    unsupported_dtypes.add(case["dtype"])

                if status in (
                    "gpu_oom", "skipped_memory_guard",
                ):
                    guard[key] = min(
                        guard.get(key, float("inf")),
                        params["batch_size"],
                    )
                    invalid = "skipped_memory_guard"

                elif data.get("memory_guard_exceeded"):
                    guard[key] = min(
                        guard.get(key, float("inf")),
                        params["batch_size"] + 1,
                    )

                if status == "nonfinite_output":
                    invalid = "skipped_invalid_output"

                index.append({
                    "case_id": case["case_id"],
                    "mode": mode,
                    "repeat": repeat,
                    "status": status,
                    "file": name,
                })
                append_summary(data)
                write(folder / "index.json", index)

                print(
                    case["case_id"],
                    mode, repeat, status,
                    flush=True,
                )

                if status == "gpu_busy":
                    raise RuntimeError(
                        "Another GPU process is running; "
                        "stop the experiment"
                    )

    print("Results:", folder)


if __name__ == "__main__":
    main()

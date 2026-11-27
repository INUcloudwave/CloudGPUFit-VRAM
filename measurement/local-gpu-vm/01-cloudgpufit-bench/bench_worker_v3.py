import argparse
import copy
import gc
import hashlib
import json
import os
import platform
import statistics
import threading
import time
import traceback
from pathlib import Path

import torch
import transformers
import pynvml
from huggingface_hub import HfApi, snapshot_download
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForImageClassification,
)
from transformers.generation.streamers import BaseStreamer

CLASSES = {
    "decoder_llm": AutoModelForCausalLM,
    "text_encdec": AutoModelForSeq2SeqLM,
    "vision_cls": AutoModelForImageClassification,
}
FIELDS = {
    "decoder_llm": ("batch_size", "seq_len", "max_new_tokens"),
    "vision_cls": ("batch_size", "resolution"),
    "text_encdec": (
        "batch_size", "encoder_seq_len", "decoder_seq_len", "max_new_tokens",
    ),
}


class NonfiniteOutput(RuntimeError):
    pass


def save(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def prepare(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    path = Path(os.environ["HF_HOME"]) / "benchmark-revisions.json"
    pins = json.loads(path.read_text()) if path.exists() else {}

    for model_id in dict.fromkeys(c["model"] for c in plan["cases"]):
        old = pins.get(model_id, {})
        info = HfApi().model_info(
            model_id, revision=old.get("revision", "main"),
        )
        safe = any(
            f.rfilename.endswith(".safetensors") for f in info.siblings
        )
        snapshot_download(
            model_id,
            revision=info.sha,
            allow_patterns=[
                "*.json",
                "*.safetensors" if safe else "pytorch_model*.bin",
            ],
        )
        pins[model_id] = {
            "revision": info.sha,
            "safetensors": safe,
        }
        save(path, pins)
        print("Prepared:", model_id, info.sha)

    save("/results/revisions.json", pins)


def summary(values):
    if not values:
        return None

    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.9
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    p90 = ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * (position - lower)

    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": p90,
        "max": max(values),
    }


def storage_bytes(tensors):
    unique = {}
    for tensor in tensors:
        if torch.is_tensor(tensor) and tensor.is_cuda:
            storage = tensor.untyped_storage()
            unique[
                (str(tensor.device), storage.data_ptr())
            ] = storage.nbytes()
    return sum(unique.values())


def cache_bytes(cache):
    if cache is None:
        return 0, 0

    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()

    if not isinstance(cache, (tuple, list)):
        raise RuntimeError(
            "Unsupported cache structure; cannot attribute KV memory"
        )

    self_tensors, cross_tensors = [], []
    for layer in cache:
        self_tensors.extend(layer[:2])
        cross_tensors.extend(layer[2:])

    return (
        storage_bytes(self_tensors),
        storage_bytes(cross_tensors),
    )


class TokenTrace(BaseStreamer):
    def reset(self):
        self.prompt = True
        self.times = []

    def put(self, value):
        if self.prompt:
            self.prompt = False
            return
        self.times.append(time.perf_counter_ns())

    def end(self):
        pass


class Sampler:
    def __init__(self, handle, capacity, interval):
        self.handle = handle
        self.capacity = capacity
        self.interval = interval
        self.rows = []
        self.errors = []
        self.stop_event = threading.Event()

    def sample(self):
        begin = time.perf_counter_ns()
        process_bytes = None

        try:
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(
                self.handle
            )
            for process in processes:
                value = process.usedGpuMemory
                if (
                    process.pid == os.getpid()
                    and value is not None
                    and 0 <= value <= self.capacity
                ):
                    process_bytes = int(value)
                    break
        except Exception as exc:
            self.errors.append(str(exc))

        self.rows.append({
            "query_begin_ns": begin,
            "nvml_process_bytes": process_bytes,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "query_end_ns": time.perf_counter_ns(),
        })

    def start(self):
        self.sample()

        def loop():
            while not self.stop_event.wait(self.interval):
                self.sample()

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.sample()



MODE_ORDER = ["performance", "memory", "first_logits", "tokens", "components"]

def emit(event, **data):
    if os.environ.get("BENCH_PROGRESS") == "0":
        return
    print("BENCH_EVENT " + json.dumps({
        "event": event, "wall_time_ns": time.time_ns(), **data,
    }), flush=True)

def load_or_reuse(session, key, factory):
    if session and session.get("key") != key:
        raise RuntimeError("A shared session cannot cross a case/model/dtype boundary")
    if session:
        model = session["model"]
        model.config = copy.deepcopy(session["config"])
        model.generation_config = copy.deepcopy(session["generation_config"])
        torch.set_rng_state(session["cpu_rng"])
        torch.cuda.set_rng_state(session["cuda_rng"])
        return model, True
    model = factory()
    session.update(key=key, model=model,
                   config=copy.deepcopy(model.config),
                   generation_config=copy.deepcopy(model.generation_config),
                   cpu_rng=torch.get_rng_state().clone(),
                   cuda_rng=torch.cuda.get_rng_state().clone())
    return model, False

def cleanup_between_modes(keep_allocator_cache=False):
    begin = time.perf_counter()
    torch.cuda.synchronize()
    gc.collect()
    if not keep_allocator_cache:
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return {"seconds": time.perf_counter()-begin, "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "allocator_cache_retained": keep_allocator_cache}

def wait_mode_idle(max_temperature, timeout):
    # The model remains resident. Utilization/temperature retain the old rule;
    # host total-memory <= 65 MiB is only valid BETWEEN fresh case processes.
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    begin = time.perf_counter()
    stable = 0
    while time.perf_counter() - begin < timeout:
        others = [p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                  if p.pid != os.getpid()]
        if others:
            raise RuntimeError(f"Another GPU process appeared: {others}")
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        stable = stable + 1 if (max_temperature <= 0 or temperature <= max_temperature) and utilization <= 5 else 0
        if stable >= (1 if max_temperature <= 0 else 3):
            return {"seconds": time.perf_counter()-begin,
                    "temperature_c": temperature, "utilization_percent": utilization,
                    "device_used_mib": pynvml.nvmlDeviceGetMemoryInfo(handle).used/(1024**2)}
        time.sleep(1)
    raise RuntimeError("Between-mode GPU temperature/utilization check timed out")

def gpu_health(handle):
    values = {}
    for name, query in {
        "temperature_c": lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
        "sm_clock_mhz": lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
        "clock_event_reasons": lambda: pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle),
    }.items():
        try:
            values[name] = int(query())
        except Exception as exc:
            values[name] = None
            values[name+"_unavailable"] = str(exc)
    reasons = values["clock_event_reasons"]
    mask = getattr(pynvml, "nvmlClocksThrottleReasonHwThermalSlowdown", 0) | getattr(pynvml, "nvmlClocksThrottleReasonSwThermalSlowdown", 0)
    values["thermal_slowdown_at_boundary"] = bool(reasons & mask) if reasons is not None and mask else None
    return values

def run_shared(args):
    plan = json.loads(Path(args.plan).read_text())
    case = next(c for c in plan["cases"] if c["case_id"] == args.case)
    if "{mode}" not in args.output:
        raise ValueError("Multi-mode output must contain {mode}")
    session, results = {}, []
    reference_input = None
    failure = None
    for mode in [m for m in MODE_ORDER if m in args.modes]:
        current = copy.copy(args)
        current.mode = mode
        current.output = args.output.format(mode=mode)
        current.iterations = json.loads(args.mode_iterations_json).get(mode, args.iterations)
        inactive = case["workload_type"] == "vision_cls" and mode in ("first_logits", "tokens")
        cleanup, idle = None, None
        try:
            if failure or inactive:
                save(current.output, {"case": case, "mode": mode, "repeat": args.repeat,
                     "status": "not_applicable" if inactive else failure})
            else:
                if session:
                    emit("phase", case_id=args.case, mode=mode, phase="between_modes_cleanup")
                    cleanup = cleanup_between_modes(keep_allocator_cache=args.keep_allocator_cache)
                    emit("phase", case_id=args.case, mode=mode, phase="between_modes_gpu_wait")
                    idle = wait_mode_idle(args.max_start_temp, args.idle_timeout)
                main(current, session=session)
            data = json.loads(Path(current.output).read_text())
            if data.get("input_sha256"):
                if reference_input is None:
                    reference_input = data["input_sha256"]
                elif reference_input != data["input_sha256"]:
                    data["status"] = "input_mismatch_between_modes"
            data.update(execution_structure="shared_case", between_mode_cleanup=cleanup,
                        between_mode_idle=idle)
            save(current.output, data)
            results.append((current.output, data))
            if data["status"] == "nonfinite_output":
                failure = "skipped_invalid_output"
                for path, previous in results[:-1]:
                    if previous["status"] == "ok":
                        previous.update(status="invalidated_by_component_check",
                                        generation_validation="failed")
                        save(path, previous)
            elif data["status"] in ("gpu_oom", "skipped_memory_guard"):
                failure = "skipped_memory_guard"
            elif data["status"] not in ("ok", "not_applicable"):
                failure = "skipped_after_worker_error"
            emit("mode_complete", case_id=args.case, mode=mode, status=data["status"],
                 timings_seconds=data.get("timings_seconds", {}))
        except Exception:
            # Cleanup/idle failures must stop rather than silently relax isolation.
            emit("fatal", case_id=args.case, mode=mode, traceback=traceback.format_exc())
            raise


def main(args, session=None):
    plan = json.loads(Path(args.plan).read_text())
    case = next(
        c for c in plan["cases"] if c["case_id"] == args.case
    )
    result = {
        "case": case,
        "mode": args.mode,
        "repeat": args.repeat,
        "status": "runtime_error",
        "stage": "initialization",
    }
    sampler, hook = None, None
    phase_name = "initialization"
    phase_begin = time.perf_counter()
    main_begin = phase_begin
    result["timings_seconds"] = {}

    def stage(name):
        nonlocal phase_name, phase_begin
        now = time.perf_counter()
        result["timings_seconds"][phase_name] = now-phase_begin
        result["stage"] = phase_name = name
        phase_begin = now
        emit("phase", case_id=args.case, mode=args.mode, phase=name)


    try:
        params = case["params"]
        kind = case["workload_type"]

        if (
            set(params) != set(FIELDS[kind])
            or any(
                type(value) is not int or value <= 0
                for value in params.values()
            )
        ):
            raise ValueError("Invalid input fields")

        assert torch.cuda.is_available()
        assert torch.cuda.device_count() == 1

        if (
            case["dtype"] == "bf16"
            and torch.cuda.get_device_capability()[0] < 8
        ):
            result["status"] = "unsupported_dtype"
            return

        if (
            kind == "vision_cls"
            and args.mode in ("tokens", "first_logits")
        ):
            result["status"] = "not_applicable"
            return

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        others = [
            process.pid
            for process in
            pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if process.pid != os.getpid()
        ]
        if others:
            result.update(
                status="gpu_busy",
                other_gpu_pids=others,
            )
            return

        capacity = int(
            pynvml.nvmlDeviceGetMemoryInfo(handle).total
        )
        torch.set_num_threads(args.threads)
        torch.manual_seed(20261006)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

        dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[case["dtype"]]

        pins = json.loads(
            (
                Path(os.environ["HF_HOME"])
                / "benchmark-revisions.json"
            ).read_text()
        )
        pin = pins[case["model"]]
        config = AutoConfig.from_pretrained(
            case["model"],
            revision=pin["revision"],
            local_files_only=True,
        )

        if (
            kind == "decoder_llm"
            and params["seq_len"] + params["max_new_tokens"]
            > config.max_position_embeddings
        ):
            result["status"] = "unsupported_shape"
            return

        if (
            config.model_type == "vit"
            and params["resolution"] != config.image_size
        ):
            result["status"] = "unsupported_shape"
            return

        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
        stage("model_reuse" if session else "model_load")

        interval = plan.get(
            "runner_settings", {}
        ).get("memory_sampling_interval_ms", 50)

        if args.mode == "memory":
            sampler = Sampler(
                handle, capacity, interval / 1000,
            )
            sampler.start()

        def factory():
            return CLASSES[kind].from_pretrained(
                case["model"], revision=pin["revision"], torch_dtype=dtype,
                low_cpu_mem_usage=True, device_map={"": "cuda:0"},
                use_safetensors=pin["safetensors"], local_files_only=True,
                attn_implementation="eager",
            ).to(device="cuda", dtype=dtype).eval()

        if session is None:
            model, reused = factory(), False
        else:
            model, reused = load_or_reuse(
                session, (args.case, case["model"], case["dtype"], pin["revision"]), factory)
        result["model_reused"] = reused
        result["startup_metrics_scope"] = "reused_resident_model" if reused else "actual_model_load"

        model.config.output_attentions = False
        model.config.output_hidden_states = False
        torch.cuda.synchronize()

        if sampler:
            sampler.stop()
            result["startup_memory_samples"] = sampler.rows
            values = [
                row["nvml_process_bytes"]
                for row in sampler.rows
                if row["nvml_process_bytes"] is not None
            ]
            result["startup_nvml_sampled_peak_bytes"] = (
                max(values) if values else None
            )
            result["startup_sampling_errors"] = sampler.errors
            sampler = None

        stage("input_and_validation")

        result["startup_peak_allocated_bytes"] = (
            torch.cuda.max_memory_allocated()
        )
        result["startup_peak_reserved_bytes"] = (
            torch.cuda.max_memory_reserved()
        )
        idle_allocated = torch.cuda.memory_allocated()

        result["memory_baseline"] = {
            "parameter_storage_bytes": storage_bytes(
                model.parameters()
            ),
            "buffer_storage_bytes": storage_bytes(
                model.buffers()
            ),
            "resident_allocated_bytes": idle_allocated,
            "resident_reserved_bytes": (
                torch.cuda.memory_reserved()
            ),
        }
        result["environment"] = {
            "gpu": torch.cuda.get_device_name(0),
            "total_vram_bytes": capacity,
            "cuda_total_vram_bytes": (
                torch.cuda.get_device_properties(0).total_memory
            ),
            "driver": pynvml.nvmlSystemGetDriverVersion(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "python": platform.python_version(),
            "kernel": platform.release(),
            "cpu_threads": args.threads,
            "available_vcpus": len(os.sched_getaffinity(0)),
            "revision": pin["revision"],
            "attention": "eager",
            "tf32": False,
            "seed": 20261006,
            "image_id": os.environ["BENCH_IMAGE_ID"],
            "instance_label": os.environ["BENCH_INSTANCE_LABEL"],
        }

        if (
            torch.cuda.memory_reserved()
            > args.memory_fraction * capacity
        ):
            result["status"] = "skipped_memory_guard"
            return

        batch = params["batch_size"]

        # CPU 입력 하나를 준비하고 반복 측정에 재사용합니다.
        if kind == "vision_cls":
            inputs = {
                "pixel_values": torch.randn(
                    batch,
                    config.num_channels,
                    params["resolution"],
                    params["resolution"],
                    dtype=dtype,
                )
            }
        else:
            length = (
                params["seq_len"]
                if kind == "decoder_llm"
                else params["encoder_seq_len"]
            )
            inputs = {
                "input_ids": torch.randint(
                    3, config.vocab_size, (batch, length)
                ),
                "attention_mask": torch.ones(
                    batch, length, dtype=torch.long
                ),
            }
            special = [
                getattr(config, name, None)
                for name in
                ("pad_token_id", "eos_token_id", "bos_token_id")
            ]
            for token in special:
                if isinstance(token, int):
                    inputs["input_ids"][
                        inputs["input_ids"] == token
                    ] = 3

            if kind == "text_encdec":
                prefix = torch.randint(
                    3, config.vocab_size,
                    (batch, params["decoder_seq_len"]),
                )
                for token in special:
                    if isinstance(token, int):
                        prefix[prefix == token] = 3
                prefix[:, 0] = config.decoder_start_token_id
                inputs["decoder_input_ids"] = prefix

        result["actual_input_shapes"] = {
            name: list(value.shape)
            for name, value in inputs.items()
        }
        result["finite_check_scope"] = (
            "every_generation_step_in_components_mode; "
            "first_forward_in_other_modes"
        )
        result["input_sha256"] = {
            name: hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for name, value in inputs.items()
        }
        stage("finite_output_check")

        with torch.inference_mode():
            probe = model(**{
                name: value.to("cuda")
                for name, value in inputs.items()
            })
            finite = bool(
                torch.isfinite(probe.logits).all().item()
            )
            del probe

        if not finite:
            result["status"] = "nonfinite_output"
            return

        tracer = TokenTrace()
        components = []
        component_sampler = Sampler(handle, capacity, 0.05)

        def component_hook(module, arguments, output):
            torch.cuda.synchronize()
            if not bool(
                torch.isfinite(output.logits).all().item()
            ):
                raise NonfiniteOutput(
                    "Nonfinite logits in a generation step"
                )

            self_bytes, cross_bytes = cache_bytes(
                getattr(output, "past_key_values", None)
            )
            component_sampler.sample()
            paired = component_sampler.rows[-1]
            allocated = paired["allocated_bytes"]

            components.append({
                **paired,
                "gpu_done_ns": time.perf_counter_ns(),
                "self_kv_storage_bytes": self_bytes,
                "cross_kv_storage_bytes": cross_bytes,
                "outside_reserved_bytes": (
                    None
                    if paired["nvml_process_bytes"] is None
                    else paired["nvml_process_bytes"]
                    - paired["reserved_bytes"]
                ),
                "unclassified_above_idle_bytes": (
                    allocated - idle_allocated
                    - self_bytes - cross_bytes
                ),
            })

        def invoke(measured=False):
            torch.cuda.synchronize()
            begin = time.perf_counter_ns()
            tracer.reset()

            with torch.inference_mode():
                gpu_inputs = {
                    name: value.to("cuda")
                    for name, value in inputs.items()
                }

                if (
                    kind == "vision_cls"
                    or args.mode == "first_logits"
                ):
                    output = model(**gpu_inputs)
                    logits = (
                        output.logits
                        if kind == "vision_cls"
                        else output.logits[:, -1, :]
                    )
                    logits.cpu()
                    new_tokens = 0
                else:
                    output = model.generate(
                        **gpu_inputs,
                        do_sample=False,
                        num_beams=1,
                        use_cache=True,
                        min_new_tokens=params["max_new_tokens"],
                        max_new_tokens=params["max_new_tokens"],
                        pad_token_id=(
                            config.pad_token_id
                            if config.pad_token_id is not None
                            else config.eos_token_id
                        ),
                        return_dict_in_generate=False,
                        output_scores=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        streamer=(
                            tracer
                            if measured and args.mode == "tokens"
                            else None
                        ),
                    )
                    output.cpu()
                    prefix_length = (
                        params["seq_len"]
                        if kind == "decoder_llm"
                        else params["decoder_seq_len"]
                    )
                    new_tokens = (
                        output.shape[1] - prefix_length
                    )
                    if new_tokens != params["max_new_tokens"]:
                        raise RuntimeError(
                            "Actual output length differs "
                            "from the condition"
                        )

            torch.cuda.synchronize()
            end = time.perf_counter_ns()
            del output, gpu_inputs

            row = {
                "start_ns": begin,
                "end_ns": end,
                "elapsed_ms": (end - begin) / 1e6,
                "actual_new_tokens_per_sample": new_tokens,
            }

            if measured and args.mode == "tokens":
                if len(tracer.times) != new_tokens:
                    raise RuntimeError(
                        "Unexpected token trace length"
                    )
                intervals = [
                    (later - earlier) / 1e6
                    for earlier, later in
                    zip(tracer.times, tracer.times[1:])
                ]
                row.update(
                    token_visible_ns=list(tracer.times),
                    ttft_ms=(
                        tracer.times[0] - begin
                    ) / 1e6,
                    itl_ms=intervals,
                    tpot_ms=(
                        statistics.mean(intervals)
                        if intervals else None
                    ),
                )
            return row

        stage("warmup")
        for _ in range(args.warmup):
            invoke()

        torch.cuda.reset_peak_memory_stats()

        if args.mode == "memory":
            sampler = Sampler(
                handle, capacity, interval / 1000,
            )
            sampler.start()

        if args.mode == "components":
            hook = model.register_forward_hook(
                component_hook
            )

        stage("measurement")
        count = (
            1 if args.mode == "components"
            else args.iterations
        )
        records = []
        result["raw_invocations"] = records
        result["component_snapshots"] = components
        result["component_sampling_errors"] = (
            component_sampler.errors
        )

        result["gpu_health_before_measurement"] = gpu_health(handle)
        for _ in range(count):
            records.append(invoke(True))

        stage("postprocessing")
        result["gpu_health_after_measurement"] = gpu_health(handle)

        if hook:
            hook.remove()
            hook = None

        if sampler:
            sampler.stop()
            rows, errors = sampler.rows, sampler.errors
            sampler = None

            # 독립적인 피크값끼리 빼지 않습니다.
            for row in rows:
                row["cached_unused_bytes"] = (
                    row["reserved_bytes"]
                    - row["allocated_bytes"]
                )
                row["outside_reserved_bytes"] = (
                    None
                    if row["nvml_process_bytes"] is None
                    else row["nvml_process_bytes"]
                    - row["reserved_bytes"]
                )

            values = [
                row["nvml_process_bytes"]
                for row in rows
                if row["nvml_process_bytes"] is not None
            ]
            result.update(
                memory_samples=rows,
                sampling_errors=errors,
                nvml_process_sampled_peak_bytes=(
                    max(values) if values else None
                ),
                nvml_sample_coverage=len(values) / len(rows),
                nvml_status=(
                    "ok" if len(values) == len(rows)
                    else "partial" if values
                    else "unavailable"
                ),
            )

        result["inference_peak_allocated_bytes"] = (
            torch.cuda.max_memory_allocated()
        )
        result["inference_peak_reserved_bytes"] = (
            torch.cuda.max_memory_reserved()
        )

        if args.mode in ("performance", "first_logits"):
            result["latency_ms"] = summary([
                row["elapsed_ms"] for row in records
            ])

        if args.mode == "performance":
            elapsed = (
                records[-1]["end_ns"]
                - records[0]["start_ns"]
            ) / 1e9
            result["throughput"] = {
                "window_seconds": elapsed,
                "batches_per_second": count / elapsed,
                "samples_per_second": batch * count / elapsed,
                "request_definition": "one_sample_in_batch",
                "requests_per_second": batch * count / elapsed,
            }
            if kind == "text_encdec":
                result["throughput"][
                    "decoder_prefix_tokens_per_second"
                ] = (
                    batch * count
                    * params["decoder_seq_len"] / elapsed
                )
            if kind != "vision_cls":
                result["throughput"].update(
                    output_tokens_per_second=(
                        batch * sum(
                            row["actual_new_tokens_per_sample"]
                            for row in records
                        ) / elapsed
                    ),
                    input_tokens_per_second=(
                        batch * count
                        * inputs["input_ids"].shape[1]
                        / elapsed
                    ),
                )

        if args.mode == "tokens":
            result["ttft_ms"] = summary([
                row["ttft_ms"] for row in records
            ])
            result["tpot_ms"] = summary([
                row["tpot_ms"] for row in records
                if row["tpot_ms"] is not None
            ])
            result["itl_ms"] = summary([
                value
                for row in records
                for value in row["itl_ms"]
            ])

        result["scope"] = (
            "classification_forward"
            if kind == "vision_cls"
            else "first_token_logits"
            if args.mode == "first_logits"
            else "full_generation"
        )
        result["timed_region"] = (
            "CPU tensors ready -> H2D -> inference "
            "-> output tensor D2H; "
            "excludes model load and preprocessing"
        )
        result["instrumented"] = args.mode in (
            "tokens", "components", "memory",
        )
        peak = max(
            result["inference_peak_reserved_bytes"],
            result.get(
                "nvml_process_sampled_peak_bytes"
            ) or 0,
        )
        result["memory_guard_exceeded"] = (
            peak > args.memory_fraction * capacity
        )
        result["end_gpu_temperature_c"] = (
            pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU,
            )
        )
        result.update(status="ok")
        stage("complete")

    except Exception as exc:
        result["status"] = (
            "nonfinite_output"
            if isinstance(exc, NonfiniteOutput)
            else "gpu_oom"
            if "CUDA out of memory" in str(exc)
            else "runtime_error"
        )
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()

    finally:
        if hook:
            hook.remove()
        if sampler:
            sampler.stop()
            result["interrupted_memory_samples"] = (
                sampler.rows
            )
        now = time.perf_counter()
        result["timings_seconds"][phase_name] = now-phase_begin
        result["worker_main_seconds"] = now-main_begin
        save(args.output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare")
    parser.add_argument("--plan", default="/plan.json")
    parser.add_argument("--case")
    parser.add_argument(
        "--mode",
        choices=[
            "performance", "memory",
            "first_logits", "tokens", "components",
        ],
    )
    parser.add_argument("--modes", nargs="+", choices=MODE_ORDER)
    parser.add_argument("--mode-iterations-json", default="{}")
    parser.add_argument("--keep-allocator-cache", action="store_true")
    parser.add_argument("--max-start-temp", type=float, default=36)
    parser.add_argument("--idle-timeout", type=float, default=180)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--memory-fraction", type=float, default=0.85,
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.prepare:
        prepare(args.prepare)
    else:
        assert args.case and args.output
        assert args.iterations > 0
        assert args.warmup >= 0 and args.threads > 0
        if args.modes:
            run_shared(args)
        else:
            main(args)

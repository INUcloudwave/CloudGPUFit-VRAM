import json
from collections import Counter
from pathlib import Path

MODELS = {
    "decoder_llm": [
        "openai-community/gpt2",
        "openai-community/gpt2-medium",
        "openai-community/gpt2-xl",
    ],
    "vision_cls": [
        "google/mobilenet_v2_1.0_224",
        "microsoft/resnet-50",
        "google/vit-large-patch16-224",
    ],
    "text_encdec": [
        "google-t5/t5-small",
        "google-t5/t5-base",
        "google-t5/t5-3b",
    ],
}

FIELDS = {
    "decoder_llm": ("batch_size", "seq_len", "max_new_tokens"),
    "vision_cls": ("batch_size", "resolution"),
    "text_encdec": (
        "batch_size", "encoder_seq_len",
        "decoder_seq_len", "max_new_tokens",
    ),
}

DTYPES = ("fp16", "fp32", "bf16")


def make_shapes(base, axes, extras):
    """기본 조건 → 한 항목씩 변경 → 복합 조건. 중복은 제거."""
    shapes = [base]
    for index, values in enumerate(axes):
        for value in values:
            shape = list(base)
            shape[index] = value
            shapes.append(tuple(shape))
    return list(dict.fromkeys(shapes + extras))


SHAPES = {
    "decoder_llm": make_shapes(
        base=(1, 256, 64),
        axes=(
            (1, 2, 4, 8),
            (128, 256, 512, 768),
            (16, 64, 128, 256),
        ),
        extras=[
            (4, 512, 128),
            (4, 768, 128),
            (8, 128, 16),
            (1, 768, 256),
        ],
    ),
    "vision_cls": make_shapes(
        base=(1, 224),
        axes=((1, 2, 4, 8, 16), (128, 224, 384, 512)),
        extras=[
            (4, 384),
            (4, 512),
            (8, 128),
        ],
    ),
    "text_encdec": make_shapes(
        base=(1, 256, 1, 64),
        axes=(
            (1, 2, 4),
            (128, 256, 512, 1024),
            (1, 32, 128),
            (16, 64, 128, 256),
        ),
        extras=[
            (2, 512, 1, 128),
            (2, 1024, 1, 128),
            (4, 128, 1, 16),
            (1, 512, 128, 128),
        ],
    ),
}

SMOKE = {
    "decoder_llm": (1, 128, 16),
    "vision_cls": (1, 224),
    "text_encdec": (1, 128, 1, 16),
}

cases = []


def add_case(suite, kind, model, size, dtype, shape, index):
    params = dict(zip(FIELDS[kind], shape))
    assert len(params) == len(shape)
    assert all(isinstance(v, int) and v > 0 for v in shape)

    if kind == "decoder_llm":
        assert params["seq_len"] + params["max_new_tokens"] <= 1024
    if model == "google/vit-large-patch16-224":
        assert params["resolution"] == 224

    cases.append({
        "case_id": f"{suite}__{kind}__{size}__{dtype}__{index:02d}",
        "suite": suite,
        "workload_type": kind,
        "model": model,
        "size": size,
        "dtype": dtype,
        "params": params,
    })


for suite in ("smoke", "core"):
    for kind, models in MODELS.items():
        for model, size in zip(models, ("small", "medium", "large")):
            shapes = [SMOKE[kind]] if suite == "smoke" else SHAPES[kind]
            if model == "google/vit-large-patch16-224":
                shapes = [s for s in shapes if s[1] == 224]

            dtypes = ("fp16",) if suite == "smoke" else DTYPES
            for dtype in dtypes:
                for index, shape in enumerate(shapes, start=1):
                    add_case(suite, kind, model, size, dtype, shape, index)

counts = Counter(c["suite"] for c in cases)
by_type = Counter(
    c["workload_type"] for c in cases if c["suite"] == "core"
)

assert counts == {"smoke": 9, "core": 342}
assert by_type == {
    "decoder_llm": 126, "vision_cls": 81, "text_encdec": 135
}
assert len({c["case_id"] for c in cases}) == len(cases)
assert len({
    (c["suite"], c["model"], c["dtype"],
     tuple(c["params"].items()))
    for c in cases
}) == len(cases)

plan = {
    "schema_version": "1.0",
    "execution_profiles": {
        "local_smoke": {
            "suite": "smoke", "repeats": 1,
            "warmup": 1, "iterations": 3,
            "modes": ["smoke"],
        },
        "local_core": {
            "suite": "core", "repeats": 1,
            "warmup": 3, "iterations": 20,
            "modes": ["performance", "memory"],
        },
        "aws_core": {
            "suite": "core", "repeats": 3,
            "warmup": 10, "iterations": 100,
            "modes": ["performance", "memory"],
        },
    },
    "runner_settings": {
        "concurrency": 1,
        "seed": 20261006,
        "eval": True,
        "inference_mode": True,
        "quantization": False,
        "cpu_offload": False,
        "compile": False,
        "use_cache": True,
        "num_beams": 1,
        "do_sample": False,
        "min_new_tokens_equals_max_new_tokens": True,
        "decoder_prefix_includes_start_token": True,
        "memory_sampling_interval_ms": 50,
        "memory_stop_fraction": 0.85,
        "record_in_ram_then_save": True,
        "profiler_enabled": False,
        "fresh_process_per_case_repeat_mode": True,
        "silent_dtype_or_shape_fallback": False,
    },
    "counts": {
        "smoke": counts["smoke"],
        "core": counts["core"],
        "core_by_type": dict(by_type),
    },
    "cases": cases,
}

output = Path(__file__).with_name("benchmark_plan.json")
with output.open("x", encoding="utf-8") as file:
    json.dump(plan, file, ensure_ascii=False, indent=2)

print(f"생성 완료: {output}")
print(f"실행 확인 {counts['smoke']}개 / 본 실험 {counts['core']}개")

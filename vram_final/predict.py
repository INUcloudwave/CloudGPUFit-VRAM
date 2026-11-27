#!/usr/bin/env python
"""CloudGPUFit VRAM predictor (final method, experiments/inugpu-inuos 2026-11-23/26).

One request in, one VRAM prediction per supported AWS GPU out. CPU only, no GPU.

Method (docs/ai-context/REPORT_CONTEXT_KO.md §0.1, §5; code from cpu-vm/03-fp16-cpu-prediction/src/cgf680, copied unchanged):
  1. GPU load phase: the allocator requests of from_pretrained(fp16, device_map=cuda) + .to(fp16), synthesized from
     the checkpoint and model structure (cgf680.load_emu).
  2. ONE invocation (input copy + generate / forward) traced on the CPU with the PyTorch profiler [memory] events.
       - decoder LLM / text encoder-decoder: real FP16 execution on a CPU with FP16 instructions (fp16_real, AWS r7i
         etc.); otherwise FP32 execution with FP16 request sizes (fp32_shadow, INUOS method).
       - image classification: fp32_shadow + per-request layer labels (cgf680.layer_shadow) so the CPU-only
         convolution library buffers can be dropped; the conv model results were validated on this stream.
  3. The invocation request list is replayed 4 times (warmup 3 + measure 1) after the load requests in the xMem
     CachingAllocator (pinned commit 1ee4bd08, unmodified) with the descending segment address rule, the cuBLAS
     workspace inserted right after the first GEMM output (cgf680.gemm_rules / replay3), and the allocator-usable
     capacity of each GPU (retry / OOM reproduced).
       - conv models additionally: CPU conv library requests dropped, cuDNN workspace of each conv call from the
         GPU generation's measured table, cuBLASLt 1 MiB after linear, max_pool2d int64 indices (final_eval_pergpu.py).
  4. VRAM(GPU) = predicted reserved + outside(GPU) (memory outside the PyTorch allocator, measured once per GPU type).

Usage (inside the xmem-cpu container, xMem checkout at /work/xMem):
  echo '{"model": "facebook/opt-125m", "workload_type": "decoder_llm", "batch_size": 8, "seq_len": 512,
         "max_new_tokens": 128, "dtype": "fp16"}' | python /work/xMem/vram_final/predict.py --json -
Progress lines "CGF_PHASE <text>" go to stdout; the result is the last JSON object printed.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
XMEM_ROOT = Path(os.environ.get("CGF_XMEM_ROOT", HERE.parent)).resolve()
for p in (str(XMEM_ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

PREDICTION_VERSION = "cpu-trace-xmem-final-20261127"
DATA = HERE / "data"
MIB = 1 << 20
GIB = 1 << 30

# gpu_profiles.json key -> (AWS family, cuDNN table of the GPU generation). g7 (RTX PRO 4500) uses the g7e table:
# same generation, 60/60 on the conv conditions without its own measurement (pergpu/README_KO.md).
GPUS = {
    "t4": ("g4dn", "turing_t4"),
    "a10g": ("g5", "ampere_rtx3090"),
    "l4": ("g6", "ada_l4"),
    "l40s": ("g6e", "ada_l4"),
    "rtxpro4500": ("g7", "blackwell_rtxpro6000"),
    "rtxpro6000": ("g7e", "blackwell_rtxpro6000"),
}
SUPPORTED_KINDS = ("decoder_llm", "text_encdec", "vision_cls")
CONV_KEYS = ("N", "C", "H", "W", "K", "R", "stride", "pad", "dil", "groups")


class UserError(Exception):
    """Request that cannot be predicted (reported as status=error with this message)."""


def phase(text):
    print(f"CGF_PHASE {text}", flush=True)


# ----------------------------------------------------------------------------- request -> case
def build_case(req):
    kind = req.get("workload_type") or req.get("type")
    if kind not in SUPPORTED_KINDS:
        raise UserError(f"지원하지 않는 워크로드 유형입니다: {kind!r} (지원: {', '.join(SUPPORTED_KINDS)})")
    dtype = req.get("dtype", "fp16")
    if dtype != "fp16":
        raise UserError("VRAM 예측은 FP16만 지원합니다 (요청 dtype: %s)" % dtype)
    model = str(req["model"])
    b = int(req.get("batch_size") or 1)
    case = {"case_id": "request", "model": model, "short": model.split("/")[-1], "kind": kind, "dtype": "fp16",
            "batch_size": b, "input_len": None, "new_tokens": None, "resolution": None}
    if kind == "decoder_llm":
        case["input_len"] = int(req["seq_len"])
        case["new_tokens"] = int(req.get("max_new_tokens") or 1)
    elif kind == "text_encdec":
        case["input_len"] = int(req["encoder_seq_len"])
        case["new_tokens"] = int(req.get("max_new_tokens") or req.get("decoder_seq_len") or 1)
    else:
        res = req.get("resolution")
        if res is None:
            cfg = req.get("model_config") or {}
            res = cfg.get("image_size") or 224
        case["resolution"] = int(res if not isinstance(res, (list, tuple)) else res[0])
    for k in ("batch_size", "input_len", "new_tokens", "resolution"):
        if case[k] is not None and case[k] <= 0:
            raise UserError(f"{k}는 양의 정수여야 합니다")
    return case


def cpu_has_fp16():
    try:
        flags = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    return bool(re.search(r"\b(avx512_fp16|amx_fp16)\b", flags))


def choose_mode(case, requested):
    """-> (mode, reason). Vision is always fp32_shadow (conv results were validated on that stream)."""
    if case["kind"] == "vision_cls":
        return "fp32_shadow", "image classification: FP32 execution + FP16 sizes with layer labels (validated path)"
    requested = (requested or "auto").lower()
    if requested in ("fp16_real", "fp32_shadow"):
        return requested, f"forced by request/env ({requested})"
    if cpu_has_fp16():
        return "fp16_real", "CPU has FP16 instructions (avx512_fp16/amx_fp16): real FP16 execution"
    return "fp32_shadow", "CPU has no FP16 instructions: FP32 execution with FP16 request sizes"


# ----------------------------------------------------------------------------- model files
def fetch_model(model, token=None):
    """Download config + weights of the current Hub revision. -> (snapshot dir, revision, use_safetensors)."""
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(token=token)
    try:
        info = api.model_info(model, token=token)
    except Exception as exc:  # noqa: BLE001
        raise UserError(f"Hugging Face에서 모델 정보를 가져오지 못했습니다: {type(exc).__name__}: {str(exc)[:300]}")
    files = [s.rfilename for s in (info.siblings or [])]
    top = [f for f in files if "/" not in f]
    st = sorted(f for f in top if f.endswith(".safetensors"))
    if st:
        weights, use_st = st, True
    elif "pytorch_model.bin" in top:
        weights, use_st = ["pytorch_model.bin"], False
    else:
        raise UserError("지원하는 가중치 파일(model*.safetensors 또는 단일 pytorch_model.bin)이 없는 모델입니다")
    patterns = [f for f in top if f.endswith(".json")] + weights
    try:
        path = snapshot_download(model, revision=info.sha, allow_patterns=patterns, token=token)
    except Exception as exc:  # noqa: BLE001
        raise UserError(f"모델 파일을 내려받지 못했습니다: {type(exc).__name__}: {str(exc)[:300]}")
    return path, info.sha, use_st


# ----------------------------------------------------------------------------- CPU trace
def _mark(k):
    import torch
    from cgf680.cpu_worker import MARK_BASE
    x = torch.empty(MARK_BASE + 2 * k, dtype=torch.uint8)
    del x


def trace_request(case, pin, snap_dir, mode, threads, tmp_dir):
    """Load synthesis + ONE traced invocation. -> dict(load, act, size, aid, ws_rel, labels|None, info)."""
    import torch
    from cgf680 import load_emu
    from cgf680 import protocol2 as P
    from cgf680.cpu_worker import MARKERS, Session, normalize, parse_trace
    from cgf680.cpu_worker2 import REGION
    from cgf680.gemm_probe import record_hits
    from cgf680.gemm_shadow import GEMM_NAMES, FirstGemmRecorder, NamedShadow
    from cgf680.layer_shadow import CALLMARK, MarkShadow
    from cgf680.shadow import emulate_fp16, match_records, patch_fp16_branches

    torch.set_num_threads(threads)
    torch.manual_seed(P.SEED)
    info = {"cpu_mode": mode, "threads": torch.get_num_threads(), "torch": torch.__version__}
    vision = case["kind"] == "vision_cls"

    phase("GPU 적재 단계 요청 합성 중…")
    m16 = P.load(case, pin, torch.float16)
    lseq, lstats = load_emu.synthesize(m16, snap_dir, pin["use_safetensors"])
    load_events, _live = load_emu.to_events(lseq)
    info["load_synthesis"] = {k: lstats[k] for k in ("warmup_bytes", "loaded_into", "post_load_casts")}
    del m16
    gc.collect()

    exec_dtype = torch.float16 if mode == "fp16_real" else torch.float32
    m = P.finish_load(P.load(case, pin, exec_dtype), exec_dtype)
    bad = P.check_valid(case, m.config)
    if bad:
        raise UserError(f"입력 길이가 모델 한도를 넘습니다: {bad}")
    host, _sp = P.host_inputs(case, m.config)
    inputs = {k: (v.to(exec_dtype) if v.is_floating_point() else v) for k, v in host.items()}
    if not vision:
        P.generation_config(m, case)
    if mode == "fp16_real":
        shadow = FirstGemmRecorder()
    else:
        info["fp16_branch_patches"] = patch_fp16_branches(m)
        shadow = MarkShadow() if vision else NamedShadow()

    phase("CPU에서 모델 1회 실행하며 메모리 요청 기록 중…")
    t0 = time.perf_counter()
    gc.collect()
    gc.disable()
    sess = Session("user")
    sess.start()
    try:
        _mark(0)
        with torch.profiler.record_function(REGION):
            with shadow:
                with torch.inference_mode():
                    gi, o, new = P.invoke(m, case, inputs)
        _mark(4)
        del gi, o
        _mark(5)
    finally:
        sess.stop()
        gc.enable()
    info["trace_seconds"] = round(time.perf_counter() - t0, 1)
    if not vision and new != case["new_tokens"]:
        raise RuntimeError(f"generated {new} tokens, expected {case['new_tokens']}")

    phase("메모리 요청 목록 정리 중…")
    raw = Path(tmp_dir) / f"trace_{os.getpid()}.json"
    sess.export(str(raw))
    del sess
    gc.collect()
    try:
        regions, mem = parse_trace(raw)
    finally:
        raw.unlink(missing_ok=True)
    ev, mpos, nstats = normalize(regions, mem, len(MARKERS))
    del mem
    if [r[0] for r in regions] != [REGION] or not {0, 4, 5} <= set(mpos):
        raise RuntimeError(f"trace boundaries: regions={[r[0] for r in regions]} markers={sorted(mpos)}")

    labels = None
    if mode == "fp16_real":
        ev16 = {k: ev[k] for k in ("action", "size", "alloc_id")}
        idx = list(range(len(ev["action"]) + 1))
        hits = record_hits(ev, shadow.records)
        first = min(hits) if hits else None
        gemm_name = shadow.first_name
    elif vision:
        ev16, idx, mpos, first, gemm_name, labels = _vision_stream(ev, mpos, shadow, CALLMARK, GEMM_NAMES,
                                                                   match_records, emulate_fp16, record_hits)
    else:
        tags, buf_src, _ = match_records(ev, shadow.records)
        ev16, _est, idx = emulate_fp16(ev, tags, buf_src)
        hits = record_hits(ev, shadow.records)
        first = next((i for i in sorted(hits) if hits[i][0] == "out" and shadow.names.get(hits[i][1]) in GEMM_NAMES),
                     None)
        gemm_name = shadow.names.get(hits[first][1]) if first is not None else None
    p0, p5 = idx[mpos[0]], idx[mpos[5]]
    ws_rel = None if first is None else int(idx[first] - p0 + 1)
    info["first_gemm"] = {"func": gemm_name, "ws_insert_rel": ws_rel}
    info["requests"] = int((ev16["action"][p0:p5] == 1).sum())
    out = {"load": [(int(a), int(s), int(k)) for a, s, k in load_events],
           "act": ev16["action"][p0:p5].astype(np.int64).tolist(),
           "size": ev16["size"][p0:p5].astype(np.int64).tolist(),
           "aid": ev16["alloc_id"][p0:p5].astype(np.int64).tolist(),
           "ws_rel": ws_rel, "labels": None, "info": info}
    if labels is not None:
        lab = {k: v[p0:p5] for k, v in labels["per_event"].items()}
        out["labels"] = {"layer": lab["layer"].tolist(), "func": [labels["names"][i] for i in lab["func"]],
                         "call": lab["call"].tolist(), "conv": labels["conv"]}
    return out


def _vision_stream(ev, mpos, sh, callmark, gemm_names, match_records, emulate_fp16, record_hits):
    """cgf680.layer_shadow.label_case, from an in-memory trace: drop the call-end markers, label every request as
    model layer (output of a torch-function call) or library layer, attach the enclosing call and its function name
    and the conv parameters; plus the first GEMM output (cuBLAS workspace position)."""
    import bisect
    act, addr, size, aid = ev["action"], ev["addr"], ev["size"], ev["alloc_id"]
    markpos = [i for i in range(len(act)) if act[i] == 1 and int(size[i]) == callmark]
    mark_ids = {int(aid[i]) for i in markpos}
    keep = np.array([int(aid[i]) not in mark_ids for i in range(len(act))], dtype=bool)
    free_at = {}
    for i in range(len(act)):
        if act[i] == 0:
            free_at.setdefault(int(aid[i]), i)
    by_addr = {}
    for i in range(len(act)):
        if act[i] == 1 and keep[i]:
            by_addr.setdefault(int(addr[i]), []).append(i)
    model_layer = set()
    for kind, p, nb, _x, call in sh.records:
        if call - 1 >= len(markpos):
            continue
        end = markpos[call - 1]
        lst = by_addr.get(int(p), [])
        j = bisect.bisect_left(lst, end) - 1
        while j >= 0:
            i = lst[j]
            if int(size[i]) == int(nb) and free_at.get(int(aid[i]), 1 << 62) > end:
                model_layer.add(i)
                break
            j -= 1
    call_of = np.searchsorted(np.array(markpos, dtype=np.int64), np.arange(len(act)), side="right") + 1
    ev2 = {k: v[keep] for k, v in ev.items()}
    oldidx = np.nonzero(keep)[0]
    tags, bsrc, _ = match_records(ev2, list(sh.records))
    ev16, _, idx = emulate_fp16(ev2, tags, bsrc)
    mpos2 = {kk: int(np.searchsorted(oldidx, v)) for kk, v in mpos.items()}
    names = sorted(set(sh.names.values()) | {"?"})
    nidx = {n: i for i, n in enumerate(names)}
    lab, fn, cl = {}, {}, {}
    for j in range(len(ev2["action"])):
        if ev2["action"][j] == 1:
            o = int(oldidx[j])
            k = int(ev2["alloc_id"][j])
            lab[k] = 0 if o in model_layer else 1
            c = int(call_of[o])
            cl[k] = c
            fn[k] = nidx.get(sh.names.get(c, "?"), nidx["?"])
    id16 = ev16["alloc_id"]
    per_event = {"layer": np.array([lab.get(int(k), 0) for k in id16], np.int8),
                 "func": np.array([fn.get(int(k), nidx["?"]) for k in id16], np.int32),
                 "call": np.array([cl.get(int(k), 0) for k in id16], np.int32)}
    hits = record_hits(ev2, sh.records)
    first = next((i for i in sorted(hits) if hits[i][0] == "out" and sh.names.get(hits[i][1]) in gemm_names), None)
    gemm_name = sh.names.get(hits[first][1]) if first is not None else None
    labels = {"per_event": per_event, "names": names, "conv": {int(k): v for k, v in sh.conv.items()}}
    return ev16, idx, mpos2, first, gemm_name, labels


# ----------------------------------------------------------------------------- GPU-side rules
def load_cudnn_table(name):
    d = json.loads((DATA / "cudnn" / f"{name}.json").read_text())
    return {tuple(r[:-1]): r[-1] for r in d["rows"]}


def conv_key(cp):
    n, c, h, w = cp["input"]
    pad = cp["padding"] if isinstance(cp["padding"], list) else [0, 0]
    return (n, c, h, w, cp["weight"][0], cp["weight"][2], cp["stride"][0], pad[0], cp["dilation"][0], cp["groups"])


def conv_model_stream(tr, table):
    """results/reserved_v3/cudnn_fit/final_eval_pergpu.py stream(drop_conv=True, tab=table, lt=True, pool=True)."""
    lab = tr["labels"]
    act, size, aid = tr["act"], tr["size"], tr["aid"]
    layer, func, call, conv = lab["layer"], lab["func"], lab["call"], lab["conv"]
    conv_funcs = {"conv2d", "conv1d", "conv3d", "_convolution", "convolution"}
    drop = {aid[i] for i in range(len(act)) if act[i] == 1 and layer[i] == 1
            and (func[i] in conv_funcs or func[i] == "max_pool2d")}
    A, S, K, nxt, seen, wr = [], [], [], -1, set(), None
    missing, conv_calls = 0, 0
    for i in range(len(act)):
        if tr["ws_rel"] is not None and i == tr["ws_rel"]:
            wr = len(A)
        if aid[i] in drop:
            continue
        A.append(act[i]); S.append(size[i]); K.append(aid[i])
        c = call[i]
        is_conv = func[i] in conv_funcs
        if act[i] == 1 and layer[i] == 0 and c not in seen and (is_conv or func[i] in ("linear", "max_pool2d")):
            seen.add(c)
            w = 0
            if is_conv and c in conv:
                conv_calls += 1
                key = conv_key(conv[c])
                if key in table:
                    w = table[key]
                else:
                    missing += 1
            elif func[i] == "linear":
                w = MIB                      # cuBLASLt workspace (transient)
            elif func[i] == "max_pool2d":
                w = 4 * int(size[i])         # int64 indices of the GPU max_pool2d (transient)
            if w:
                A += [1, 0]; S += [w, w]; K += [nxt, nxt]; nxt -= 1
    if tr["ws_rel"] is not None and wr is None:
        wr = len(A)
    return {"load": tr["load"], "act": A, "size": S, "aid": K}, wr, {"conv_calls": conv_calls,
                                                                     "conv_shapes_missing_in_table": missing}


def load_cudnn_candidates(name):
    d = json.loads((DATA / "cudnn_candidates" / f"{name}.json").read_text())
    return {tuple(r[:10]): r[10] for r in d["rows"]}


def conv_model_stream_fallback(tr, cands):
    """Same stream as conv_model_stream, but each conv call's workspace is a replay5 action-2 entry that points at the
    conv shape's candidate list (results/reserved_v3/fallback/fallback_eval.py streams()). replay5 uses candidate 1
    while it fits; under memory pressure it switches to the next candidate like PyTorch 2.6 / cuDNN (Conv_v8.cpp)."""
    lab = tr["labels"]
    act, size, aid = tr["act"], tr["size"], tr["aid"]
    layer, func, call, conv = lab["layer"], lab["func"], lab["call"], lab["conv"]
    conv_funcs = {"conv2d", "conv1d", "conv3d", "_convolution", "convolution"}
    drop = {aid[i] for i in range(len(act)) if act[i] == 1 and layer[i] == 1
            and (func[i] in conv_funcs or func[i] == "max_pool2d")}
    A, S, K, nxt, seen, wr, keys = [], [], [], -1, set(), None, []
    for i in range(len(act)):
        if tr["ws_rel"] is not None and i == tr["ws_rel"]:
            wr = len(A)
        if aid[i] in drop:
            continue
        A.append(act[i]); S.append(size[i]); K.append(aid[i])
        c = call[i]
        is_conv = func[i] in conv_funcs
        if act[i] == 1 and layer[i] == 0 and c not in seen and (is_conv or func[i] in ("linear", "max_pool2d")):
            seen.add(c)
            if is_conv and c in conv:
                key = conv_key(conv[c])
                if key not in keys:
                    keys.append(key)
                A.append(2); S.append(keys.index(key)); K.append(0)
                continue
            w = MIB if func[i] == "linear" else (4 * int(size[i]) if func[i] == "max_pool2d" else 0)
            if w:
                A += [1, 0]; S += [w, w]; K += [nxt, nxt]; nxt -= 1
    if tr["ws_rel"] is not None and wr is None:
        wr = len(A)
    return {"load": tr["load"], "act": A, "size": S, "aid": K}, wr, [cands.get(k, [0]) for k in keys]


def is_conv_model(tr):
    """Conv-model recipe for vision models with real conv stacks; a ViT has a single patch-embedding conv and was
    validated with the plain stream (LLM/ViT group)."""
    return tr["labels"] is not None and len(tr["labels"]["conv"]) >= 2


def predict_gpus(tr, profiles):
    from cgf680 import replay3, replay5
    from cgf680.gpu_rules import cublas_workspace
    results, cache, notes = {}, {}, []
    conv = is_conv_model(tr)
    tables, fallback_streams = {}, {}
    for key, (family, table_name) in GPUS.items():
        p = profiles[key]
        ws = cublas_workspace(p["gpu_name"], p.get("compute_capability"), p.get("runtime_torch", "2.6.0"))
        if conv:
            if table_name not in tables:
                tables[table_name] = conv_model_stream(tr, load_cudnn_table(table_name))
            st, wr, cinfo = tables[table_name]
            stream_key = table_name
        else:
            st, wr, cinfo = {"load": tr["load"], "act": tr["act"], "size": tr["size"], "aid": tr["aid"]}, tr["ws_rel"], None
            stream_key = "plain"
        ck = (stream_key, ws["bytes"])
        if ck not in cache:
            cache[ck] = replay3.replay(st, replay3.CAP_INF, "descending", wr, ws["bytes"])
        inf = cache[ck]
        cap = int(p["capacity_bytes"])
        outside = int(p["non_allocator_bytes"])
        row = {"gpu": p["gpu_name"], "profile_key": key, "aws_family": family, "architecture": p["architecture"],
               "cuda_total_memory_bytes": int(p["cuda_total_memory"]), "allocator_capacity_bytes": cap,
               "outside_estimate_bytes": outside, "cublas_workspace_bytes": ws["bytes"],
               "cudnn_table": table_name if conv else None}
        if cinfo:
            row.update(cinfo)
        if inf.get("oom"):
            raise RuntimeError(f"unbounded replay hit OOM: {inf['oom']}")
        fallbacks = 0
        if inf["max_reserved"] <= cap:
            reserved, retry, oom = inf["m0_start_reserved"], False, False
        elif conv:
            # memory pressure on a conv model: cuDNN plan fallback (replay5). With enough memory replay5 equals
            # replay3 with candidate-1 workspaces (108/108 conditions), so it is only needed here.
            if table_name not in fallback_streams:
                fallback_streams[table_name] = conv_model_stream_fallback(tr, load_cudnn_candidates(table_name))
            s5, w5, cl = fallback_streams[table_name]
            r = replay5.replay(s5, cap, cl, "descending", w5, ws["bytes"])
            if r["oom"]:
                reserved, retry, oom = None, False, True
            else:
                # the unbounded replay needs more than cap, so this run released cached segments (retry) to fit
                reserved, retry, oom = r["m0_start_reserved"], True, False
                fallbacks = len(r["fallback"]["keys"])
        else:
            r = replay3.replay(st, cap, "descending", wr, ws["bytes"])
            if r["oom"]:
                reserved, retry, oom = None, False, True
            else:
                reserved, retry, oom = r["m0_start_reserved"], r["retry_released"] > 0, False
        if oom:
            row.update(status="gpu_oom", reserved_bytes=None, required_vram_bytes=None,
                       note="할당자 시뮬레이션에서 이 GPU의 사용 가능 용량을 넘어 OOM")
        else:
            row.update(status="ok", reserved_bytes=int(reserved), allocated_peak_bytes=int(inf["max_allocated"]),
                       allocator_retry=bool(retry), cudnn_plan_fallbacks=fallbacks, estimated_nvml_bytes=int(reserved) + outside,
                       required_vram_bytes=int(reserved) + outside,
                       required_vram_gb=round((int(reserved) + outside) / GIB, 3))
        results[p["gpu_name"]] = row
    if conv:
        miss = max(v[2]["conv_shapes_missing_in_table"] for v in tables.values())
        if miss:
            notes.append(f"합성곱 형태 {miss}개가 측정한 cuDNN 표에 없어 작업 공간 0으로 두었습니다(과소예측 가능).")
    return results, notes


# ----------------------------------------------------------------------------- main
def predict(req):
    t_all = time.perf_counter()
    case = build_case(req)
    mode, reason = choose_mode(case, req.get("cpu_mode") or os.environ.get("CGF_VRAM_CPU_MODE"))
    threads = int(req.get("threads") or os.environ.get("CGF_THREADS") or min(os.cpu_count() or 4, 20))
    token = req.get("hf_token") or os.environ.get("HF_TOKEN") or None

    from cgf_xmem_check import verify_xmem
    xmem = verify_xmem(XMEM_ROOT, DATA / "xmem-allocator-lock.json")

    phase("모델 파일 내려받는 중…")
    snap_dir, rev, use_st = fetch_model(case["model"], token)
    pin = {"revision": rev, "use_safetensors": use_st}
    import tempfile
    with tempfile.TemporaryDirectory(prefix="cgf-vram-") as tmp:
        tr = trace_request(case, pin, snap_dir, mode, threads, tmp)
    gc.collect()

    phase("GPU별 할당자 시뮬레이션 중…")
    profiles = json.loads((DATA / "gpu_profiles.json").read_text())["profiles"]
    by_gpu, notes = predict_gpus(tr, profiles)
    ok = [v for v in by_gpu.values() if v["status"] == "ok"]
    if not ok:
        raise UserError("지원하는 6개 GPU 모두에서 메모리 부족(OOM)으로 예측됩니다. 배치나 입력 길이를 줄여 보세요.")
    top = max(ok, key=lambda v: v["required_vram_bytes"])
    reserved_vals = sorted({v["reserved_bytes"] for v in ok})
    warnings = [
        "VRAM = 예측 reserved(PyTorch 할당자) + GPU별 할당자 밖 사용량(outside, GPU 종류별 실측 상수). FP16, 단일 요청 기준.",
        f"CPU 실행 방식: {reason}.",
    ] + notes
    phase("결과 계산 중…")
    return {
        "status": "ok",
        "prediction_version": PREDICTION_VERSION,
        "model": case["model"], "revision": rev, "workload_type": case["kind"],
        "case": {k: case[k] for k in ("batch_size", "input_len", "new_tokens", "resolution")},
        "cpu_mode": mode,
        "predicted_reserved_bytes": top["reserved_bytes"],
        "predicted_allocated_bytes": top["allocated_peak_bytes"],
        "required_vram_bytes": top["required_vram_bytes"],
        "required_vram_gb": round(top["required_vram_bytes"] / GIB, 2),
        "required_vram_scope": "maximum_among_supported_gpus",
        "reserved_range_bytes": [reserved_vals[0], reserved_vals[-1]],
        "gpu_requirements": by_gpu,
        "breakdown_bytes": {"predicted_reserved_max_gpu": top["reserved_bytes"],
                            "outside_gpu_estimate": top["outside_estimate_bytes"], "fixed_margin": 0},
        "trace": tr["info"], "xmem": {"commit": xmem["commit"], "ok": xmem["ok"]},
        "warnings": warnings,
        "elapsed_seconds": round(time.perf_counter() - t_all, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="request JSON file, or - for stdin")
    a = ap.parse_args()
    raw = sys.stdin.read() if a.json == "-" else Path(a.json).read_text()
    try:
        req = json.loads(raw)
        result = predict(req)
        code = 0
    except UserError as exc:
        result, code = {"status": "error", "error": str(exc)}, 1
    except BaseException as exc:  # noqa: BLE001
        msg = str(exc)
        if isinstance(exc, MemoryError) or "can't allocate memory" in msg or "DefaultCPUAllocator" in msg:
            err = "CPU 메모리 부족으로 예측하지 못했습니다. 더 작은 배치/입력으로 시도하거나 RAM이 큰 인스턴스를 쓰세요."
        elif "not implemented for 'Half'" in msg:
            err = "이 CPU에서 FP16 실행이 지원되지 않는 연산이 있습니다 (CGF_VRAM_CPU_MODE=fp32_shadow로 다시 시도)."
        else:
            err = f"{type(exc).__name__}: {msg[:1500]}"
        result, code = {"status": "error", "error": err, "traceback": traceback.format_exc()[-4000:]}, 1
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())

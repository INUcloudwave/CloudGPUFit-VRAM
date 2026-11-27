# 03. xMem 검증 프로브 · reserved 오차 원인 분리용 GPU trace 수집 (INUGPU)

> 레포 경로 대응: VM `xmem-work/` → 레포 `gpu-vm/03-xmem-probes/` (xMem 원본 저장소 제외, 로컬 수정분은 `xMem-local-changes/`), `cgf-reserved-diag-20261110/` → `gpu-vm/04-reserved-diag/`. 대용량 제외 파일은 `EXCLUDED_LARGE_FILES.md` 참고.


- 작성 기준: 2026-11-15, 파일에서 확인된 내용만 기록 (해석·추정은 "미확인"으로 표시)
- 경로는 모두 `/home/inugpu` 기준 상대경로
- 날짜는 파일 mtime 기준 (별도 기록이 있는 경우 그 값 사용)
- 대상
  - Scope A: `xmem-work/` (xMem 단독 검증 스크립트, FP16 allocator 재생 실험)
  - Scope B: `cgf-reserved-diag-20261110/` (INUOS 요청 GPU allocator trace 수집)

---

## Scope A. `xmem-work/`

### A-1. 개요

- 목적
  - xMem(PyTorch CUDA caching allocator의 CPU 재구현)이 실제 GPU allocator와 같은 결정을 내리는지 단독 검증
  - GPU ground truth(peak allocated/reserved, NVML peak) 수집용 소형 스크립트 모음
  - 2026-11-11: FP16 추론 GPU allocator 이력을 xMem에 그대로 재생해 reserved/allocated 일치 여부 확인
- 기간 (mtime)
  - 1차: 2026-08-16 ~ 2026-08-23 (top-level 스크립트, `results/`)
  - 2차: 2026-11-11 14:34 ~ 16:09 (`cgf-fp16-replay-20261111/`)
- 실행 환경 (코드에서 확인)
  - 스크립트 대부분 컨테이너 내 `/work`, `/work/xMem` 경로를 가정 (`sys.path.insert(0, "/work/xMem")`, `/work/gt_result.json` 저장)
  - `gt_generate.py`, `gt_result.json`, `results/` 는 root 소유 → 컨테이너 안에서 생성된 것으로 보임 (미확인)

### A-2. `xMem/` (upstream 저장소, 내부는 문서화 대상 아님)

| 항목 | 값 |
|---|---|
| 정체 | xMem upstream 저장소 clone (`perf_estimator/allocator` 등 포함) |
| remote | `https://github.com/Stone-ResearchLife/xMem.git` (fetch/push) |
| HEAD | `1ee4bd08dda277a6248abee5b6d40e0d6067efaa` / 2026-07-26 12:57:38 +0100 / `feat(dp): enhance rediection output capability` |
| 로컬 수정 | `M exp/trainer/train_loop.py`, `M exp/trainer/trainer.py`, `M perf_estimator/estimator.py` |
| 미추적 파일 | `perf_estimator/estimator.py.bak`, `predict_vram.py` |
| 크기 | 디렉터리 336M, `.git/objects/pack/pack-91b3…c9c5b.pack` 약 86.5MiB (**>50MB**) |

- Scope B manifest의 기준 xMem commit(`1ee4bd0…`)과 HEAD 동일
- `predict_vram.py`(미추적)는 `gt_generate.py`, `gt_synth.py`가 import (`load_model`, `build_loader`)
- 로컬 수정 내용은 이번 문서 범위 밖 (git에 올리면 upstream과 달라지는 점 유의)

### A-3. 실험·스크립트 목록

| 파일 | 날짜 | 구분 | 한 줄 요약 |
|---|---|---|---|
| `bench_infer.py` | 08-16 | GT/트레이스 | CPU: Profiler trace 생성 / CUDA: NVML 1ms 샘플링 peak |
| `gt_run.py` | 08-16 | GT(학습) | xMem `FastRunner.train_on_gpu()` → NVML delta peak + snapshot segment peak |
| `gt_fast.py` | 08-16 | GT(학습) | `ModelTrainer`로 GPU 학습 → NVML peak + torch peak, `CUDNN_MODE` 전환 |
| `gt_result.json` | 08-16 | 결과 | `gt_fast.py` 출력 (ResNet152, SGD, bs32) |
| `results/gt_distilgpt2_bs1.json` | 08-16 | 결과 | `bench_infer.py --device cuda` 출력 |
| `probe_cnn_alloc.py` | 08-16 | 프로브 | CNN 모델 로드/입력/forward 단계별 allocated·reserved 분해 |
| `probe_allocator_match.py` | 08-16 | 프로브 | 고정 alloc/free 시나리오 4종을 GPU와 xMem에 동시 실행, 단계별 비교 |
| `probe_allocator_replay.py` | 08-16 | 프로브 | distilgpt2 bs30 forward×3 GPU 이력 → xMem 재생, peak 비교 |
| `probe_gpu_allocator_trace.py` | 08-16 | 프로브 | 같은 조건 GPU 이력의 action/size/segment 히스토그램 출력 |
| `dtype_ratio.py` | 08-16 | GT | 동일 모델 fp32/fp16/bf16 no_grad forward peak 비교 |
| `gt_synth.py` | 08-22 | GT(추론) | `predict_vram` 모델·로더로 forward×3 peak (dtype·크기 지정) |
| `gt_generate.py` | 08-23 | GT(추론) | LLM `generate()` (KV cache) peak |
| `cgf-fp16-replay-20261111/` | 11-11 | 재생 실험 | FP16 추론 GPU allocator trace 13개 → xMem 재생 |

### A-4. 코드 설명 (파일별)

#### `bench_infer.py` (xMem 추론 검증용 공용 벤치)
- 정책: prefill only, `use_cache=False`, FP32 고정, warmup 2 / profile 3 iter, seq_len 128
- 모델: LLM 3종(distilgpt2, Qwen3-0.6B, DeepSeek-R1-Distill-Qwen-1.5B, `from_config` 랜덤 가중치), vision 3종(resnet152, convnext_tiny, mobilenet_v2)
- `--device cpu`: `torch.profiler`(profile_memory, record_shapes, with_stack) → `trace_<model>_bs<b>.json` (xMem 입력)
- `--device cuda`: NVML 샘플러 스레드(1ms) → `gt_<model>_bs<b>.json`
  - 출력 키: `ctx_overhead_bytes`(모델 로드 전 NVML used), `nvml_peak_bytes`, `nvml_peak_minus_ctx_bytes`, `torch_max_reserved/allocated_bytes`
- 실행: `python bench_infer.py --device {cpu|cuda} --model <name> --batch <b> --outdir /work/results`

#### `gt_run.py` / `gt_fast.py` (학습 워크로드 GT)
- 입력: `<model> <optimiser> <batch>` (argv)
- `gt_run.py`: xMem `FastRunner.train_on_gpu()` 실행 → NVML 기록 파일에서 시작값 대비 delta peak, `SnapshotAnalyser`로 snapshot segment peak → `/work/gt_result.json`
- `gt_fast.py`: `ModelTrainer`(iterations=0) + `HostMonitorPlugin`(10ms) 직접 구성 → `nvml_peak`, `reserved_peak`, `allocated_peak`, `framework_mem`(empty_cache 후 GPUtil used) → `/work/gt_result.json`
  - 환경변수 `CUDNN_MODE` = default / deterministic / off

#### `dtype_ratio.py`
- `FastRunner` 모델을 deepcopy 후 fp32/fp16/bf16 각각 forward×2 → max allocated/reserved
- fp16·bf16의 allocated를 fp32 대비 비율(`ratio_vs_fp32`)로 출력 (stdout만, 저장 없음)

#### `gt_synth.py` / `gt_generate.py`
- `xMem/predict_vram.py`의 `load_model`, `build_loader` 사용
- `gt_synth.py <model> <batch> <dtype> <size>`: LLM이면 seq_len, vision이면 해상도. forward 3회, `use_cache=False`
- `gt_generate.py <model> <batch> <dtype> <seq_len> <max_new>`: LLM 전용, greedy `generate(use_cache=True)` 1회
- 출력: `allocated_peak`, `reserved_peak` JSON (stdout)

#### `probe_cnn_alloc.py <model> <batch>`
- 단계: baseline → 모델 `.to(cuda)` → 입력 이동 → forward(no_grad)
- 각 단계 `memory_allocated/reserved`, `requested_bytes`, `allocated_bytes` 기록
- 파생값: `model_over_raw`(실할당 − param+buffer 원크기), `input_over_raw`, `forward_extra_peak`

#### `probe_allocator_match.py`
- 시나리오 4종을 실제 GPU(`torch.empty(uint8)`)와 xMem `CachingAllocator`에 같은 순서로 적용
  - `small_pool`(100K/900K/free/200K), `large_20mb`(2M/6M/7M), `large_over_10mb`(11M/13M), `fragmentation`(6M×3, free b, 7M)
- 매 단계 REAL vs SIM allocated/reserved, `RES_DIFF` 출력 (stdout)

#### `probe_gpu_allocator_trace.py` / `probe_allocator_replay.py`
- 공통: distilgpt2, bs30, eval forward×3 (`TRACE_FORWARDS` 기본 3), `_record_memory_history(stacks="python")`
- `probe_gpu_allocator_trace.py`: action 수·bytes, stream 분포, alloc/segment 크기 히스토그램, 처음·마지막 30 이벤트 출력
- `probe_allocator_replay.py`
  - `free_requested`→`free_completed` 지연 분석 (사이에 끼는 alloc 수, 최대 gap)
  - GPU `alloc`→xMem `malloc`, `free_completed`→xMem `free` 재생
  - 출력: real/sim peak allocated·reserved 차이, reserved 오차 %

#### `cgf-fp16-replay-20261111/xmem_replay.py`
- 실행: `CUDA_VISIBLE_DEVICES= python xmem_replay.py <xmem_repo> <trace.json> <out.json>` (CPU only)
- 입력: GPU worker(`cgf-fp16-vram-worker/1`, mode `trace`)가 만든 `trace.json` + `trace.snapshot.pickle.gz`
- 흐름
  - `alloc`(원요청 크기) → xMem `malloc(device, stream, size)`, `free_completed` → xMem `free`
  - GPU `segment_alloc/free`는 재생하지 않고 개수·크기만 집계 (xMem이 자체 segment 결정)
  - worker가 기록한 경계(`allocation.all.allocated/freed` 카운터)를 이벤트 위치에 매핑
  - 창(window)별 xMem start/peak/end reserved·allocated vs GPU 값 비교 (GPU peak는 measure 창만 존재)
- 출력: `out/.../<condition>.json` (+ 실행 로그 `.log`)

### A-5. `cgf-fp16-replay-20261111/` 실험 설계

- 입력 trace 공통 조건 (`in/*/trace.json` 기록값)
  - RTX 3090, torch 2.6.0+cu124, transformers 4.52.4, python 3.11.11, allocator backend native
  - `CUDA_MODULE_LOADING=LAZY`(driver 조회도 LAZY), `PYTORCH_CUDA_ALLOC_CONF` 미설정, TF32·cudnn.benchmark off
  - 파라미터 dtype float16, 프로토콜 warmup 3 / measure 1, iteration 사이 allocator 캐시 유지
  - measure 직전 `reset_peak_memory_stats()`
- 조건 그룹 (디렉터리)

| 그룹 | 시각 | attention | 조건 |
|---|---|---|---|
| `smoke_gpt2` | 11-11 14:34 | eager | gpt2 b1 in32 new32 |
| top-level (`in/<cond>`) | 11-11 15:01 | eager | gpt2·opt-125m·qwen2.5-1.5b-instruct·t5-small b8 in128 new128, resnet-50 b8 r224 |
| `rtx3090__attn-default` | 11-11 16:05~16:07 | sdpa (resnet-50·t5-small은 eager) | 위 5개 + gpt2 b16 in512 new512 |
| `rtx3090` | 11-11 16:08 | eager | gpt2 b16 in512 new512 |

- trace 생성 worker·실행 스크립트는 이 디렉터리에 없음
  - 관련 기록: `cgf-fp16-vram-20261111/README_KO.md`, `cgf-fp16-vram-20261111/results/REPORT_rtx3090_pilot_eager_vs_default_KO.md` (별도 문서 범위)
  - README에 기록된 실행 형태: `docker exec -e CUDA_VISIBLE_DEVICES= xmem-gpu python /work/cgf-fp16-replay-20261111/xmem_replay.py /work/xMem <trace.json> <out.json>`

### A-6. 결과 요약

#### `cgf-fp16-replay-20261111/out/` (measure_0 창, MiB = bytes/2^20)

| 그룹 | 조건 | 이벤트 수 | segment GPU/xMem | load 후 reserved (GPU=xMem) | peak reserved GPU / xMem | peak allocated GPU / xMem |
|---|---|---|---|---|---|---|
| smoke | gpt2 b1 in32 new32 | 179,240 | 12 / 12 | 474.0 | 478.0 / 478.0 | 265.8 / 265.8 |
| eager | gpt2 b8 | 732,151 | 11 / 11 | 474.0 | 476.0 / 476.0 | 418.4 / 418.4 |
| eager | opt-125m b8 | 641,247 | 4 / 4 | 316.0 | 418.0 / 418.0 | 388.1 / 388.1 |
| eager | qwen2.5-1.5b b8 | 2,261,106 | 62 / 62 | 3448.0 | 3512.0 / 3512.0 | 3051.9 / 3051.9 |
| eager | t5-small b8 | 819,144 | 36 / 36 | 272.0 | 304.0 / 304.0 | 154.2 / 154.2 |
| eager | resnet-50 b8 r224 | 4,641 | 12 / 12 | 82.0 | 132.0 / 132.0 | 107.4 / 107.4 |
| rtx3090 (eager) | gpt2 b16 in512 new512 | 2,925,574 | 26 / 26 | 474.0 | 1982.0 / 1982.0 | 1489.9 / 1489.9 |
| attn-default | gpt2 b8 | 613,867 | 11 / 11 | 474.0 | 476.0 / 476.0 | 417.7 / 417.7 |
| attn-default | opt-125m b8 | 559,827 | 4 / 4 | 316.0 | 418.0 / 418.0 | 388.1 / 388.1 |
| attn-default | qwen2.5-1.5b b8 | 2,081,393 | 61 / 61 | 3448.0 | 3510.0 / 3510.0 | 3048.6 / 3048.6 |
| attn-default | t5-small b8 | 819,144 | 36 / 36 | 272.0 | 304.0 / 304.0 | 154.2 / 154.2 |
| attn-default | resnet-50 b8 r224 | 4,641 | 12 / 12 | 82.0 | 132.0 / 132.0 | 107.4 / 107.4 |
| attn-default | gpt2 b16 in512 new512 | 2,452,472 | 24 / 24 | 474.0 | 1838.0 / 1838.0 | 1490.8 / 1490.8 |

- 13개 전부: `missing_frees` 0, 경계 10/10 위치 확인, GPU segment 수 = xMem segment 수
- 13개 전부: measure_0 peak reserved·allocated가 GPU와 바이트 단위 동일, warmup 창 start/end도 동일
- 재생 소요: 0.1 s (resnet-50) ~ 54.0 s (qwen2.5-1.5b eager)
- 즉, 같은 요청 순서를 주면 xMem allocator 결정이 PyTorch 2.6 native allocator와 일치 (이 13개 조건 한정)

#### 1차(08월) 결과 파일

| 파일 | 조건 | 값 |
|---|---|---|
| `results/gt_distilgpt2_bs1.json` | distilgpt2 FP32 bs1 seq128, prefill | ctx 783,220,736 / NVML peak 1,196,359,680 / peak−ctx 413,138,944 / torch reserved 404,750,336 / allocated 370,046,464 |
| `gt_result.json` | ResNet152 SGD bs32 학습 (`gt_fast.py`) | nvml_peak 333,447,168 / reserved_peak 325,058,560 / allocated_peak 298,906,112 / framework_mem 603,979,776 / samples 31 / oom false |

- probe 스크립트 4종, `dtype_ratio.py`, `gt_synth.py`, `gt_generate.py`는 stdout 출력만 하며 결과 파일 없음 → 수치 미확인

### A-7. 결과 파일 위치

- `xmem-work/results/gt_distilgpt2_bs1.json`, `xmem-work/gt_result.json`
- `xmem-work/cgf-fp16-replay-20261111/in/<group>/<cond>/trace.json` (13~16KB, 조건·환경·경계·measure 값)
- `xmem-work/cgf-fp16-replay-20261111/in/<group>/<cond>/trace.snapshot.pickle.gz` (24KB ~ 12.8MB, 원본 allocator 이력)
- `xmem-work/cgf-fp16-replay-20261111/out/<group>/<cond>.{json,log}` (재생 결과)
- Scope A 내 50MB 초과 파일: `xMem/.git/objects/pack/*.pack`(약 86.5MiB)만 해당. `cgf-fp16-replay-20261111/` 전체 59M, 단일 파일 최대 약 12.2MiB

### A-8. 한계 / 미확인

- 08월 probe 실행 결과(REAL vs SIM 차이, free 지연 통계 등)는 파일로 남지 않음
- `bench_infer.py --device cpu` trace 파일, `gt_run.py` 결과는 이 디렉터리에 없음
- `gt_result.json`의 `result_dir`는 컨테이너 경로(`/root/DL-Estimator/...`), 원본 NVML 기록 위치 미확인
- FP16 재생은 GPU 이력의 요청을 그대로 넣은 것 → CPU trace 기반 예측 오차(요청 자체 차이)는 검증 범위 밖
- trace 생성 worker 코드는 `cgf-fp16-replay-20261111/` 안에 없음 (sha256만 `trace.json`에 기록)
- xMem 로컬 수정(3개 파일)이 재생 결과에 영향 있는지 미확인 (allocator 파일 자체 수정 여부는 `git status` 상 표시 없음)

---

## Scope B. `cgf-reserved-diag-20261110/`

### B-1. 개요

- 요청자: INUOS(CPU VM), 요청서 작성 2026-11-10 (`handoff/GPU_COLLECTION_REQUEST.md`)
- 배경: CPU 예측(xMem 시뮬레이션) reserved peak가 GPU 실측보다 크게 작음
  - 요청서 예시: 이전 L4 기준 gpt2 FP16 b4/768/128 → CPU 928MiB, GPU 1678MiB
- 분리하려는 원인 후보 (manifest `question`)
  - Q1 초기 캐시·실행 이력(로딩·검증·워밍업·반복) 차이가 reserved를 얼마나 바꾸는가
  - Q2 같은 요청에 대해 xMem과 PyTorch 2.6 allocator가 같은 결정을 하는가
  - Q3 이력을 맞춘 뒤 잔차가 GPU 전용 요청(cuBLAS workspace 등)·중간 텐서·해제 시점 중 어디서 오는가
- INUGPU 역할: 실제 allocator 이력 원본(`_snapshot()` pickle)과 단계 경계 수집·반환. 분석은 INUOS가 수행
- 실행 일시: 2026-11-10 19:57:52 ~ 20:04:34 (+09:00), 반환 tarball 2026-11-11 09:14

### B-2. 실험 설계

- 환경 요구 (요청서 §2) 및 실제 기록값 (`run_meta.json`)

| 항목 | 요구 | 실제 |
|---|---|---|
| torch / transformers / python | 2.6.0+cu124 / 4.52.4 / 3.11.11 | 동일 |
| `CUDA_MODULE_LOADING` | LAZY (torch import 전) | env LAZY, driver 조회 `{rc:0, mode:LAZY}` |
| `PYTORCH_CUDA_ALLOC_CONF`, `CUBLAS_WORKSPACE_CONFIG`, `CUBLASLT_WORKSPACE_SIZE` | 미설정 | 모두 None |
| attention / TF32 / cudnn.benchmark | eager / False / False | 동일 |
| seed / threads | 20261006 / 8 | 동일 |
| GPU | 단일 GPU, 타 프로세스 없음 | RTX 3090 (total 25,306,529,792 B), driver 570.211.01, 실행 전 compute app 없음 확인 |
| 격리 | (case, run)마다 새 프로세스 | run마다 새 docker 컨테이너 |

- Case 3종 (`handoff/common_manifest.json`)

| case_id | 모델 @revision | dtype | batch | 입력 | 생성 |
|---|---|---|---|---|---|
| `core__decoder_llm__small__fp16__01` | openai-community/gpt2 @607a30d | FP16 | 1 | 256 | 64 |
| `core__decoder_llm__small__fp16__12` | openai-community/gpt2 @607a30d | FP16 | 4 | 768 | 128 |
| `core__decoder_llm__medium__fp32__12` | openai-community/gpt2-medium @6dcaa7a | FP32 | 4 | 768 | 128 |

- 입력: `torch.Generator().manual_seed(20261006)` 기반 randint, special token → 3 치환, sha256을 manifest 기대값과 대조
- 생성: greedy, `min_new_tokens = max_new_tokens`, `use_cache=True`
- Run 구성 (각각 fresh process, 3 case × 4 run = 12 + 선택 1)

| run_id | 그룹 | 단계 | 목적 |
|---|---|---|---|
| `R0_aws_replica` | 기준 | load → perf/finite×1 → perf/warmup×1 → perf/measure×5 → memory/between_modes×1 → memory/finite×1 → memory/warmup×1 → memory/measure×3 | 이전 AWS 구조(bench_worker_v3) 재현 |
| `R1_controlled` | 기준 | load → finite×1 → warmup×1 → measure×3 | CPU와 맞출 통제 실행 |
| `R2_cold_single` | 기준 | load → measure×1 | 로딩 이력과 반복 이력 분리 |
| `D1_cold_single_empty_cache` | 진단 전용 | load → empty_cache → measure×1 | CPU 기준선 초기 상태에 가장 가까운 조건 |
| `R1_controlled__stacks_all(_dladdr)` | 선택 | R1과 동일, `stacks="all"` | workspace 블록 C++ 출처 식별용 (medium_fp32_12만) |

### B-3. 코드 설명

| 파일 | 역할 |
|---|---|
| `handoff/GPU_COLLECTION_REQUEST.md` | 요청서: 목적, 환경, case/run, 필수 기록, 금지 사항, 반환 구조 |
| `handoff/common_manifest.json` | 기준 정의: 질문, 소프트웨어 버전, xMem commit·allocator lock sha, case별 입력 recipe·기대 sha, CPU 예측값, 이전 L4 참고값, run 단계 |
| `handoff/xmem-allocator-lock.json` | xMem `1ee4bd0` allocator/profiler/config 8개 파일 sha256 |
| `handoff/gpu_collector_reference.py` | INUOS 작성 참조 수집기 (GPU 미실행, 문법 검사만) |
| `collector/gpu_collector.py` | 실제 사용 수집기 = 참조본 + inugpu mod 1~7 |
| `run_one.sh` / `run_one.sh.v1` | (case, run) 1개 = docker 컨테이너 1개 실행 래퍼 |
| `run_all.sh` | 12 run 순차 실행 후 선택 run(stacks=all) 실행 |
| `tools/summarize.py` | `return/*/*/run_meta.json`·`boundaries.jsonl` → 관측값 표(MiB) 출력 |

#### `collector/gpu_collector.py` 흐름
1. `CUDA_MODULE_LOADING=LAZY` 설정 후 torch import, seed·TF32·threads 고정
2. 첫 CUDA 할당 전 `_record_memory_history(enabled="all", context="all", stacks=python|all, max_entries=50,000,000)`
3. `record_start` 경계 기록 → reserved 0·카운터 0 확인 (`record_start_check`)
4. `run_meta.json` 1차 저장 (버전, env, GPU, collector/manifest sha, bench_worker_v3 대비 차이)
5. manifest 단계 순회: 각 invocation 전후 `phase_begin`/`phase_end` 경계 (synchronize 후 카운터·memory_stats)
   - measure 블록 직전에만 `reset_peak_memory_stats()`
   - phase 마지막 invocation 뒤 `segments_<k>_<phase>.pickle.gz` (`memory_snapshot()` segment 상태)
   - `invocations.jsonl`: 실제 생성 토큰 수, 출력 sha256, 소요 ms, finite 여부
6. 종료: `record_end` → `_snapshot()` 원본을 `memory_snapshot.pickle`로 저장
7. 잘림 검사: trace alloc/segment_alloc 이벤트 수 vs 카운터 증가분, entries < max_entries → `trace_check`, `truncated`

#### 참조본 대비 수정 (inugpu mod, `return/collector/COLLECTOR_CHANGES.md`)
- 기록 의미(이벤트·경계·snapshot 형식) 변경 없음, 추가·보강만 (diff 82줄)
- mod 1 `--run-label-suffix` (선택 run을 별도 디렉터리로) / mod 2 run_meta 보강 / mod 3 attention_mask sha 검증
- mod 4 nonfinite logits면 실패 / mod 5 토큰 수 불일치 시 관측값 기록 후 실패 / mod 6 실패 invocation도 기록 / mod 7 잘림 검사
- sha256: 참조본 `513cbd80…66b6`, 실제 `ab6c96e2…88d2`

#### 런처 (`run_one.sh`)
- 사전 검사: 출력 디렉터리 존재 시 거부(90), 이미지 id 불일치(92), 다른 GPU 프로세스 존재(93)
- docker: 이미지 `cloudgpufit-bench:cu124-v1`(id `sha256:457da255…`, manifest의 이전 AWS image id와 동일), `--network none`, HF cache read-only, `HF_HUB_OFFLINE=1`, `python -I`, 특정 GPU 지정
- 기록: `.cmd`, `.start/.end`, `.exitcode`, stdout/stderr, 실행 전후 `nvidia-smi` CSV → `logs/` 및 run 디렉터리에 복사
- `run_one.sh.v1` 대비 `run_one.sh` 차이: `TORCH_SYMBOLIZE_MODE` 환경변수 전달 1줄 추가 (dladdr 재시도에만 사용)

### B-4. 실행 이력 (`logs/*.exitcode`, `return/_inugpu_logs/run_all.txt`)

| 순서 | case | run | rc | 시간(+09:00) |
|---|---|---|---|---|
| 사전 | fp16_01 | R2_cold_single (`_precheck`) | 0 | 본 실행 전 |
| 1~4 | fp16_01 | R0 / R1 / R2 / D1 | 0 | 19:57:52 ~ 19:58:47 |
| 5~8 | fp16_12 | R0 / R1 / R2 / D1 | 0 | 19:58:47 ~ 20:00:15 |
| 9~12 | medium_fp32_12 | R0 / R1 / R2 / D1 | 0 | 20:00:15 ~ 20:02:44 |
| 13 | medium_fp32_12 | R1_controlled__stacks_all | **1** | 20:02:44 ~ 20:02:52 |
| 14 | medium_fp32_12 | R1_controlled__stacks_all_dladdr | 0 | 20:03:38 ~ 20:04:34 |

- 13번 실패 원인 (기록): `stacks="all"` 심볼화가 `addr2line` 호출, 이미지에 없음 → `failed execvp` / `RuntimeError: stoi`
  - `memory_snapshot.pickle` 없음, invocations.jsonl 중복 9줄 (fork 자식 8개), 디렉터리는 보존
- 14번: `TORCH_SYMBOLIZE_MODE=dladdr`로 성공, 내부 심볼은 "??" (라이브러리 경로는 기록)

### B-5. 결과 요약 (관측값, `return/_inugpu_logs/summary_observed.txt`)

- 성공 run 전부: `start_empty=True`, `truncated=False`, alloc 이벤트 수 = 카운터 증가분, segment 이벤트 수 = 카운터
- 실제 생성 토큰 수 = 기대값(64/128), `num_alloc_retries` 0, `num_ooms` 0
- 입력 sha(input_ids·attention_mask) manifest와 일치 (`run_meta.json` `input_sha_match`)

#### case별 reserved/allocated (MiB, `record_end` 시점 reserved peak, measure 블록 allocated peak)

| case | 항목 | R0 | R1 | R2 | D1 |
|---|---|---|---|---|---|
| fp16_01 | load 후 reserved | 474.0 | 474.0 | 474.0 | 474.0 → empty_cache 후 326.0 |
| fp16_01 | reserved (record_end) | 492.0 | 492.0 | 492.0 | 344.0 |
| fp16_01 | measure allocated peak | 298.7 | 298.7 | 298.7 | 298.7 |
| fp16_01 | segment 수 (end) | 19 | 19 | 19 | 18 |
| fp16_12 | load 후 reserved | 474.0 | 474.0 | 474.0 | 474.0 → 326.0 |
| fp16_12 | finite_check 후 reserved | 1672.0 | 1672.0 | - | - |
| fp16_12 | reserved (record_end) | 1678.0 | 1678.0 | 920.0 | 934.0 |
| fp16_12 | measure allocated peak | 719.7 | 719.7 | 719.7 | 719.7 |
| fp16_12 | segment 수 (end) | 27 | 27 | 20 | 26 |
| medium_fp32_12 | load 후 reserved | 1380.0 | 1380.0 | 1380.0 | 1380.0 → 1380.0 |
| medium_fp32_12 | finite_check 후 reserved | 4570.0 | 4570.0 | - | - |
| medium_fp32_12 | reserved (record_end) | 4574.0 | 4574.0 | 3572.0 | 3572.0 |
| medium_fp32_12 | measure allocated peak | 2850.7 | 2850.7 | 2850.7 | 2850.7 |
| medium_fp32_12 | segment 수 (end) | 53 | 53 | 51 | 51 |

- finite_check(전체 시퀀스 1회 forward) 단계 allocated peak: fp16_12 1453.6, medium 3871.6 MiB (R0·R1)
- `R1_controlled__stacks_all_dladdr` 관측값은 `R1_controlled`와 동일 (이벤트 수 460,159 / segment 53 / reserved 4574.0)
- trace 규모: alloc 이벤트 30,086 (fp16_01 R2) ~ 1,149,440 (medium R0), snapshot entries 최대 3,447,699

#### manifest의 비교 기준값 (참고, MiB)

| case | CPU 예측 allocated / reserved | 이전 L4 inference peak allocated / reserved | 이번 RTX 3090 R0 measure allocated / reserved |
|---|---|---|---|
| fp16_01 | 296.5 / 330.0 | 298.7 / 492.0 | 298.7 / 492.0 |
| fp16_12 | 717.1 / 928.0 | 719.7 / 1678.0 | 719.7 / 1678.0 |
| medium_fp32_12 | 2842.6 / 3552.0 | 2850.7 / 4574.0 | 2850.7 / 4574.0 |

- 관측 사실만 정리
  - R0 수치가 이전 L4 참고값과 같은 숫자로 나옴 (manifest는 "정답값 아님, 참고치"로 명시)
  - R2(cold single) reserved는 R0/R1보다 작음: fp16_12 920.0 vs 1678.0, medium 3572.0 vs 4574.0, fp16_01은 동일 492.0
  - fp16_12 R2 reserved 920.0, D1 934.0 vs CPU 예측 928.0 / medium R2·D1 3572.0 vs CPU 예측 3552.0
  - 원인 판정(Q1~Q3)은 INUOS 분석 대상이며 이 디렉터리에 분석 결과 없음

### B-6. 결과 파일 위치·구조

```
cgf-reserved-diag-20261110/
  handoff/                      요청서·manifest·참조 수집기·allocator lock·SHA256SUMS (읽기 전용)
  collector/gpu_collector.py    실제 사용 수집기
  run_all.sh, run_one.sh(.v1)   런처
  tools/summarize.py            관측값 요약
  logs/                         run별 cmd/start/end/exitcode/stdout/stderr/nvidia_smi_{before,after}.csv (115개),
                                _precheck/, run_all.txt, summary_observed.txt
  return/                       반환 묶음 (INUOS 전달용)
    common_manifest.json, SHA256SUMS(271줄)
    collector/                  gpu_collector.py, 참조본(.received), run 스크립트, SHA256, COLLECTOR_CHANGES.md
    _inugpu_logs/               run_all.txt, summary_observed.txt, precheck_logs/
    _precheck/<fp16_01>/R2_cold_single/
    <case_id>/<run_id>/         run_meta.json, boundaries.jsonl, invocations.jsonl, memory_snapshot.pickle,
                                segments_<k>_<phase>.pickle.gz, collector.log, stdout/stderr, cmd, start/end, exitcode, nvidia_smi_*.csv
  cgf-reserved-diag-20261110-return.tar.gz (+ .sha256)   return/ 전체 압축본 (296 entries)
```

- 해시
  - `handoff/SHA256SUMS`: 받은 4개 파일 (요청서 `ab0ae995…`, 참조 수집기 `513cbd80…`, manifest `adddaa10…`, lock `352dff3f…`)
  - `return/SHA256SUMS`: 반환 묶음 전 파일
  - tarball sha256 `030d3475…132ec`
- `memory_snapshot.pickle` 크기 (run별)

| case / run | 크기(bytes) |
|---|---|
| fp16_01 R0 / R1 / R2 / D1 | 102,008,264 / 40,875,113 / 10,239,008 / 8,422,733 |
| fp16_12 R0 / R1 / R2 / D1 | 209,522,340 / 83,884,255 / 18,149,398 / 20,989,661 |
| medium R0 / R1 / R2 / D1 | 323,947,640 / 129,665,241 / 32,790,256 / 32,790,302 |
| medium R1 stacks_all_dladdr | 548,129,191 |
| medium R1 stacks_all (실패) | 없음 |
| _precheck fp16_01 R2 | 10,239,008 |

- 디렉터리 전체 1.6G. 그 외 파일은 모두 수십 KB 이하 (boundaries.jsonl 최대 29KB)

### B-7. 50MB 초과 파일 (git 제외 후보)

| 파일 | 크기 |
|---|---|
| `cgf-reserved-diag-20261110/return/core__decoder_llm__medium__fp32__12/R1_controlled__stacks_all_dladdr/memory_snapshot.pickle` | 548,129,191 B (~523 MiB) |
| `cgf-reserved-diag-20261110/return/core__decoder_llm__medium__fp32__12/R0_aws_replica/memory_snapshot.pickle` | 323,947,640 B (~309 MiB) |
| `cgf-reserved-diag-20261110/return/core__decoder_llm__small__fp16__12/R0_aws_replica/memory_snapshot.pickle` | 209,522,340 B (~200 MiB) |
| `cgf-reserved-diag-20261110/return/core__decoder_llm__medium__fp32__12/R1_controlled/memory_snapshot.pickle` | 129,665,241 B (~124 MiB) |
| `cgf-reserved-diag-20261110/return/core__decoder_llm__small__fp16__01/R0_aws_replica/memory_snapshot.pickle` | 102,008,264 B (~97 MiB) |
| `cgf-reserved-diag-20261110/cgf-reserved-diag-20261110-return.tar.gz` | 89,819,792 B (~86 MiB) |
| `cgf-reserved-diag-20261110/return/core__decoder_llm__small__fp16__12/R1_controlled/memory_snapshot.pickle` | 83,884,255 B (~80 MiB) |
| `xmem-work/xMem/.git/objects/pack/pack-91b354bb….pack` | 90,664,439 B (~86 MiB, upstream git 내부) |

### B-8. 한계 / 미확인

- INUOS 쪽 분석 결과(xMem 재생 비교, 최초 불일치 지점, workspace 식별)는 이 VM에 없음 → Q1~Q3 결론 미확인
- `stacks="all"` 첫 시도 실패, dladdr 모드는 내부 심볼 "??" → C++ 출처 식별 정밀도 제한
- NVML 샘플러·idle 대기·tokens/components 모드 생략, memory 모드 model_reuse 직전 peak reset 생략 (bench_worker_v3 대비, peak 통계에만 영향이라고 기록)
- GPU가 L4가 아닌 RTX 3090 → manifest상 이전 L4 값은 참고치, 이번 값이 L4 정답을 대체하지 않음
- 이전 AWS 실측의 `CUDA_MODULE_LOADING` 여부는 기록 없음 (요청서 명시)
- `memory_snapshot.pickle`은 이번 문서 작성 시 열어보지 않음 (크기·잘림 검사 결과는 run_meta 기준)
- `run_meta.json`에 hostname, GPU UUID, docker 명령 등이 포함됨 → 외부 공유 시 확인 필요

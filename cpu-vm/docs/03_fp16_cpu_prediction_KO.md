# INUOS FP16 VRAM 예측 실험 정리 (CPU 요청 추적 → xMem 재생 → RTX 3090 대조)

- 대상 저장소: `/home/inuos/cgf-fp16-680-20261111` (git 브랜치 `exp/fp16-680-cpu-20261111`, HEAD `cbf37b5`)
- 작성 기준 시각: 2026-11-15 18:4x KST, 읽기 전용 조사 (파일·git·로그·결과 JSON만 확인, 실험·컨테이너 실행 없음)
- 표기
  - 사실: 파일·로그·코드로 확인한 값
  - **추정**: 근거는 있으나 직접 확인하지 못한 해석
  - 시각은 모두 KST. 결과 JSON의 epoch 값을 변환
  - MiB = 2^20 B. 결과 파일 원값은 byte

---

## 1. 개요

### 1.1 목적
- GPU 없이 CPU VM(INUOS)에서 FP16 추론의 GPU VRAM(PyTorch CUDA caching allocator의 allocated/reserved)을 예측
- 방법
  1. CPU에서 추론을 실행하며 `torch.profiler`로 CPU 텐서 할당/해제 요청(`[memory]` 이벤트)을 순서대로 기록
  2. 요청 크기를 FP16 실행 기준으로 환산
  3. xMem(PyTorch CUDA caching allocator 시뮬레이터, 커밋 `1ee4bd08`, 무수정)에 순서대로 재생
  4. 재생 결과를 INUGPU(RTX 3090) 실측과 같은 키(`summary.first_measure_peak_reserved` 등, byte)로 대조
- 비교 기준: GPU **첫 번째 측정 회차(measure_0)**. GPU 5회 median/max는 CPU 쪽에 대응값 없음

### 1.2 기간
- 2026-11-11 15:05 (첫 smoke) ~ 2026-11-15 18:23 (마지막 커밋 `cbf37b5`)
- 커밋 9개

| 커밋 | 시각 | 내용 |
|---|---|---|
| `4b5465d` | 11-11 15:29 | 680 조건표, 공통 프로토콜 v1, 단일 세션 CPU tracer, 순서 재생, runner |
| `cbf5fcb` | 11-11 15:40 | README, collect/compare 도구, FLOP 추정, probe |
| `7332115` | 11-11 16:19 | FP16 shadow-dtype 에뮬레이션(FP32 실행), T5 FP16 분기 패치, 검증 도구, FP32 보정 |
| `0d39be7` | 11-12 11:07 | v2: GPU 프로토콜 worker, 로드 단계 합성(13/13 == RTX 3090), 단일 trace ×4 재생, 395 조건표, 병렬 스케줄러 |
| `42d96a9` | 11-12 11:09 | collect395, GPU trace 요청 비교, 본실험 실행 체인 |
| `8e1f7d3` | 11-15 17:31 | main395(L1024) 사용자 중단, 388개 최종 수집, GPU-CPU 정렬표 |
| `912f7fe` | 11-15 18:08 | L256 전환: 조건표, 3슬롯 병렬 스케줄러(저메모리 시 최근 시작 작업 선점), 용량 재생, 비교 도구 |
| `73d9c17` | 11-15 18:15 | 스레드 확인: 스레드 수별 요청 trace 다름, 예측 summary 12/12 동일 |
| `cbf37b5` | 11-15 18:23 | 용량 근처 15조건 용량 재생 결과 |

- 커밋되지 않은 항목(`git status`): `archive/`, `results/main395_l256/`

### 1.3 환경
| 항목 | 값 | 근거 |
|---|---|---|
| VM CPU | Intel Xeon (Cascadelake), vCPU 24, `avx512_fp16`/`avx512_bf16`/AMX 플래그 없음 | `lscpu`, `/proc/cpuinfo` |
| RAM | MemTotal 약 47 GiB (49,326,380 kB) | `/proc/meminfo` |
| Docker 이미지 | `cloudgpufit-cpu-validation:20261108`, ID `sha256:875290b5854ad6dbc8ea554fecb292dd5ffdfea2a03e8c4e8179a3c827968fa5` | result.json `environment.image_id` |
| SW | python 3.11.11, torch 2.6.0+cu124, transformers 4.52.4, numpy 2.2.2 | result.json |
| torch CPU | `cpu_capability=AVX512`, `mkldnn_fp16_supported=False` | v1 result.json |
| xMem | `/home/inuos/xmem-work/xMem` @ `1ee4bd08dda277a6248abee5b6d40e0d6067efaa`, 파일 sha256 잠금 `provenance/xmem-allocator-lock.json`, worker가 매번 `verify_xmem` | result.json `xmem` |
| 재사용 코드 | `/home/inuos/cgf-reserved-diag-20261110/src/cgf_diag` (`replay.Replayer`, `replay.verify_xmem`, `events.load_snapshot`), trace 파서 `/home/inuos/CloudGPUFit-cpu-singlepass-20261108/inference_trace_runtime.iter_trace_events` | import 문, `cpu_manifest.json` |
| 컨테이너 공통 | 조건마다 새 컨테이너, `--network none`, `--memory = --memory-swap`(swap 없음), HF 오프라인, `/exp`=저장소 bind | `tools/run_suite.py`, `run395.py`, `run_l256.py` |

- CPU FP16 성능 (README·memory 기록, probe `tests/probes/bmm_layouts.py`, `bf16_ops.py`)
  - FP16 bmm 약 1.8 GFLOPS, 스레드 확장 없음
  - conv 약 10 GFLOPS, 일부 matmul 0.3 GFLOPS (GPT-2 Conv1D row-major)
  - 결과: 네이티브 FP16 CPU 실행은 큰 조건에서 비현실적으로 느림 → v2에서 FP32 실행 + FP16 크기 환산으로 전환

### 1.4 모델 (13개, revision은 `manifest/models.json` sha256 `7b11e3e3…6589`, GPU 쪽과 동일)
| 모델 | short (v2) | 유형 | revision | safetensors |
|---|---|---|---|---|
| facebook/opt-125m | opt-125m | decoder_llm | 27dcfa74d334bc871f3234de431e71c6eeba5dd6 | false (.bin) |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | tinyllama-1.1b-chat | decoder_llm | fe8a4ea1ffedaf415f4da2f062534de366a451e6 | true |
| Qwen/Qwen2.5-1.5B-Instruct | qwen2.5-1.5b-instruct | decoder_llm | 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 | true |
| openai-community/gpt2 | gpt2 | decoder_llm | 607a30d783dfa663caf39e06633721c8d4cfcd7e | true |
| openai-community/gpt2-medium | gpt2-medium | decoder_llm | 6dcaa7a952f72f9298047fd5137cd6e4f05f41da | true |
| openai-community/gpt2-xl | gpt2-xl | decoder_llm | 15ea56dee5df4983c59b2538573817e1667135e2 | true |
| microsoft/resnet-18 | resnet-18 | vision_cls | 65a5785d9156231087c481e0c7dd33a5ff6f7e3e | true |
| microsoft/resnet-50 | resnet-50 | vision_cls | 34c2154c194f829b11125337b98c8f5f9965ff19 | true |
| google/mobilenet_v2_1.0_224 | mobilenet-v2-1.0-224 | vision_cls | 75e607b00aeae1297cc89d026a118bce012f5c5a | true |
| google/vit-large-patch16-224 | vit-large-patch16-224 | vision_cls | 9e2727f4250d3973839eecfa5c4b42e41b709a50 | false (.bin) |
| google-t5/t5-small | t5-small | text_encdec | df1b051c49625cf57a3d0d8d3863ed4d13564fe4 | true |
| google-t5/t5-base | t5-base | text_encdec | a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1 | true |
| google-t5/t5-3b | t5-3b | text_encdec | bed96aab9ee46012a5046386105ee5fd0ac572f0 | true |

- v1은 `manifest/model_pins.json`(같은 revision, short 이름은 `tinyllama-1.1b`, `qwen2.5-1.5b`, `mobilenet-v2`, `vit-large`)
- 가중치는 `hf_cache/hub/models--*/snapshots/<revision>` (git 제외)
- 참고: `models.json`의 `common.attn_implementation`/`load_kwargs`에는 `"eager"`가 적혀 있으나 GPU worker(d06f0f15…)와 CPU v2는 실제로 인자를 생략(기본값)

### 1.5 worker 버전
| 이름 | 파일 | 도입 | 사용 실행 |
|---|---|---|---|
| CPU v1 | `src/cgf680/cpu_worker.py` + `protocol.py` (sha256 앞 16자 `1d1e3196…`, `2b7b5011…`) | `4b5465d`, 에뮬레이션 옵션 `7332115` | smoke*, pilot(v1), verify-userscope, pilot-fullscope-ref, fp32-calib, smoke-emu, emu-validate, emu-overhead |
| CPU v2 | `src/cgf680/cpu_worker2.py` + `protocol2.py` + `load_emu.py` + `shadow.py` (`9d7f672c…`, `4825296e…`, `3377d758…`, `33709871…`) | `0d39be7` 이후 무수정 (11-12 매니페스트 해시와 동일, 작업 트리 clean) | smoke2, main395(pilot 포함), thread_check, main395_l256(재사용분) |
| GPU worker (기준) | INUGPU `vram_worker.py` sha256 `d06f0f15982c811c4bc6a24028f71728e461c6ee3651bcfd520ba49b0d675883`, runner `run_conditions.py` `77e3f4a9…` | handoff 11-12, 11-15 | RTX 3090 L1024(11-12), L256(11-15 17:00 시작) |

- GPU 환경(11-15 handoff 문서): `cloudgpufit-bench:cu124-v1`, torch 2.6.0+cu124, cuDNN 90100, transformers 4.52.4, torch 스레드 4, `cuda_total_memory` 25,306,529,792 B

---

## 2. 실험 목록 (시간순)

### 2.0 한눈에 보기
| # | 라벨 / 실행 폴더 | 시각 | worker | 조건 | 상태 | 핵심 결과 |
|---|---|---|---|---|---|---|
| 1 | 680 조건표 + 프로토콜 v1 | 11-11 15:01~15:29 | - | 680 (파일럿 26) | 생성 | sha256 `3f94c510…` |
| 2 | `runs/smoke/opt` (수동), `smoke-gpt2-rowmajor`, `smoke-gpt2-colmajor` | 11-11 15:05~15:12 | v1 | smoke 3 | ok 5 | Conv1D col-major: 예측 동일, 6~12배 단축 |
| 3 | `pilot-fullscope-ref` / `verify-userscope` | 15:13~15:29 | v1 | 파일럿 2 | ok 2 / ok 2 | USER_SCOPE: 요청 수·예측 동일, RSS 16.4→1.6 GiB |
| 4 | `pilot` (v1, 네이티브 FP16) | 15:30~15:49 | v1 | 파일럿 26 | ok 11, 12번째(gpt2-medium) 실행 중 중단 | 아래 2.4 |
| 5 | `fp32-calib` | 15:50~20:24 | v1 | FP32 26 | ok 26 | 680 순차 추정 474.4 h |
| 6 | `bf16-calib`, `verify-layout-rowmajor` | - | v1 | - | **미실행** (스크립트만 존재) | - |
| 7 | `smoke-emu`, `emu-validate` | 16:09~16:18 | v1 + 에뮬 | smoke 3, 파일럿 11 | ok 3, ok 11 | LLM·T5 예측 일치, CNN 불일치 |
| 8 | `emu-overhead` | 11-12 10:46~10:57 | v1 + 에뮬 | 2 | ok 2 | 에뮬 오버헤드 측정 |
| 9 | 로드 단계 합성 검증 | 11-12 11:02 | load_emu | 13 모델 | 13/13 일치 | `results/load_emu_validation.jsonl` |
| 10 | `smoke2` | 11-12 11:04~11:05 | v2 (직접 실행) | 3 | ok 3 | v2 동작 확인 |
| 11 | 단일 trace ×4 재생 검증 | 11-12 (시각 기록 없음) | 도구 | 36~37 | 전부 동일 | 저장된 출력 없음, 아래 2.11 |
| 12 | pilot395 (`runs/main395`의 파일럿분) | 11-12 11:07~12:48 | v2 | 26 | ok 26 | RTX 3090 대비 reserved 평균 |오차| 5.09% |
| 13 | GPU trace 요청 비교 | 11-12 11:09 | 도구 | 4 | - | 크기 multiset 4/4 불일치 |
| 14 | main395 (L1024) | 11-12 12:48 ~ 11-15 17:09, 사용자 중단 | v2 | 395 | 388 결과: ok 375 / OOM 6 / timeout 7 | 아래 2.14 |
| 15 | GPU-CPU 정렬 (L1024) | 11-15 17:11~18:23 | 도구 | 374 비교 | - | LLM·T5 allocated ±1.5%, CNN 불일치, 용량 문제 15건 |
| 16 | L256 전환 | 11-15 18:02~18:08 | v2 무수정 | 395 (재사용 244, 실행 대상 151) | **151개 미실행** | 아래 2.16 |
| 17 | 스레드 확인 `runs/thread_check` | 11-15 18:04~18:15 | v2 | 4조건 × 4스레드 | ok 16 | trace 다름, summary 12/12 동일 |
| 18 | 용량 재생 15조건 / 1 TiB self-check 5조건 | 11-15 ~18:22 / 18:06 | 도구 | 15 / 5 | - | ±5% 9건, 5~10% 3건, 크게 어긋남 3건 / 5/5 일치 |

### 2.1 680 조건표와 프로토콜 v1 (커밋 `4b5465d`)
- 설계
  - `src/cgf680/cases.py`: 13모델 × 배치 {1,8,16,32,128}
    - LLM(opt/tinyllama/qwen)·T5: L×N ∈ {32,128,512,1024}² → 모델당 80
    - GPT-2 3종: {32,128,512}² → 모델당 45
    - CNN 3종: R ∈ {128,224,256,512} → 20, ViT: 224만 → 5
    - 합계 680, 파일럿 26 (텍스트 (8,128,128),(16,512,512); CNN (8,224),(16,512); ViT (8,224),(16,224))
    - case_id 예: `fp16__gpt2__b8__in128__new128`, T5는 `enc`
  - `manifest/cases.jsonl` sha256 `3f94c510e189129d879dceeb11782a7b8b54052417e78decbacff00baf8ae608`
  - `src/cgf680/protocol.py`(v1): eager attention 고정, hub `generation_config.json` 기본값 + 호출 인자 덮어쓰기, 입력은 호스트에서 만든 텐서를 1회 배치 후 모든 회차에 공유, special id(pad/eos/bos) → 3
- `src/cgf680/cpu_worker.py`(v1): profiler 세션 1개 안에서 warmup 3 + measure 1을 실제 실행·추적 → 정규화 → xMem 순서 재생 (상세 §3)
  - 상주 storage(파라미터·버퍼·입력)는 빈 allocator에 malloc 후 고정. GPU 로드 이력 미반영
  - 경계 표식 M0~M5: 구간 밖 고유 크기(1,234,567 + 2k B) uint8 할당
- 당시 GPU 쪽 680 runner 매니페스트가 VM에 없어 CPU 자체 프로토콜로 시작 (memory 기록)

### 2.2 smoke와 Conv1D column-major (11-11 15:05~15:12)
- `runs/smoke/opt/`: 수동 실행 1건(`trace.json` 보존, runner 기록 없음)
- `smoke-gpt2-rowmajor` vs `smoke-gpt2-colmajor` (8 스레드, 20g)
  - GPT-2 Conv1D 가중치 stride만 column-major로(shape·bytes 동일) → FP16 addmm 빠른 경로

| 조건 | 예측 R/A 피크 MiB (row / col) | 요청 수 | 컨테이너 wall s (row → col) | trace 세션 s |
|---|---|---|---|---|
| smoke gpt2 b2 in32 new8 | 274.0/267.0 / 동일 | 17,048 / 동일 | 79.8 → 12.6 | 71.2 → 3.4 |
| smoke gpt2 b4 in64 new16 | 332.0/298.8 / 동일 | 33,976 / 동일 | 302.8 → 25.4 | 288.5 → 11.8 |

- `tools/compare_events.py`로 완전 일치 확인했다고 README에 기록. 비교 출력 파일은 없음
- 파일럿 크기 검증 `docs/run_verify_layout.sh` → `runs/verify-layout-rowmajor`는 **존재하지 않음** (미실행)
- v2는 FP32 실행이라 이 최적화를 쓰지 않음

### 2.3 profiler USER_SCOPE 기법 (11-11 15:13~15:29)
- 문제: 기본 profiler(모든 aten op 기록)는 RSS·디스크 폭증 (이전 실험의 RAM 문제, memory 기록)
- 해결: 저수준 `torch.autograd._enable_profiler(cfg, {CPU}, {RecordScope.USER_SCOPE})`
  - aten op 이벤트는 기록하지 않고, allocator 메모리 보고 hook에서 나오는 `[memory]` 이벤트와 `record_function` 구간만 기록
- 비교: `pilot-fullscope-ref`(전체 범위) vs `verify-userscope`(USER_SCOPE), 20 스레드, 42g

| 조건 | 메모리 이벤트 | 요청 수 | 예측 R/A 피크 MiB | worker max RSS GiB (all → user) | 컨테이너 피크 GiB | wall s |
|---|---|---|---|---|---|---|
| fp16 opt-125m b8 128/128 | 421,212 (동일) | 210,600 (동일) | 416.0/387.0 (동일) | 16.39 → 1.62 | 18.25 → 1.29 | 224 → 77 |
| fp16 gpt2 b8 128/128 | 541,948 (동일) | 270,968 (동일) | 468.0/415.7 (동일) | 16.59 → 1.93 | 18.79 → 1.81 | 409 → 84 |

- `pilot-fullscope-ref`는 원래 파일럿 텍스트 10개 실행(`docs/run_pilot_text_a.sh`)이었고 3번째(gpt2-medium)에서 멈춤. 폴더 이름을 참조용으로 바꾼 것으로 **추정** (suite_args의 run_name은 `pilot`)
- probe: `tests/probes/rss_profiler_probe.py`, `clock_probe.py`

### 2.4 v1 파일럿 (네이티브 FP16, `runs/pilot`)
- 명령: `docs/run_pilot.sh` (`run_suite.py --select pilot --threads 20 --mem 42g --conv1d-colmajor --timeout-hours 24`)
- 시각: 11-11 15:30:24 시작, 15:47:43까지 11개 ok, 12번째 `fp16__gpt2-medium__b8__in128__new128`가 measure0 추적 중 15:49:49에 기록 종료 (result.json 없음)
  - 바로 뒤 15:50에 `fp32-calib` 시작 → 의도적 중단으로 **추정** (FP16 속도 문제)
- 결과 (cost 순, 4회 실제 실행)

| case | wall s | 요청 수(4회) | 예측 reserved/allocated 피크 MiB |
|---|---|---|---|
| resnet-18 b8 r224 | 15.5 | 500 | 78.0 / 65.9 |
| resnet-18 b16 r512 | 176.8 | 500 | 758.0 / 469.2 |
| resnet-50 b8 r224 | 29.7 | 1,156 | 132.0 / 101.4 |
| resnet-50 b16 r512 | 306.1 | 1,156 | 1074.0 / 585.6 |
| mobilenet-v2 b8 r224 | 15.2 | 86,880 | 72.0 / 58.5 |
| mobilenet-v2 b16 r512 | 28.5 | 86,880 | 974.0 / 544.8 |
| opt-125m b8 128/128 | 77.5 | 210,600 | 416.0 / 387.0 |
| gpt2 b8 128/128 | 84.8 | 270,968 | 468.0 / 415.7 |
| t5-small b8 128/128 | 63.5 | 371,620 | 164.0 / 146.7 |
| vit-large b8 r224 | 109.0 | 2,624 | 682.0 / 649.1 |
| t5-base b8 128/128 | 130.7 | 689,716 | 560.0 / 527.1 |

- 참고: RTX 3090 eager 파일럿(`cgf-fp16-inuos-handoff-20261112/reference/replay_table.md`)의 m1 피크 R/A MiB: opt b8 418.0/388.1, gpt2 b8 476.0/418.4, t5-small b8 304.0/154.2, resnet-50 b8 132.0/107.4
  - v1은 로드 이력이 없어 reserved 시작 상태가 GPU와 다름. v1 결과는 GPU와 공식 대조하지 않음

### 2.5 FP32 보정 (`runs/fp32-calib`, 11-11 15:50~20:24)
- 목적: FP32 실행 시간으로 680 전체 소요 추정 (FP16 대신 FP32를 쓰는 방안 검토)
- 명령: `docs/run_fp32_calib.sh` (`run_suite.py --cases-file manifest/fp32_calib_cases.jsonl --threads 20 --mem 42g --timeout-hours 3`, Conv1D 변경 없음) → `collect.py`
- 결과: 26/26 ok (`results/fp32-calib/summary.json`)
  - 가장 긴 조건: t5-3b b16 512/512 4,478 s, gpt2-xl b16 512/512 3,851 s
  - 예: t5-3b b16 512/512 FP32 reserved 피크 24,076 MiB, allocated 23,336 MiB
- `tools/fit_eta.py` → `results/fp32-calib/eta.json` (11-12 09:49 갱신)
  - 680 순차 합계 474.4 h. 모델별 t5-3b 222.4 h, tinyllama 88.4 h, qwen 84.1 h, gpt2-xl 33.1 h
  - 제한 시간 시나리오: 1 h → 587 완료/93 초과(184.5 h), 2 h → 624/56(252.9 h), 4 h → 652/28, 6 h → 664/16
- 주의: 16:09~16:18 `smoke-emu`·`emu-validate`(8 스레드)가 동시에 돌아 일부 시간값에 경합 영향 가능 (**추정**)
- `docs/run_bf16_calib.sh`(BF16 보정)는 `runs/bf16-calib`이 없어 **미실행**

### 2.6 FP16 shadow-dtype 에뮬레이션 (`shadow.py`, 커밋 `7332115`)
- 설계: FP32로 실행하면서 FP16 실행이었다면 생길 요청 크기를 계산
  - `TorchFunctionMode`로 모든 torch 호출 결과 storage에 shadow 태그
    - `h`: 모델 dtype을 따름 → FP16에서 절반 (`(s+1)//2`)
    - `f`: 고정 FP32 (명시적 `float32`/`.float()`) → 크기 그대로
    - `x`: 비부동소수 → 그대로
    - `b`: 에뮬레이션 버퍼 (이미 FP16 기준 크기)
  - FP32에서는 no-op이지만 FP16에서는 실제 복사인 cast(업캐스트 `h→f`, 다운캐스트 `f→h`)를 uint8 버퍼로 재현, 다운캐스트는 원본을 조기 해제
  - 명시적 float32 여부는 호출 소스 텍스트(co_positions)로 판단
  - profiler `[memory]` alloc과 주소·크기·호출 순서로 매칭, 매칭 안 된 요청은 다음 매칭 요청의 태그 상속(`inferred_*`)
  - T5 전용: `T5LayerNorm.forward`의 FP16 분기 강제, `T5Block.forward`의 `dtype == float16` inf clamp 분기를 shadow 태그로 판정 (소스 문자열 치환 후 재컴파일)
- v1 worker 옵션 `--emulate-fp16-from-fp32`: 상주 크기도 부동소수 storage만 절반, FP32 원값(`prediction_fp32_raw`)도 기록, `events_fp16_emulated.npz` 저장

### 2.7 에뮬레이션 검증 (`smoke-emu`, `emu-validate`, 11-11 16:09~16:18)
- `smoke-emu`(4 스레드): opt/gpt2/t5-small smoke 3건 ok, 매칭 100% (`unmatched_records` 0, `undetermined_casts` 0)
  - t5-small smoke 예측 132.0/118.7 MiB = 네이티브 smoke(`runs/smoke`, 16:10) 132.0/118.7
- `emu-validate`(8 스레드, 30g): 파일럿 11건을 에뮬로 실행 → v1 네이티브 FP16(`runs/pilot`)과 비교

| case | 네이티브 FP16 R/A 피크 MiB | 에뮬 R/A 피크 MiB | 요청 수 (네이티브/에뮬, 4회) | 에뮬 cast (up/down) |
|---|---|---|---|---|
| opt-125m b8 | 416.0/387.0 | 416.0/387.0 | 210,600 / 204,456 | 0 / 6,144 |
| gpt2 b8 | 468.0/415.7 | 468.0/415.7 | 270,968 / 270,968 | 0 / 0 |
| t5-small b8 | 164.0/146.7 | 164.0/146.7 | 371,620 / 342,800 | 15,948 / 15,948 |
| t5-base b8 | 560.0/527.1 | 562.0/527.1 | 689,716 / 633,104 | 31,380 / 31,380 |
| vit-large b8 | 682.0/649.1 | 662.0/639.7 | 2,624 / 2,532 | 0 / 96 |
| resnet-18 b8 r224 | 78.0/65.9 | 62.0/50.0 | 500 / 660 | 0 / 0 |
| resnet-18 b16 r512 | 758.0/469.2 | 304.0/303.2 | 500 / 660 | 0 / 0 |
| resnet-50 b8 r224 | 132.0/101.4 | 116.0/95.1 | 1,156 / 1,712 | 0 / 0 |
| resnet-50 b16 r512 | 1074.0/585.6 | 588.0/521.6 | 1,156 / 1,712 | 0 / 0 |
| mobilenet-v2 b8 r224 | 72.0/58.5 | 110.0/72.6 | 86,880 / 1,856 | 0 / 0 |
| mobilenet-v2 b16 r512 | 974.0/544.8 | 1108.0/690.8 | 86,880 / 1,856 | 0 / 0 |

- 정리 (감사 매니페스트 `handoff_codex/cpu_manifest.json`의 `emulated_vs_native_summary`)
  - 이벤트 action+size 순서 완전 일치 0/11
  - 저장된 예측 8개 값 모두 일치 3/11 (opt, gpt2, t5-small)
  - CNN은 네이티브 FP16 CPU 경로 자체가 FP32와 다른 커널·버퍼를 씀 (mobilenet 요청 수 47배 차이). 네이티브 FP16 CPU도 GPU 기준값이 아니므로 어느 쪽이 GPU에 가까운지는 이 비교로 판단 불가
- 도구: `tools/compare_emulated.py` (출력 저장 파일 없음), `tools/replay_variant.py`(FP32 trace 일괄 0.5배 재생 비교용)

### 2.8 에뮬레이션 오버헤드 (`runs/emu-overhead`, 11-12 10:46~10:57)
| case | 에뮬 wall s (20 스레드) | fp32-calib wall s | 에뮬 R/A 피크 MiB | FP32 원값 R/A MiB |
|---|---|---|---|---|
| opt-125m b16 512/512 | 371.5 | 360.3 | 1832.0/1332.5 | 3312.0/2651.2 |
| tinyllama-1.1b b8 128/128 | 265.5 | 206.5 | 2304.0/2180.7 | 4426.0/4363.6 |

### 2.9 v2 GPU 프로토콜 worker와 로드 단계 합성 (커밋 `0d39be7`)
- 계기: 11-12 10:33 INUGPU handoff(`cgf-fp16-inuos-handoff-20261112`) 도착. GPU worker 코드·조건표 395·파일럿 reference·trace 6개 제공
- v1 → v2 변경 (`protocol2.py`, `cpu_worker2.py`)
  - attention: 인자 생략(기본값). 실제 OPT·TinyLlama·Qwen·GPT-2·ViT=sdpa, T5·CNN=eager (CPU result `model_info` 확인)
  - 생성: `GenerationConfig.from_model_config` + greedy, min=max=N (hub generation_config 미사용)
  - special id에 `decoder_start` 추가
  - 매 회차 입력 복사(`clone`, GPU의 `.to(cuda)` 대응), 출력과 함께 기록(M4) 후 `del`(M5)
  - 실행 dtype: FP32 + shadow 태그로 FP16 요청 크기 (`dtype_mode=fp32_execution_fp16_request_sizes`)
  - 추적은 호출 1회만, 재생은 같은 trace를 warmup 3 + measure 1로 4회 반복 (캐시 유지)
  - 로드 단계: FP16 모델 구조와 체크포인트 헤더로 GPU `from_pretrained(device_map=cuda)` 요청 순서를 합성(`load_emu.py`)해 먼저 재생
  - cuBLAS workspace 8.125 MiB(8,519,680 B) 1회 주입한 변형값도 별도 기록(예측값 아님)
  - 실행 후 B=1·2토큰 attention 커널 probe(전체 op 범위)로 CPU aten 경로 기록
- 로드 합성 규칙 (`load_emu.py` docstring, GPU trace에서 도출)
  1. `caching_allocator_warmup`: state_dict 전체 크기 W를 alloc 후 free
  2. 체크포인트 순서(safetensors는 키 정렬, .bin은 파일 순서)대로 텐서 alloc. dtype이 다르면 ckpt 크기 alloc → load 크기 alloc → ckpt free. tie된 키는 alloc 후 즉시 free
  3. 체크포인트에 없는 non-persistent 버퍼 alloc
  4. `.to(fp16)`: FP16이 아닌 부동소수 텐서(`_keep_in_fp32_modules`, FP32 rotary 버퍼) cast
  - GPT-2처럼 base prefix 없는 체크포인트는 base model에 로드
- 검증 (`tests/probes/validate_load_emu.py` → `results/load_emu_validation.jsonl`, 11-12 11:02)
  - 13/13 모델: after_load reserved, 로드 중 피크 allocated가 RTX 3090 파일럿 값과 소수 1자리 MiB까지 동일
  - GPU trace가 있는 5모델(opt, qwen, gpt2, resnet-50, t5-small)은 요청 순서 완전 일치(`trace_exact_sequence: true`)

| 모델 | 합성 이벤트 | after_load R MiB (합성=GPU) | 로드 피크 A MiB (합성=GPU) |
|---|---|---|---|
| opt-125m | 200 | 316.0 | 312.5 |
| tinyllama | 608 | 2234.0 | 2106.2 |
| qwen2.5-1.5b | 1,019 | 3448.0 | 3390.0 |
| gpt2 / medium / xl | 470 / 926 / 1,838 | 474.0 / 1002.0 / 3486.0 | 386.5 / 873.1 / 3278.8 |
| resnet-18 / 50 | 328 / 856 | 48.0 / 82.0 | 31.7 / 57.1 |
| mobilenet-v2 | 840 | 26.0 | 7.4 |
| vit-large | 1,178 | 620.0 | 595.5 |
| t5-small / base / 3b | 395 / 773 / 1,529 | 272.0 / 678.0 / 7166.0 | 234.0 / 674.3 / 7164.0 |

- probe 이력: `tests/probes/load_seq.py`, `load_seq2.py`(CPU 프로파일 로드 vs GPU, difflib), `load_synth.py`(헤더 기반 합성 초안)

### 2.10 smoke2 (v2, 11-12 11:04~11:05)
- `runs/smoke2/<cid>/` 3건, runner 없이 직접 실행(8 스레드, `image`/`cpuset` 환경값 없음) → 모두 ok
- 예: opt b8 128/128 → after_load R 316.0, m1 피크 R/A 416.0/380.0 MiB, 태그 매칭 28,455/28,455

### 2.11 단일 trace ×4 재생 검증
- 근거: v1 trace는 4회를 실제 실행했으므로 회차별 요청을 비교 가능
  - `tools/check_run_identity.py`: 4회 요청 순서(action·size) 동일 여부
  - `tools/check_replicated_replay.py`: 회차 0만 4번 이어 붙여 재생한 값 = 실제 4회 재생값인지
- 기록
  - `cpu_worker2.py` docstring: "36/36 cases" 동일
  - 감사 매니페스트(11-12 11:28, `cpu_manifest.json` `trace_replication_evidence`): 저장된 v1 trace 37건(fp32-calib 26 + pilot 11) 모두 4회 동일
  - 두 도구의 실행 출력 파일은 저장소에 없음
- 범위 제약: v1(eager) trace 기준. v2 기본 attention 조건 전체에 대한 증명은 아님

### 2.12 pilot395 (`runs/main395`의 파일럿 26)
- 명령: `tmux cgf395-pilot` → `docs/run_pilot395.sh` → `run395.py --run-name main395 --select pilot --phases A B`
  - 같은 실행 폴더라 본실험에서 재실행 안 함
- 시각: 11-12 11:07:02 ~ 12:48:05, 26/26 ok (A 13, B 13)
- 결과: `results/main395/pilot_vs_rtx3090.csv` (§4.1). 마지막 수집에서도 값 동일(`.after_pilot.csv`와 동일 파일 내용)

### 2.13 GPU trace 요청 비교 (`results/main395/gpu_trace_request_diff.jsonl`)
- 도구: `tools/compare_gpu_trace.py` (GPU snapshot의 measure_0 구간 alloc 크기 vs CPU 단일 호출 FP16 요청, 크기 multiset 비교)
- 대상: 기본 attention GPU trace가 있는 파일럿 4조건

| 조건 | GPU 요청 수 / bytes | CPU 요청 수 / bytes | GPU에만 (상위) | CPU에만 (상위) |
|---|---|---|---|---|
| opt-125m b8 128/128 | 46,635 / 19.04 GB | 42,411 / 9.36 GB | 1 MiB × 9,216 (9.66 GB), 50272 B × 512 | 62080 B × 12, 24576 B × 12 (합 5.5 MB) |
| gpt2 b8 128/128 | 51,103 / 16.85 GB | 59,039 / 10.39 GB | 1 MiB × 6,144 (6.44 GB), 50257 B × 512 | 합 5.5 MB |
| t5-small b8 128/128 | 68,215 / 4.05 GB | 85,702 / 4.07 GB | 8192 B × 2,304, 32128 B × 512, 1 MiB × 12 | 16384 B × 2,304, 2 MiB × 12, 8 B × 9,266 |
| resnet-50 b8 r224 | 292 / 0.65 GB | 429 / 0.92 GB | 6.2 MiB × 3 등 (합 119 MB) | 12.25 MiB × 8, 3.06 MiB × 30 등 (합 389 MB) |

- 해석
  - LLM의 GPU 전용 1 MiB 요청 다수는 cuBLASLt 임시 workspace로 **추정**. 짧은 수명이라 피크 영향은 작음 (allocated 피크 오차 −0.3~−2%)
  - T5: GPU 8 KB ↔ CPU 16 KB, GPU 1 MiB ↔ CPU 2 MiB로 정확히 2배 → 일부 텐서가 `f`(FP32 고정)로 태그되어 절반 처리되지 않은 것으로 **추정**. GPU 쪽 질문 §8-4에서도 미확인으로 남음
  - CNN: CPU(mkldnn)와 GPU(cuDNN)의 conv 경로 차이 (원인 미확정)

### 2.14 main395 본실험 (구 L1024 조건표, 중단)
- 조건표: `manifest/conditions_main395.json` sha256 `2634bda7…fe258` → `cases395.jsonl` (`cases395.py`)
  - 680표에서 텍스트 길이 후보 32 제거: LLM·T5 {128,512,1024}² ×5배치 = 모델당 45, GPT-2 {128,512}² → 20, CNN 20, ViT 5. 파일럿 26 포함
- 명령: `tmux cgf395-main` → `docs/run_main395.sh` (파일럿 종료 대기 → collect → `run395.py --select all --phases A B C` → collect)
- 스케줄 (`tools/run395.py`)
  - A: "작은" 조건(작은 모델 + 크기 기준) 4슬롯 병렬, cpuset 0-4/5-9/10-14/15-19, 5 스레드, 10g, 2 h 제한
  - B: 나머지 순차, 0-19, 20 스레드, 42g, 2 h 제한
  - C: A의 OOM + A/B timeout을 순차, 42g, 제한 없음 → **실행 전에 중단되어 미실행**
- 시각
  - 본실험 호출 11-12 12:48:39 (todo 369 = small 120, large 249)
  - A 단계 11-12 12:48~13:43 (파일럿 포함 A 133건, wall 합 3.7 h)
  - B 단계 ~ 11-15 17:09:18 (파일럿 포함 B 255건, wall 합 77.0 h)
  - 389번째 `qwen2.5-1.5b-instruct__b128__in1024__new1024`가 17:09:25에 trace 시작 후 사용자 중단 (result.json 없음, `progress.json` stage=trace_invocation)
  - 최종 수집 17:30 (`results/main395/collect_final_20261115.log`), 보관 17:31
- 상태 (`results/main395/status_summary.json`)
  - 결과 388: ok 375 (A 133, B 242), `worker_killed_cpu_oom` 6, `timeout` 7
  - 결과 없음 7: 위 qwen 1건 + t5-3b b128 6건(128/1024, 512/512, 512/1024, 1024/128, 1024/512, 1024/1024, 미시작)

| 모델 | ok | 실패 | 결과 없음 |
|---|---|---|---|
| opt-125m, gpt2, gpt2-medium, resnet-18, resnet-50, mobilenet-v2, vit-large, t5-small, t5-base | 전부 ok (45/20/20/20/20/20/5/45/45) | - | - |
| tinyllama | 43 | timeout 2 | - |
| qwen2.5-1.5b | 43 | timeout 1 | 1 |
| gpt2-xl | 17 | OOM 3 | - |
| t5-3b | 32 | timeout 4, OOM 3 | 6 |

- 실패 상세 (모두 B 단계, attempt 1)

| condition_id | 상태 | 단계 | wall s | 컨테이너 메모리 피크 |
|---|---|---|---|---|
| gpt2-xl b128 128/512 | worker_killed_cpu_oom | trace_invocation | 2,226 | 42 GiB(한도) |
| gpt2-xl b128 512/128 | worker_killed_cpu_oom | trace_invocation | 113 | 42 GiB |
| gpt2-xl b128 512/512 | worker_killed_cpu_oom | trace_invocation | 118 | 42 GiB |
| t5-3b b128 128/512 | worker_killed_cpu_oom | trace_invocation | 1,614 | 42 GiB |
| t5-3b b128 512/128 | worker_killed_cpu_oom | trace_invocation | 207 | 42 GiB |
| t5-3b b32 1024/1024 | worker_killed_cpu_oom | trace_invocation | 4,237 | 42 GiB |
| t5-3b b16 128/1024 | timeout | xmem_replay | 7,202 | 15.5 GiB |
| t5-3b b16 512/1024 | timeout | xmem_replay | 7,202 | 21.5 GiB |
| t5-3b b32 128/1024 | timeout | xmem_replay | 7,202 | 30.6 GiB |
| t5-3b b32 512/1024 | timeout | xmem_replay | 7,202 | 42 GiB |
| tinyllama b128 512/1024 | timeout | trace_invocation | 7,202 | 18.3 GiB |
| tinyllama b128 1024/1024 | timeout | trace_invocation | 7,202 | 24.8 GiB |
| qwen b128 512/1024 | timeout | trace_invocation | 7,203 | 22.1 GiB |

- ok 조건 wall: 중앙값 234 s, 최대 6,288 s (tinyllama b128 128/1024)
- timeout 4건은 `xmem_replay` 단계: 1 TiB 용량 재생에서 캐시가 계속 쌓이는 조건으로 재생 자체가 오래 걸림 (**추정**, §2.15의 용량 문제와 같은 조건군)
- 보관: `archive/main395_cpu_L1024_stopped_20261115.tar.gz` (약 398 MB, 2,341 항목: `runs/main395/` 전체, console log, `results/main395/`, manifest 일부), sha256 파일 동봉(`d8bd86db…`)
  - 원본 `runs/main395/`도 삭제되지 않고 남아 있음

### 2.15 GPU-CPU 정렬 (L1024, `handoff_codex/`)
- 계기: 11-15 17:04 INUGPU handoff(`cgf-fp16-inuos-handoff-20261115`, `INUOS_GPU_EXPERIMENT_SETUP_KO.md`)
  - GPU 조건표 L256 전환, RTX 3090 L1024 결과 395(measure ok 381 / gpu_oom 14) 조건별 CSV 제공
- 산출물
  - `handoff_codex/gpu_cpu_alignment.md`: 15항목 일치/불일치/구조적 불가 판정 (§4.3)
  - `handoff_codex/alignment_20261115/compare_l1024.py` → `.out`, `_rows.json` (17:11): 둘 다 ok인 374조건 모델별 오차
  - `alignment_20261115/replay_capacity.py` → `replay_capacity_24979374080.jsonl` (6행 17:31 커밋, 15행 완료 18:22)
- 재확인 (이번 조사에서 같은 입력으로 다시 계산)
  - 입력 sha256 전체 64자 일치 375/375 (CPU ok 중 GPU 행 있는 것 전부)
    - `compare_l1024.out`의 "input sha same 0"은 GPU CSV 형식(`k=v;k=v`)을 JSON으로만 파싱해 전부 None이 된 스크립트 문제
  - `_attn_implementation` 374/374 동일
  - 용량 근처 15조건 모두 GPU `measure.json`에서 warmup_0부터 `num_alloc_retries > 0` (1~14)
- `cpu_manifest.json` (11-12 11:28 생성, 4.3 MB, 폴더명으로 보아 Codex 에이전트 작성으로 **추정**): 당시 CPU 실행 상태·코드 근거·검증 감사. GPU 측에 전달된 파일

### 2.16 L256 조건표 전환 (11-15 18:02~18:08, 커밋 `912f7fe`)
- GPU 조건표 변경: LLM·T5 길이 {128,512,1024}² → {128,256,512}². GPT-2·비전·배치 그대로
  - `manifest/conditions_main395_l256.json` sha256 `d4a9b09caff3064c00c04358bf2e462996bced23de93da98652bd1aba04b7289`
  - `cases395.py --conditions ... --expect-sha ...` → `manifest/cases395_l256.jsonl` (sha256 `469c5ead…`)
  - 이전 표와 겹침 245, 새 150(256 포함), 빠짐 150(1024 포함)
- 결과 재사용: `tools/link_overlap.py` → `runs/main395_l256/cases/<cid>` → `../../main395/cases/<cid>` 상대 심볼릭 링크
  - `runs/main395_l256/reuse_manifest.json` (18:05:25): overlap 245, linked 244 (ok 239 + worker_killed_cpu_oom 5), skipped 1 (`t5-3b__b128__in512__new512`, 원본 결과 없음), to_run 151
  - 실행 대상 151: t5-3b 26, opt/tinyllama/qwen/t5-small/t5-base 각 25
  - OOM 5건(gpt2-xl b128 3, t5-3b b128 2)은 GPU에서도 gpu_oom → 재실행 안 함
- 새 스케줄러 `tools/run_l256.py` (worker v2 무수정)
  - P: 3슬롯 병렬, cpuset 0-6/7-13/14-20, 7 스레드, 슬롯당 42g, 8 h 제한, 예상 비용 큰 순
  - 메모리 감시(0.5 s): 호스트 MemAvailable < 5 GiB이고 2개 이상 실행 중이면 가장 늦게 시작한 작업 kill → `preempted_low_mem`
  - 입장 조건: MemAvailable ≥ 12 GiB, 마지막 kill 후 60 s, 마지막 시작 후 30 s
  - S: P 종료 후 메모리 실패·timeout을 0-19, 20 스레드, 42g, 제한 없음으로 순차. S 결과가 최종
  - 컨테이너 `--oom-score-adj 500`, 심볼릭 링크 조건 실행 거부
- 실행 스크립트 `docs/run_main395_l256.sh` (tmux `cgf-l256`로 실행 예정, README)
- 현재 상태 (조사 시점)
  - `runs/main395_l256/`에 `attempts.jsonl`·`suite_invocations.jsonl`·console log 없음, tmux `cgf-l256` 없음 → **151개 미실행**
  - `results/main395_l256/` (collect395 18:07): 재사용 244건만. ok 239, OOM 5. `pilot_vs_rtx3090.csv`는 main395와 동일
  - GPU L256 결과(`cgf-fp16-inuos-handoff-20261115/results/rtx3090_main395_L256/`) 아직 없음 → `compare_gpu_l256.py`, `check_gpu_overlap.py` 미실행
  - `capacity_rtx3090.jsonl`(본 용량 재생) 미생성. 1 TiB self-check 5건만 실행
- `manifest/gpu_profiles.json`: rtx3090 `capacity_bytes` 24,979,374,080 = `cuda_total_memory` 25,306,529,792 − allocator 밖 사용량 327,155,712 B(312 MiB)
  - 312 MiB = L1024 measure_0 381개 ok의 NVML 프로세스 피크 − torch reserved 피크 (min 312 / median 312 / max 316 MiB)
- INUGPU 요청서 `handoff_codex/INUGPU_CNN_TRACE_REQUEST_KO.md`: GPU L256 본실험 후 `resnet-18__b32__r224`, `mobilenet-v2-1.0-224__b32__r224`(여유 시 `resnet-50__b32__r224`) trace 모드 기록 요청. 회신 미도착

### 2.17 스레드 영향 확인 (`runs/thread_check`, 11-15 18:04~18:15)
- 동기: GPU torch 스레드 4, CPU는 A 5 / B 20 / L256 P 7 / S 20 → 요청 trace가 스레드에 따라 바뀌는지
- 명령: `docs/run_thread_check.sh` (tmux `cgf-threadcheck`): `run_l256.py --cpusets 0-19 --par-threads k --phases P`, k = 20, 7, 5, 1, 이후 `compare_requests_npz.py`를 이미지 안에서 실행
- 조건 4: `resnet-18__b8__r224`, `mobilenet-v2-1.0-224__b8__r224`, `gpt2__b1__in128__new128`, `t5-small__b1__in128__new128` → 16회 모두 ok
- 결과 (`results/thread_check/compare.jsonl`, 기준 20 스레드)

| 조건 | 요청 수 (20 / 7 / 5 / 1) | trace 완전 동일 | summary 동일 | 첫 크기 차이 위치 |
|---|---|---|---|---|
| resnet-18 b8 r224 | 166 / 166 / 166 / 166 | 0/3 (크기만 다름) | 3/3 | 57번째 요청 |
| mobilenet-v2 b8 r224 | 465 / 465 / 465 / 363 | 2/3 (7·5 동일, 1은 다름) | 3/3 | 19 (1 스레드) |
| gpt2 b1 128/128 | 57,510 전부 | 0/3 | 3/3 | 39 |
| t5-small b1 128/128 | 84,148 / 84,148 / 84,148 / 84,136 | 0/3 | 3/3 | 80 |

- 요청 바이트 합 차이 예: resnet-18 226,593,734 (20) vs 223,257,030 (1), gpt2 1,415,892,790 vs 1,398,120,646
- 커밋 메시지 해석: 스레드별 CPU scratch 버퍼 차이. 이 4조건에서는 예측 summary 12/12 동일. 큰 조건에서의 영향은 미확인
- `results/thread_check/tdiff.py`(차이 크기 쌍 집계)는 출력 저장 없음

### 2.18 용량 재생 (15조건, self-check 5조건)
- 동기: L1024 대조에서 LLM·T5 reserved 오차 >20% 15건이 전부 CPU 예측 reserved > GPU 용량 (1 TiB 재생이라 캐시 반환 없음)
- 15조건 분석 재생: `alignment_20261115/replay_capacity.py`(cpu_worker2.replay_protocol, `CAP` = 24,979,374,080) → 15/15 OOM 없음 (§4.4)
  - `replay_capacity.err`는 xMem 경고 `OOM when allocating …` 줄만 포함 (캐시 반환 후 재시도 과정의 로그로 **추정**, 최종 oom=false)
- 생산 도구 `tools/replay_capacity.py --run main395_l256 --cap-bytes 1099511627776 --self-check` → `results/main395_l256/selfcheck_1TiB.jsonl` (18:06): 5/5 `self_check_equal=true`, 캐시 반환 0

---

## 3. 코드 설명

### 3.1 `src/cgf680/` 전체
| 파일 | 줄 | 역할 | CLI |
|---|---|---|---|
| `__init__.py` | 0 | 패키지 | - |
| `cases.py` | 90 | 680 조건표 + 파일럿 26 생성, csv 동시 출력, sha 출력 | `PYTHONPATH=src python3 -m cgf680.cases --out manifest/cases.jsonl` |
| `cases395.py` | 34 | GPU 조건표(json) + `pilot.json` + `models.json` → cases jsonl (case_id = condition_id) | `python3 src/cgf680/cases395.py [--conditions F] [--out F] [--expect-sha SHA]` (기본: main395 → cases395.jsonl) |
| `protocol.py` | 153 | v1 공통 프로토콜 (로드 eager, 입력, generate 인자, invoke, 예외) | 라이브러리 |
| `cpu_worker.py` | 581 | v1 worker: 4회 실제 실행 추적 → 정규화 → 재생. v2가 `Session`·`parse_trace`·`normalize`·`Progress` 재사용 | `python -m cgf680.cpu_worker --case-id ID --out DIR [--cases] [--pins] [--threads 20] [--conv1d-colmajor] [--emulate-fp16-from-fp32] [--profiler-scope user\|all] [--keep-raw-trace] [--expected-input-sha JSON] [--stop-after trace\|parse]` |
| `protocol2.py` | 102 | v2 = GPU vram_worker 의미론 (기본 attention, GenerationConfig, 회차별 입력 복사) | 라이브러리 |
| `load_emu.py` | 141 | GPU 로드 단계 allocator 요청 합성 | 라이브러리 |
| `shadow.py` | 344 | FP16 shadow-dtype 태그, cast 재현, T5 분기 패치, 매칭·크기 변환 | 라이브러리 |
| `cpu_worker2.py` | 333 | v2 worker (생산) | `python -m cgf680.cpu_worker2 --case-id ID --out DIR [--cases /exp/manifest/cases395.jsonl] [--models] [--threads 20] [--keep-raw-trace] [--no-attn-probe]` |

### 3.2 `cpu_worker2.py` (v2, 생산 worker)
| 줄 범위 | 단계 |
|---|---|
| 1-14 | docstring: 6단계 절차 |
| 31-35 | import (v1에서 `MARK_BASE, MARKERS, Progress, Session, normalize, parse_trace, start_rss_sampler` 재사용), `CAP = 1<<40`(1 TiB), `REGION="cgf680::00::invoke::0"` |
| 42 | `CUBLAS_WS = 8,519,680 B` (변형값 전용) |
| 45-89 | `replay_protocol`: xMem `Replayer(policy="xmem_pinned")` |
| 53-58 | 로드 이벤트 재생, `after_load`(reserved/allocated/로드 중 피크/segment 수) |
| 62-87 | 같은 호출 trace를 4회(warmup_0~2, measure_0) 재생. 회차 시작 = start, 회차 내 최대 = peak, `m4_local` 위치 = end_with_output, 회차 끝 = released. 캐시 유지 |
| 63-66 | `gpu_workspace>0`일 때 warmup_0 시작에 해제 안 되는 요청 1개 주입 (변형) |
| 92-120 | `attention_probe`: B=1 복제, 2토큰 생성, 전체 op 범위 profiler로 attention 관련 aten 이름 집계 |
| 123-132 | 인자 |
| 133-150 | 출력 폴더, `progress.json`, RSS 샘플러(1 s, `mem.jsonl`), 결과 골격(protocol 설명 포함) |
| 153-158 | `fail()` |
| 160-170 | 환경 기록, `verify_xmem("/work/xMem", lock)` |
| 172-184 | 1단계: FP16 모델 로드(구조용) → 스냅샷 경로 → `load_emu.synthesize` → `to_events` → 모델 해제 |
| 185-193 | 2단계: FP32 모델 로드(`finish_load`), `model_info`, 위치 길이 검사(`invalid_input`) |
| 194-203 | 입력 생성(FP16 기준 해시 기록, 실행은 FP32 cast), GenerationConfig, T5 패치, `ShadowMode` |
| 205-231 | 3단계: gc 비활성, USER_SCOPE 세션, M0 → `record_function(REGION)` + shadow + inference_mode 안에서 `invoke` → M4 → `del gi, o` → M5, 생성 길이 검사 |
| 233-250 | trace export → 파싱(`[memory]` CPU 이벤트) → `normalize` → 구간·표식 검사, raw trace 삭제 |
| 252-267 | 4단계: `match_records` → `emulate_fp16` → M0~M5 구간 잘라 `invocation_fp16_requests.npz` 저장(+`m4_local`, `load_events`) |
| 269-290 | 5단계: `replay_protocol` → OOM이면 예외 → `summary`(GPU 키) |
| 291-296 | cuBLAS workspace 변형 재생 (`variant_with_cublas_workspace`) |
| 297-298 | `finite_check = not_assessable`, status ok |
| 300-308 | 6단계: attention probe (실패해도 예측 유효) |
| 309-320 | 실패 분류: `invalid_input`, `generation_length_mismatch`, `cpu_oom`, `fp16_unsupported`, `compat_or_runtime_error` |
| 321-329 | `result.json` 원자적 저장, `CGF_RESULT` 출력 |

### 3.3 `protocol2.py`
| 줄 범위 | 단계 |
|---|---|
| 1-10 | v1과의 차이 설명 |
| 19-22 | SEED 20261006, N_WARMUP 3, loader 매핑 |
| 29-33 | `load`: `from_pretrained(revision, torch_dtype, low_cpu_mem_usage, use_safetensors, local_files_only)`, attention 인자 없음 |
| 36-40 | `finish_load`: `.to(cpu, dtype).eval()`, output_attentions/hidden_states=False |
| 43-62 | special id(pad/eos/bos/decoder_start) → 3, 입력 생성 (텍스트 randint int64 + mask 1, 비전 randn FP32 → FP16) |
| 65-70 | `check_valid`: L+N > max_position_embeddings 검사 |
| 73-83 | `generation_config`: `from_model_config` + greedy, min=max=N, pad=pad or eos, `model.generation_config` 교체 |
| 86-94 | `invoke`: 입력 clone(=GPU `.to(cuda)`), 비전 forward / 텍스트 generate, 새 토큰 수 반환 |
| 97-102 | `InvalidInput`, `LengthMismatch` |

### 3.4 `load_emu.py`
| 줄 범위 | 단계 |
|---|---|
| 1-18 | 합성 규칙 1~4 (transformers 4.52.4 기준) |
| 28-30 | dtype 크기표 |
| 33-45 | `checkpoint_entries`: safetensors 헤더 직접 읽기(파일·키 정렬) / `.bin`은 `torch.load(map_location="meta")` |
| 48-57 | `_post_order_tensors`: `Module._apply` 순서(자식 먼저) |
| 60-124 | `synthesize(model_fp16, snapshot_dir, use_safetensors)` |
| 63-69 | base prefix 없는 체크포인트 처리, `_keep_in_fp32_modules` |
| 76-83 | load dtype 결정, warmup W alloc/free |
| 88-107 | 체크포인트 순서 alloc, dtype 변환 시 ckpt 크기 임시 버퍼, tie 중복은 즉시 free |
| 108-113 | non-persistent 버퍼 alloc |
| 114-123 | `.to(fp16)` cast: alloc(fp16) + free(원본) |
| 127-137 | `to_events`: (action, size, id) 리스트, 라벨 기준 짝 |
| 140-141 | `signed` |

### 3.5 `shadow.py`
| 줄 범위 | 단계 |
|---|---|
| 1-17 | 개요, 알려진 한계 (kernel 내부 임시 버퍼는 둘러싼 호출 태그, 0-dim 승격 예외 등) |
| 23-37 | 대상 함수 집합, `shadow_of` (기본: 부동소수 → `h`, 나머지 `x`) |
| 40-69 | 텐서 수집, cast 목표 dtype 판정 |
| 77-120 | `explicit_fp32_at_call`: 호출 소스 구간에서 `float32`/`torch.float` 문자 검색 (Python 3.11 co_positions) |
| 123-202 | `ShadowMode(TorchFunctionMode)` |
| 132-151 | `_emulate`: FP32에선 no-op인 cast를 uint8 버퍼로 재현, 다운캐스트 원본 주소 기록 |
| 153-194 | `__torch_function__`: cast 처리 / 출력 태그(`dtype=float32` 명시면 `f`, 입력에 `f` 있으면 `f`, 아니면 `h`) |
| 196-202 | 새 출력 storage 기록 |
| 206-240 | `patch_fp16_branches` (T5만): `T5LayerNorm.forward` 분기 강제, `T5Block.forward` clamp 분기 → shadow 판정 |
| 244-307 | `match_records`: 주소+크기, 호출 단위 단조 커서로 매칭, 미매칭은 다음 매칭 태그 상속 |
| 310-344 | `emulate_fp16`: `h`/`inferred_h` 크기 절반, 다운캐스트 원본 조기 free, 이후 원래 free 제거, 인덱스 매핑 반환 |

### 3.6 `cpu_worker.py` (v1)
| 줄 범위 | 단계 |
|---|---|
| 1-19 | 출력 파일·재생 규칙 |
| 39-42 | 표식 크기(1,234,567+2k), M0~M5 이름, 용량 1 TiB |
| 49-81 | `Progress`, RSS 샘플러 |
| 84-93 | 고유 storage 크기 (tie 1회) |
| 96-110 | 매 forward logits numpy view finite 검사 (c10 allocator 미경유) |
| 113-123 | 예외 분류 (`cpu_ram_oom`, `fp16_unsupported`, `input_mismatch`, `error`) |
| 126-157 | `Session`: 저수준 kineto, `profile_memory=True`, USER_SCOPE 또는 전체 |
| 161-180 | `parse_trace`: chrome trace 스트리밍, `cgf680::` 구간 + CPU `[memory]`(Device Type 0) |
| 183-237 | `normalize`: Ev Idx 순 정렬, 구간 밖 alloc 제외(표식만 위치 기록), 주소로 free 짝, 통계 |
| 241-276 | `replay`: 상주 → 요청 순서 재생, 표식 위치 스냅샷, 구간별 피크 |
| 279-293 | `measure_stats`: start=M3, peak=max(M3~M5), end=M4, released=M5 |
| 297-313 | 인자 |
| 357-382 | 모델 로드 (에뮬이면 FP32), 형태 검사, Conv1D col-major |
| 384-425 | 입력, 입력 sha 대조, 상주 크기 (에뮬은 부동소수 절반) |
| 441-480 | profiler 세션: M0, warmup 3 + measure 1 각각 `cgf680::<j>::<phase>::<i>` 구간, M1~M5 |
| 482-521 | export → 파싱 → 정규화 → `events.npz` |
| 523-543 | 에뮬: FP32 원값 재생, 매칭, FP16 변환, `events_fp16_emulated.npz` |
| 544-562 | xMem 재생, `prediction`, 상태 (`replay_capacity_oom`, `nonfinite_output`, ok) |
| 563-577 | 예외 처리, `result.json` 저장 |

### 3.7 `tools/` 전체
| 파일 | 역할 | CLI / 비고 |
|---|---|---|
| `run_suite.py` | v1 순차 실행기 (케이스별 새 컨테이너, `--cpus`, 재개, `.attempt-<t>` 보존) | `--run-name X --select pilot\|all\|ids [--ids] [--models] [--order cost] [--threads 20] [--mem 42g] [--timeout-hours] [--retry-failed] [--conv1d-colmajor] [--emulate-fp16-from-fp32] [--profiler-scope] [--cases-file]` |
| `collect.py` | v1 결과 → `results/<run>/cpu_results.{csv,jsonl}`, `summary.json` | `--run-name X` |
| `compare_gpu.py` | v1 결과와 GPU 결과 조인 (revision·입력 sha·프로토콜 태그) | `--cpu --gpu --gpu-label --out [--protocol] [--map]`. 실제 사용 흔적 없음 |
| `compare_events.py` | v1 두 실행의 `events.npz` 완전 일치 | `A_DIR B_DIR` |
| `estimate_flops.py` | FLOP 기반 680 소요 추정 (FP16 속도 상수) | 인자 없음 |
| `fit_eta.py` | 보정 실행으로 모델별 시간 상수 적합 → `results/<run>/eta.json` | `--run-name fp32-calib [--dtype] [--timeout-hours ...]` |
| `replay_variant.py` | trace 크기 일괄 배율 재생 vs 기준 비교 | `--events DIR --resident-from DIR --scale 0.5 [--ref DIR]` |
| `compare_emulated.py` | 에뮬 trace vs 네이티브 FP16 trace 비교 | `EMU_DIR REF_DIR` |
| `check_run_identity.py` | v1 trace 4회 요청 동일성 | `CASE_DIR ...` |
| `check_replicated_replay.py` | 회차 0 ×4 합성 재생 = 실제 4회 재생 여부 | `CASE_DIR ...` |
| `run395.py` | v2 395 스케줄러 (A 병렬 / B 순차 / C 재시도) | `--run-name main395 --select pilot\|all\|ids [--ids] [--phases A B C] [--timeout-hours 2]` |
| `collect395.py` | v2 결과 수집, 상태 요약, 파일럿 vs RTX 3090, 선택적 GPU 비교 | `--run-name X [--gpu F --gpu-label L]` |
| `compare_gpu_trace.py` | GPU snapshot measure_0 요청 vs CPU 요청 multiset | 이미지 안에서 `CID ...` (`/handoff` mount 가정) |
| `link_overlap.py` | 두 조건표 공통 조건의 결과를 심볼릭 링크로 재사용 | `--src --src-cases --dst --dst-cases` |
| `run_l256.py` | L256 스케줄러 (P 병렬+선점 / S 순차) | `--run-name --cases [--ids] [--cpusets 0-6,7-13,14-20] [--par-threads 7] [--par-mem 42g] [--par-timeout-hours 8] [--kill-below-gib 5] [--admit-gib 12] [--admit-after-kill-s 60] [--admit-gap-s 30] [--seq-cpus 0-19] [--seq-threads 20] [--seq-mem 42g] [--phases P S]` |
| `replay_capacity.py` | 저장된 요청 trace를 GPU 용량으로 재생 (후처리) | 이미지 안: `--run R (--gpu NAME \| --cap-bytes N) [--ids] [--jobs 4] [--self-check] [--out]` |
| `compare_gpu_l256.py` | CPU 예측(+용량 재생) vs GPU 조건별 CSV, 모델별 오차표 md | `--run --gpu-csv --gpu-label [--capacity rtx3090]`. 미실행 |
| `check_gpu_overlap.py` | GPU L1024 vs L256 겹치는 245개 값 동일성 | `OLD.csv NEW.csv [--out]`. 미실행 |
| `compare_requests_npz.py` | v2 `invocation_fp16_requests.npz` 완전 일치 (action·size·짝·m4·로드) | 이미지 안: `REF_DIR OTHER_DIR ...` |

### 3.8 `run395.py`
| 줄 범위 | 단계 |
|---|---|
| 1-14 | 단계 설명 |
| 27-35 | 경로 상수, 이미지, A 슬롯 cpuset 4개, 작은 모델 집합 |
| 43-51 | `is_small`: 비전 B·R² ≤ 32·256², 텍스트 B·(L+N) ≤ 16K (gpt2-medium/t5-base는 8K) |
| 54-60 | `cost`: 모델 순위 × 크기 |
| 63-86 | `meminfo`, cgroup `memory.current`/`memory.peak` 1 s 폴링 |
| 89-157 | `run_one`: 기존 폴더 rename → `docker run -d` (network none, memory=swap, cpuset, 환경변수, 4개 bind mount) → `docker logs -f` → `docker wait`(timeout 시 kill) → inspect/rm → runner 기록 → 상태 덮어쓰기(`timeout`, `worker_killed_cpu_oom[_suspected]`, `worker_exit_<rc>`) |
| 160-173 | `attempts.jsonl` 기록, 현재 상태 조회 |
| 176-197 | 인자, 선택, `suite_invocations.jsonl`, todo = 결과 없는 조건 |
| 198-215 | A: 큐 + 4 스레드, 5 스레드/10g |
| 216-220 | B: 순차, 20 스레드/42g |
| 221-232 | C: A/B timeout + A OOM, 제한 없음, attempt 2 |

### 3.9 `run_l256.py`
| 줄 범위 | 단계 |
|---|---|
| 1-18 | P/S 설명 |
| 31-37 | `run395` 상수·함수 재사용, `MEMORY_FAIL`, `DEFER` |
| 40-46 | `Job` |
| 57-138 | `run_job`: 심볼릭 링크 거부, docker 실행(`--oom-score-adj 500`, `--cases` 전달), running 등록, 대기/timeout, 상태 결정(선점 > timeout > 비정상 종료), runner 기록(`scheduler`, `slot`, `preempted`, `cases_manifest`) |
| 141-146 | `status_of` |
| 149-174 | 인자, 조건 로드, `suite_invocations.jsonl` |
| 176-184 | 결과 없음 → P, P에서 DEFER 상태 → S, P는 cost 내림차순 |
| 186-208 | 감시 스레드: MemAvailable < kill 기준 & 2개 이상 → 최근 시작 작업 kill, `scheduler_events.jsonl` 기록, 10 s 대기 |
| 213-222 | 슬롯 작업 종료 시 DEFER면 S 목록 추가 |
| 224-246 | 입장 루프 (빈 슬롯, kill 후 60 s, 시작 간 30 s, MemAvailable ≥ 12 GiB) |
| 248-261 | S: cost 오름차순, 0-19/20 스레드/42g/제한 없음 |

### 3.10 `collect395.py`
| 줄 범위 | 단계 |
|---|---|
| 15-21 | 경로, GPU 파일럿 reference(`…handoff-20261112/reference/pilot_rtx3090_eager_vs_default.csv`), summary 키 11개 |
| 24-65 | `row`: 조건, 상태, 입력 sha, summary, ws 변형, FP32/FP16 요청 bytes, cast 수, attention ops, 단계별 시간, RSS, 컨테이너 피크, cpuset/스레드, 이미지, 버전, 파일 경로 |
| 82-101 | `.attempt-` 제외 수집 → `cpu_predictions.{csv,jsonl}`, `status_summary.json`(상태, 단계×상태, 실패, 실패 시도 이력) |
| 103-126 | 파일럿 vs RTX 3090 기본 attention: after_load R, 로드 피크 A, m1 start R/A, m1 peak R/A (MiB, diff, rel), ws 변형 rel. 입력 sha는 GPU CSV가 16자 접두라 **접두 비교** |
| 127-131 | 요약 출력 (mean |rel|, max) |
| 132-150 | `--gpu`: condition_id 조인 → `compare_<label>.csv` |

### 3.11 `replay_capacity.py`
| 줄 범위 | 단계 |
|---|---|
| 1-11 | 용도·실행 예 |
| 25-65 | `replay(cdir, cap)`: npz 로드 → 로드 이벤트 → 4회 재생, malloc 중 reserved 감소 = 캐시 반환 횟수·bytes 집계, OOM 시 중단 |
| 68-94 | `one`: 행 생성(`oom_predicted`, `oom_stage`, after_load, m1 start/peak, 캐시 반환, 1 TiB worker 값), `--self-check`면 6개 키 일치 여부 |
| 102-117 | 인자, 용량 = `gpu_profiles.json` 또는 `--cap-bytes` |
| 118-131 | ok + npz 있는 조건만, 이미 처리한 행 건너뜀(재개) |
| 133-138 | `multiprocessing.Pool(jobs)`로 병렬, 행 단위 append |

### 3.12 `run_suite.py` (v1) 주요 줄
| 줄 범위 | 단계 |
|---|---|
| 28-29 | FINAL / RETRYABLE 상태 |
| 69-77 | 재개: FINAL 또는 (RETRYABLE & `--retry-failed` 없음) 건너뜀, 아니면 `.attempt-<t>`로 rename |
| 82-99 | docker 실행 (`--cpus`, v1 worker 옵션 전달) |
| 111-151 | 대기/timeout, `runner.json`, 상태 (`timeout`, `cpu_ram_oom`, `killed`, `no_result`) |
| 186-191 | cost 순 정렬 |

### 3.13 `docs/*.sh`
| 파일 | 역할 | 실행 여부 |
|---|---|---|
| `download.sh` + `download_models.py` | pinned config/가중치 다운로드 (네트워크 필요) | 11-11 15:18 (`runs/download.log`), t5-3b 15:37 |
| `run_in_image.sh` | 고정 이미지에서 python 실행 (network none, 기본 16g/4 CPU, 4개 mount) | 분석 도구용 |
| `run_pilot_text_a.sh` | v1 파일럿 텍스트 5모델 | 15:13 시작, 3번째에서 중단 (`pilot-fullscope-ref`) |
| `run_pilot.sh` | v1 파일럿 26 | 11/26 후 중단 |
| `run_verify_layout.sh` | Conv1D row-major 파일럿 재실행 + 비교 | 미실행 |
| `run_fp32_calib.sh` | FP32 보정 26 | 실행 |
| `run_bf16_calib.sh` | BF16 보정 26 | 미실행 |
| `run_pilot395.sh` | v2 파일럿 26 (main395 폴더) | 실행 |
| `run_main395.sh` | v2 본실험 체인 (A B C + collect) | 실행, B 도중 중단 |
| `run_main395_l256.sh` | L256 151개 P/S + collect | 미실행 |
| `run_thread_check.sh` | 스레드 1/5/7/20 비교 | 실행 |

### 3.14 `tests/probes/`
- `bf16_ops.py`, `bmm_layouts.py`: FP32/BF16/FP16 bmm·layout별 GFLOPS 측정
- `clock_probe.py`: profiler 시각과 Python 시계 대조
- `rss_profiler_probe.py`: profiler 범위별 RSS
- `load_seq.py`, `load_seq2.py`, `load_synth.py`: GPU 로드 순서 분석·합성 초안
- `validate_load_emu.py`: `load_emu` 13모델 검증

### 3.15 출력 스키마
- v2 `runs/<run>/cases/<cid>/`
  - `result.json`
    - 식별: `condition_id`, `case`, `model`, `revision`, `status`, `failed_stage`, `dtype_mode`, `protocol`
    - 환경: `environment`(python/torch/transformers/threads/cpu_capability/image/image_id/mem_limit/cpuset), `xmem`(commit, 파일 검증)
    - 로드: `load_synthesis`(warmup_bytes, ckpt_skipped_unexpected, tied_dup, keep_fp32, loaded_into, post_load_casts, events, live_blocks), `after_load`
    - 실행: `model_info`(class, attn_implementation, FP16 로드 시 attn), `inputs`(shape/dtype/sha256 전체), `special_tokens_replaced_by_3`, `generation_config`, `fp16_branch_patches`, `profiler`, `new_tokens`
    - trace: `trace_stats`(memory_events, unmatched_frees, marker_positions 등), `fp16_emulation`(torch_function_calls, emulated_casts, undetermined_casts, match·tag_counts·tag_bytes, early_freed, fp32/fp16 요청 bytes)
    - 재생: `runs`(4회 start/peak/end_with_output/released/segments_end), `replay`(oom, unknown_frees, live_blocks_at_end)
    - **`summary`**: `first_measure_{start,peak,end_with_output,released}_{reserved,allocated}`, `after_load_{reserved,allocated}`, `load_peak_allocated` (byte, GPU와 같은 키)
    - `variant_with_cublas_workspace`(예측 아님), `finite_check`, `attention_ops_cpu`, `timings_s`, `rss`, `files`
    - runner가 덧붙임: `runner`(phase, attempt/slot, cpuset, threads, mem_limit, timeout_s, exit_code, oom_killed, timed_out, preempted, wall_seconds, container_memory_peak_bytes, host_mem_before, image_id, started, container)
  - `invocation_fp16_requests.npz`: `action`(1/0), `size`(FP16 기준), `alloc_id`, `region`, `m4_local`, `load_events`(N×3)
  - `mem.jsonl`(RSS 1 s), `progress.json`, `container.log`
- v1 `runs/<run>/cases/<cid>/`: `result.json`(`prediction` = `allocated/reserved_{start,peak,end,end_after_output_release}_bytes`, `replay`, `resident`, `runs`, `trace_stats`, `finite_check`, 에뮬 시 `prediction_fp32_raw`, `fp16_emulation`), `events.npz`, `events_fp16_emulated.npz`, `runner.json`
- 실행 폴더: `attempts.jsonl`, `suite_args.json`(v1) / `suite_invocations.jsonl`(v2), `scheduler_events.jsonl`(run_l256)
- `results/<run>/`
  - v2: `cpu_predictions.{csv,jsonl}` (조건당 1행, `collect395.row` 열), `status_summary.json`, `pilot_vs_rtx3090.csv`, (예정) `compare_<label>.csv`, `compare_<label>_summary.md`, `capacity_<gpu>.jsonl`
  - v1: `cpu_results.{csv,jsonl}`, `summary.json`, `eta.json`

---

## 4. 결과 요약

### 4.1 pilot395 vs RTX 3090 기본 attention 파일럿 (`results/main395/pilot_vs_rtx3090.csv`, 26/26)
- 입력 sha 접두 일치 26/26, attention 이름 일치 26/26, after_load reserved 26/26 동일
- m1 peak reserved: 평균 |오차| 5.09%, 최대 30.7% / m1 peak allocated: 평균 4.68%, 최대 36.8%

| condition_id | attn | m1 peak R CPU/GPU MiB (rel) | m1 peak A CPU/GPU MiB (rel) | ws 변형 R / A rel |
|---|---|---|---|---|
| opt-125m b8 128/128 | sdpa | 416.0/418.0 (−0.48%) | 380.0/388.1 (−2.10%) | −0.48% / −0.01% |
| opt-125m b16 512/512 | sdpa | 1510.0/1524.0 (−0.92%) | 1325.9/1334.0 (−0.61%) | −0.13% / 0.00% |
| tinyllama b8 128/128 | sdpa | 2330.0/2330.0 (0.00%) | 2173.5/2182.5 (−0.41%) | 0.00% / 0.00% |
| tinyllama b16 512/512 | sdpa | 2882.0/2926.0 (−1.50%) | 2670.4/2678.6 (−0.30%) | −0.82% / 0.00% |
| qwen2.5 b8 128/128 | sdpa | 3512.0/3510.0 (+0.06%) | 3040.4/3048.6 (−0.27%) | +0.06% / 0.00% |
| qwen2.5 b16 512/512 | sdpa | 4008.0/3868.0 (+3.62%) | 3689.2/3697.3 (−0.22%) | 0.00% / 0.00% |
| gpt2 b8 128/128 | sdpa | 474.0/476.0 (−0.42%) | 409.6/417.7 (−1.95%) | −0.42% / +0.13% |
| gpt2 b16 512/512 | sdpa | 1836.0/1838.0 (−0.11%) | 1481.8/1490.8 (−0.60%) | −0.11% / 0.00% |
| gpt2-medium b8 128/128 | sdpa | 1004.0/1004.0 (0.00%) | 950.8/958.8 (−0.84%) | 0.00% / 0.00% |
| gpt2-medium b16 512/512 | sdpa | 3324.0/3390.0 (−1.95%) | 2655.0/2663.2 (−0.31%) | −0.06% / 0.00% |
| gpt2-xl b8 128/128 | sdpa | 3778.0/3812.0 (−0.89%) | 3627.9/3636.2 (−0.23%) | −0.52% / +0.01% |
| gpt2-xl b16 512/512 | sdpa | 9172.0/9174.0 (−0.02%) | 7877.7/7885.1 (−0.09%) | −0.02% / +0.01% |
| t5-small b8 128/128 | eager | 306.0/304.0 (+0.66%) | 146.1/154.2 (−5.25%) | +0.66% / +0.02% |
| t5-small b16 512/512 | eager | 738.0/738.0 (0.00%) | 547.6/555.7 (−1.47%) | 0.00% / 0.00% |
| t5-base b8 128/128 | eager | 710.0/710.0 (0.00%) | 506.7/515.7 (−1.74%) | 0.00% / 0.00% |
| t5-base b16 512/512 | eager | 1474.0/1474.0 (0.00%) | 1073.3/1082.3 (−0.83%) | 0.00% / 0.00% |
| t5-3b b8 128/128 | eager | 7220.0/7220.0 (0.00%) | 6219.8/6228.5 (−0.14%) | 0.00% / 0.00% |
| t5-3b b16 512/512 | eager | 12554.0/12996.0 (−3.40%) | 11671.7/11678.9 (−0.06%) | −0.94% / −0.01% |
| vit-large b8 r224 | sdpa | 640.0/656.0 (−2.44%) | 624.7/632.9 (−1.29%) | +1.83% / +0.14% |
| vit-large b16 r224 | sdpa | 732.0/738.0 (−0.81%) | 666.1/674.2 (−1.20%) | −0.81% / 0.00% |
| resnet-18 b8 r224 | eager | 66.0/80.0 (−17.5%) | 49.5/61.2 (−19.1%) | 0.00% / −4.40% |
| resnet-18 b16 r512 | eager | 330.0/458.0 (−27.95%) | 302.7/343.7 (−11.93%) | −27.95% / −9.31% |
| resnet-50 b8 r224 | eager | 112.0/132.0 (−15.15%) | 95.5/107.4 (−11.08%) | −4.55% / −2.70% |
| resnet-50 b16 r512 | eager | 618.0/812.0 (−23.89%) | 530.4/563.1 (−5.81%) | −23.89% / −4.21% |
| mobilenet-v2 b8 r224 | eager | 88.0/88.0 (0.00%) | 73.0/62.3 (+17.17%) | +22.73% / +30.01% |
| mobilenet-v2 b16 r512 | eager | 1108.0/848.0 (+30.66%) | 690.8/505.0 (+36.80%) | +30.66% / +38.41% |

- 경향: LLM·T5·ViT allocated는 CPU가 약 8 MiB 작게 나오는 경우가 많음 (GPU 전용 workspace 등으로 **추정**). ws 변형(8.125 MiB 주입)을 넣으면 LLM allocated가 거의 0%가 됨

### 4.2 main395 상태 (§2.14 재게)
- 395 중 결과 388: ok 375 / `worker_killed_cpu_oom` 6 / `timeout` 7, 결과 없음 7
- GPU L1024 measure gpu_oom 14개와 CPU 상태 대조 (이번 조사 재계산)

| GPU gpu_oom 조건 (모두 warmup_0) | CPU 상태 |
|---|---|
| gpt2-xl b128 128/512, 512/128, 512/512 | worker_killed_cpu_oom ×3 |
| t5-3b b128 128/512, 512/128 | worker_killed_cpu_oom ×2 |
| t5-3b b32 1024/1024 | worker_killed_cpu_oom |
| t5-3b b32 512/1024 | timeout (xmem_replay) |
| t5-3b b32 1024/512 | **ok**: 1 TiB 예측 peak reserved 29,043,458,048 B(27,698 MiB) > 용량 24,979,374,080 B → 용량 판정 시 OOM으로 분류 가능 |
| t5-3b b128 128/1024, 512/512, 512/1024, 1024/128, 1024/512, 1024/1024 | 결과 없음 ×6 |

- 반대로 CPU OOM/timeout 중 GPU ok: tinyllama b128 512/1024·1024/1024, qwen b128 512/1024, t5-3b b16 128/1024·512/1024, t5-3b b32 128/1024 (timeout 6) → GPU 값은 있으나 CPU 예측 없음

### 4.3 GPU-CPU 정렬 판정 (`handoff_codex/gpu_cpu_alignment.md` 요약)
| # | 항목 | 판정 | 근거 요지 |
|---|---|---|---|
| 1 | 조건표 | 불일치 → L256 전환으로 대응 중 | 겹침 245, 새 150 미실행 |
| 2 | 모델·revision | 일치 | models.json sha 동일 |
| 3 | 입력 | 일치 | sha256 375/375 |
| 4 | dtype | 구조적 불가 | FP32 실행 + shadow 환산, LLM·T5 allocated 중앙값 −0.1~−1.5% |
| 5 | attention | 이름 일치 / 커널 구조적 차이 | 374/374 동일, GPU flash/mem-eff vs CPU sdpa(`_scaled_dot_product_flash_attention_for_cpu`) |
| 6 | 생성 설정 | 일치 | 코드 대조 |
| 7 | 측정 순서 | 일치 (measure_0 기준) | 로드 합성 → 4회 재생, GPU measure 1~4 없음 |
| 8 | 텐서 수명 | 일치 | clone, M4/M5 |
| 9 | 기준값 | 일치 | after_load reserved 374/374 오차 0% |
| 10 | 용량/OOM | 불일치 | xMem 1 TiB → LLM·T5 큰 오차 15건 전부 원인 |
| 11 | GPU 전용 할당 | 구조적 불가 | cuBLAS 8,519,680 B, cuBLASLt 1 MiB, cuDNN workspace |
| 12 | 프로세스 격리 | 일치 | 조건마다 새 컨테이너 |
| 13 | 단위 | 일치 | byte |
| 14 | torch 스레드 | 불일치(영향 미확인 → §2.17에서 소규모 확인) | GPU 4, CPU 5/20(L256 7/20) |
| 15 | GPU OOM 조건 | 구조적 제약 | CPU RAM 42g 한도로 예측 자체 실패, FP32라 메모리 2배 이상 |

### 4.4 CPU vs RTX 3090 L1024 모델별 오차 (둘 다 ok 374, `alignment_20261115/compare_l1024.out`, rel = (CPU−GPU)/GPU)
- m1 peak reserved (1 TiB 재생)

| 모델 | n | ≤1% | ≤5% | ≤20% | >20% | 중앙값 | 최소 | 최대 |
|---|---|---|---|---|---|---|---|---|
| opt-125m | 45 | 33 | 41 | 43 | 2 | −0.000 | −0.113 | +18.473 |
| TinyLlama-1.1B | 43 | 25 | 41 | 41 | 2 | −0.008 | −0.043 | +10.467 |
| Qwen2.5-1.5B | 43 | 24 | 37 | 42 | 1 | +0.000 | −0.079 | +0.742 |
| gpt2 | 20 | 14 | 19 | 20 | 0 | −0.001 | −0.053 | +0.043 |
| gpt2-medium | 20 | 14 | 19 | 19 | 1 | +0.000 | −0.024 | +5.833 |
| gpt2-xl | 17 | 15 | 16 | 16 | 1 | −0.000 | −0.012 | +2.868 |
| resnet-18 | 20 | 10 | 10 | 15 | 5 | −0.047 | −0.360 | +0.000 |
| resnet-50 | 20 | 4 | 6 | 10 | 10 | −0.208 | −0.292 | +0.023 |
| mobilenet_v2 | 20 | 4 | 4 | 7 | 13 | +0.307 | +0.000 | +0.409 |
| vit-large | 5 | 2 | 5 | 5 | 0 | −0.019 | −0.024 | +0.000 |
| t5-small | 45 | 38 | 41 | 44 | 1 | +0.000 | −0.072 | +4.660 |
| t5-base | 45 | 35 | 39 | 42 | 3 | +0.000 | −0.006 | +9.980 |
| t5-3b | 31 | 23 | 26 | 27 | 4 | +0.000 | −0.034 | +7.868 |

- m1 peak allocated

| 모델 | n | ≤1% | ≤5% | ≤20% | >20% | 중앙값 | 최소 | 최대 |
|---|---|---|---|---|---|---|---|---|
| opt-125m | 45 | 27 | 45 | 45 | 0 | −0.006 | −0.031 | −0.000 |
| TinyLlama-1.1B | 43 | 43 | 43 | 43 | 0 | −0.003 | −0.004 | −0.001 |
| Qwen2.5-1.5B | 43 | 43 | 43 | 43 | 0 | −0.002 | −0.003 | −0.001 |
| gpt2 | 20 | 12 | 20 | 20 | 0 | −0.009 | −0.032 | −0.001 |
| gpt2-medium | 20 | 18 | 20 | 20 | 0 | −0.005 | −0.011 | −0.000 |
| gpt2-xl | 17 | 17 | 17 | 17 | 0 | −0.002 | −0.003 | −0.001 |
| resnet-18 | 20 | 0 | 0 | 14 | 6 | −0.157 | −0.355 | −0.105 |
| resnet-50 | 20 | 0 | 1 | 20 | 0 | −0.087 | −0.154 | −0.019 |
| mobilenet_v2 | 20 | 0 | 1 | 4 | 16 | +0.280 | −0.442 | +0.392 |
| vit-large | 5 | 1 | 5 | 5 | 0 | −0.012 | −0.013 | −0.006 |
| t5-small | 45 | 20 | 38 | 45 | 0 | −0.015 | −0.064 | −0.001 |
| t5-base | 45 | 27 | 45 | 45 | 0 | −0.007 | −0.020 | −0.000 |
| t5-3b | 31 | 31 | 31 | 31 | 0 | −0.001 | −0.001 | −0.000 |

- after_load reserved: 13모델 374/374 오차 0
- m1 start reserved: peak reserved 표와 같은 분포 (warmup에서 캐시가 다 만들어져 measure_0 동안 reserved가 늘지 않음)
- reserved >20% 43건 = 비전 28(resnet-18 5, resnet-50 10, mobilenet 13) + LLM·T5 15(전부 CPU 1 TiB 예측 reserved > 용량)

### 4.5 용량 재생 (15조건, 용량 24,979,374,080 B, `replay_capacity_24979374080.jsonl`)
| 조건 | GPU m1 peak R MiB | CPU 1 TiB MiB (rel) | CPU 용량 재생 MiB (rel) | peak A CPU/GPU MiB | GPU retries(warmup_0) |
|---|---|---|---|---|---|
| opt-125m b128 128/512 | 4,972 | 25,076 (+4.04) | 4,640 (−6.7%) | 3,289/3,297 | 1 |
| opt-125m b128 128/1024 | 8,986 | 174,988 (+18.47) | 12,380 (+37.8%) | 5,692/5,700 | 9 |
| tinyllama b128 128/512 | 20,730 | 40,340 (+0.95) | 21,420 (+3.3%) | 4,533/4,541 | 1 |
| tinyllama b128 128/1024 | 13,542 | 155,286 (+10.47) | 20,054 (+48.1%) | 6,453/6,461 | 8 |
| qwen2.5 b128 128/1024 | 20,562 | 35,826 (+0.74) | 20,556 (−0.0%) | 7,990/7,999 | 1 |
| gpt2-medium b128 128/512 | 18,578 | 126,938 (+5.83) | 18,276 (−1.6%) | 8,592/8,599 | 7 |
| gpt2-xl b32 128/512 | 15,458 | 59,784 (+2.87) | 14,784 (−4.4%) | 9,086/9,095 | 3 |
| t5-small b128 128/1024 | 9,588 | 54,266 (+4.66) | 10,302 (+7.4%) | 2,024/2,032 | 2 |
| t5-base b128 128/512 | 16,430 | 58,460 (+2.56) | 17,020 (+3.6%) | 3,461/3,471 | 2 |
| t5-base b128 128/1024 | 21,302 | 233,892 (+9.98) | 6,476 (−69.6%) | 5,863/5,872 | 11 |
| t5-base b128 512/1024 | 13,762 | 27,920 (+1.03) | 12,828 (−6.8%) | 7,668/7,674 | 1 |
| t5-3b b8 128/1024 | 16,870 | 31,684 (+0.88) | 16,706 (−1.0%) | 8,964/8,972 | 1 |
| t5-3b b8 512/1024 | 22,024 | 35,396 (+0.61) | 21,472 (−2.5%) | 10,122/10,130 | 1 |
| t5-3b b16 128/512 | 10,268 | 39,288 (+2.83) | 10,080 (−1.8%) | 9,352/9,360 | 2 |
| t5-3b b32 128/512 | 21,150 | 187,554 (+7.87) | 20,158 (−4.7%) | 13,265/13,273 | 14 |

- ±5% 9건, 5~10% 3건, 크게 어긋남 3건 (opt 128/1024, tinyllama 128/1024, t5-base 128/1024)
- 15건 모두 OOM 예측 없음 (GPU도 measure ok) → OOM 판정 일치
- allocated는 15건 모두 CPU가 GPU보다 6~10 MiB 작음
- 크게 어긋난 3건 원인 후보: GPU 전용 할당, 요청 순서의 작은 차이, 캐시 반환 시점 민감도 (미확인)
- 용량 근처 reserved는 반환 시점에 민감 → OOM 판정을 1차 목표로 두기로 정렬표에 기록

### 4.6 스레드 확인 (§2.17)
- trace 완전 동일 2/12, 예측 summary 동일 12/12 (작은 4조건 한정)

---

## 5. 결과·로그 파일 위치

| 구분 | 경로 (저장소 기준, 절대경로 `/home/inuos/cgf-fp16-680-20261111/…`) | 내용 |
|---|---|---|
| 설명 | `README.md` | 현재 상태(L256), 구조, 재현 명령, 알려진 한계 (일부 v1 기준 서술 남음) |
| 조건표 | `manifest/cases.jsonl`, `.csv` | 680 (v1) |
| | `manifest/conditions_main395.json`, `cases395.jsonl` | L1024 395 |
| | `manifest/conditions_main395_l256.json`, `cases395_l256.jsonl` | L256 395 |
| | `manifest/smoke_cases.jsonl`, `fp32_calib_cases.jsonl`, `bf16_calib_cases.jsonl`, `pilot.json` | smoke 5, 보정 26×2, 파일럿 id |
| 모델 | `manifest/models.json`, `model_pins.json` | revision |
| GPU 용량 | `manifest/gpu_profiles.json` | rtx3090 용량 |
| xMem 잠금 | `provenance/xmem-allocator-lock.json` | 커밋·파일 sha256 |
| v1 실행 | `runs/{smoke,smoke-gpt2-rowmajor,smoke-gpt2-colmajor,pilot-fullscope-ref,verify-userscope,pilot,fp32-calib,smoke-emu,emu-validate,emu-overhead}/` | `attempts.jsonl`, `suite_args.json`, `cases/<cid>/` |
| v1 로그 | `runs/pilot.console.log`, `runs/fp32-calib.console.log`, `runs/pilot-fullscope-ref/console.log`, `runs/download*.log` | |
| v1 결과 | `results/fp32-calib/{cpu_results.csv,jsonl,summary.json,eta.json}` | |
| 로드 합성 검증 | `results/load_emu_validation.jsonl` | 13행 |
| v2 smoke | `runs/smoke2/<cid>/` | 3건 |
| main395 | `runs/main395/{attempts.jsonl,suite_invocations.jsonl,cases/}` (389 폴더, 결과 388) | 원본 유지 |
| main395 로그 | `runs/main395.pilot.console.log`, `runs/main395.main.console.log` | |
| main395 결과 | `results/main395/{cpu_predictions.csv,jsonl,status_summary.json,pilot_vs_rtx3090.csv,pilot_vs_rtx3090.after_pilot.csv,gpu_trace_request_diff.jsonl,collect_final_20261115.log,pilot_collect.log}` | |
| 보관 | `archive/main395_cpu_L1024_stopped_20261115.tar.gz` (+ `.sha256`) | 미커밋 |
| 정렬 | `handoff_codex/gpu_cpu_alignment.md` | 판정표·용량 재생 결과 |
| | `handoff_codex/alignment_20261115/{compare_l1024.py,.out,_rows.json,replay_capacity.py,replay_capacity_24979374080.jsonl,replay_capacity.err}` | |
| | `handoff_codex/cpu_manifest.json` (+ `.sha256` `57f4cdd6…`) | 11-12 CPU 감사 매니페스트 |
| | `handoff_codex/INUGPU_CNN_TRACE_REQUEST_KO.md` | CNN trace 요청서 |
| L256 | `runs/main395_l256/{reuse_manifest.json,cases/}` (심볼릭 링크 244) | 151개 미실행 |
| | `results/main395_l256/{cpu_predictions.csv,jsonl,status_summary.json,pilot_vs_rtx3090.csv,selfcheck_1TiB.jsonl}` | 미커밋 |
| 스레드 확인 | `runs/thread_check/t{20,7,5,1}/`, `results/thread_check/{compare.jsonl,run.log,tdiff.py}` | |
| GPU handoff 11-12 | `/home/inuos/cgf-fp16-inuos-handoff-20261112/` (`INUOS_GPU_MEASUREMENT_PROTOCOL_KO.md`, `src/vram_worker.py`, `config/`, `reference/{pilot_rtx3090_eager_vs_default.csv,replay_table.md,traces_attn_default/,attention_kernels_rtx3090_default.json,REPORT_…md}`) | GPU 프로토콜·파일럿 reference·trace 6 |
| GPU handoff 11-15 | `/home/inuos/cgf-fp16-inuos-handoff-20261115/` (`INUOS_GPU_EXPERIMENT_SETUP_KO.md`, `config/`, `src/`, `results/rtx3090_main395_L1024/{gpu_rtx3090_main395_L1024_per_condition.csv,summary.csv,runs_rtx3090__main395/<cid>/{measure,validate}.json}`) | GPU L1024 결과, L256 조건표 |

---

## 6. 한계 / 미확인

### 6.1 방법상 구조적 한계
- FP32 실행 + shadow 태그 환산
  - 네이티브 FP16 실행과 요청 순서가 같다는 증명 없음 (v1 비교에서 완전 일치 0/11)
  - 미매칭 요청은 이웃 태그 상속(`inferred_*`): 추론일 뿐 GPU dtype 증명 아님
  - CPU kernel 내부 임시 버퍼는 CPU 고유. GPU kernel 임시 버퍼(cuBLAS/cuBLASLt/cuDNN workspace)는 CPU trace에 없음
  - T5 일부 요청이 정확히 2배(8 KB↔16 KB, 1 MiB↔2 MiB): `f` 태그 과다 가능성 (미확인)
  - FP16 출력 finite 여부는 판단 불가 (GPU validate 결과 사용)
- attention: 이름은 같지만 커널이 다름 (GPU flash/mem-efficient, CPU `_scaled_dot_product_flash_attention_for_cpu`). allocated 일치로 L×L 임시 텐서 과대 예측은 없다고 판단 (정렬표)
- CNN(resnet-18/50, mobilenet-v2) 오차가 큼 (allocated 중앙값 −16% / −9% / +28%). 원인 미확정. GPU CNN trace 회신 대기
- 용량: 생산 worker 값은 1 TiB 재생. 용량 근처 조건은 후처리 `replay_capacity.py` 필요하며, 용량 재생에서도 3/15건 크게 어긋남
- 비교 대상은 GPU measure_0만. GPU measure 1~4, NVML, allocator 밖 사용량은 예측하지 않음
- CPU RAM 한도(42g, FP32라 FP16 대비 2배 이상)로 큰 조건은 예측 자체 실패 → GPU OOM과 구분해 `cpu_not_predictable_ram`으로 분류 예정(`compare_gpu_l256.py`)
- torch 스레드: GPU 4, CPU 7/20 등. 작은 4조건에서 trace는 다르지만 summary는 같음. 큰 조건 영향 미확인

### 6.2 검증 기록의 공백
- 단일 trace ×4 동일성("36/36"), Conv1D·USER_SCOPE 동일성의 비교 도구 출력이 파일로 남아 있지 않음 (결과 JSON의 요청 수·예측값으로만 재확인 가능, 이 문서 §2.2·2.3에서 확인)
- Conv1D col-major의 파일럿 크기 검증(`verify-layout-rowmajor`) 미실행. v2에서는 사용 안 함
- BF16 보정 미실행
- `compare_l1024.out`의 "input sha same 0"은 파싱 문제. 전체 해시 375/375 일치는 이번 조사에서 재계산으로 확인
- 정렬표 §3의 "C단계 재시도 6개" 서술과 최종 timeout 7건이 다름 (C단계는 실행되지 않음)
- `README.md`의 "CPU 프로토콜" 절은 v1(eager, 42g, `--cpus 20`, 4회 실제 추적) 서술이 남아 있어 현재 v2와 다름
- `models.json` 공통 설명의 `attn_implementation: "eager"`는 실제 GPU/CPU v2 실행(기본값)과 다름

### 6.3 진행 상태 (조사 시점)
- main395(L1024): 사용자 중단, 결과 없음 7, C단계 미실행
- L256: 실행 대상 151개 **미실행** (`runs/main395_l256`에 시도 기록 없음, 해당 tmux 없음)
- GPU L256 결과 미도착 → L256 비교(`compare_gpu_l256.py`), GPU L1024↔L256 겹침 동일성(`check_gpu_overlap.py`) 미실행
- 본 용량 재생 `results/main395_l256/capacity_rtx3090.jsonl` 미생성 (self-check 5건만)
- tmux `cgf-threadcheck` 세션 존속 (스레드 확인은 18:15에 출력 완료)
- AWS GPU 결과: GPU handoff 문서상 미확인
- 정렬표 §5 수정 계획은 "사용자 확인 전"으로 적혀 있으나, 이후 커밋에서 조건표 교체·용량 재생 도구·스레드 확인은 구현됨. 사용자 승인 기록은 저장소에 없음

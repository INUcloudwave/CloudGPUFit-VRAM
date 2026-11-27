# GPU 수집 요청 — CloudGPUFit reserved 오차 원인 분리 (inugpu용)

- 작성: inuos (CPU VM), 2026-11-10
- 함께 전달: `common_manifest.json`, `gpu_collector_reference.py`, `xmem-allocator-lock.json`
- 경로 주의: 이 문서의 inuos 경로는 inugpu에 없다고 가정. 자기 디렉터리 구조에 맞춰 배치

## 1. 목적

- CPU 예측(xMem allocator 시뮬레이션)의 reserved 피크가 GPU 실측보다 크게 작음
  - 예: 이전 L4 기준 gpt2 FP16 b4/768/128 → CPU 928MiB, GPU 1678MiB
- 원인 후보를 분리하려면 **GPU에서 실제 PyTorch allocator가 받은 요청과 내린 결정 전체**가 필요
  - (a) 모델 로딩·검증·워밍업·반복 실행 이력(초기 캐시)
  - (b) xMem과 PyTorch 2.6 allocator 정책 차이
  - (c) CPU trace에 없는 GPU 전용 요청(cuBLAS/cuBLASLt workspace 등), 해제 시점
- inuos는 받은 trace의 요청을 그대로 xMem에 재생해 GPU segment 결정과 요청 단위로 비교
- NVML 값은 보조 관측. reserved 정답으로 쓰지 않음

## 2. 환경 조건

| 항목 | 요구 | 비고 |
|---|---|---|
| torch | 2.6.0+cu124 | 다르면 그대로 진행하되 run_meta에 기록, 결과는 별도 버전군으로 취급 |
| transformers | 4.52.4 | |
| python | 3.11.11 | |
| `CUDA_MODULE_LOADING` | **LAZY를 torch import 전에 명시** | 실제 값 기록. 가능하면 `cuModuleGetLoadingMode` 조회값도 기록 (참조 수집기에 구현됨) |
| `PYTORCH_CUDA_ALLOC_CONF` | 설정하지 않음 | 실제 값 기록 (expandable_segments 등 금지) |
| `CUBLAS_WORKSPACE_CONFIG`, `CUBLASLT_WORKSPACE_SIZE` | 설정하지 않음 | 실제 값 기록 |
| attention | `attn_implementation="eager"` | CUDA module loading의 EAGER와 무관한 설정 |
| TF32 / cudnn.benchmark | False / False | |
| seed | `torch.manual_seed(20261006)` | |
| GPU 점유 | 단일 GPU, 다른 GPU 프로세스 없음 | |
| 프로세스 | **(case, run_id)마다 새 프로세스** | |

- 이전 AWS 실측의 LAZY 여부는 기록이 없어 확인되지 않음. 이번 결과로 소급 확정하지 않음
- 새 GPU가 L4가 아니면 이전 L4 숫자는 참고치일 뿐, 정답값 아님

## 3. Case (모델·입력은 `common_manifest.json`이 기준)

| case_id | 모델 / revision | dtype | batch | 입력 | 생성 |
|---|---|---|---|---|---|
| core__decoder_llm__small__fp16__01 | openai-community/gpt2 @607a30d | FP16 | 1 | 256 | 64 |
| core__decoder_llm__small__fp16__12 | openai-community/gpt2 @607a30d | FP16 | 4 | 768 | 128 |
| core__decoder_llm__medium__fp32__12 | openai-community/gpt2-medium @6dcaa7a | FP32 | 4 | 768 | 128 |

- 모델 로딩: `AutoModelForCausalLM.from_pretrained(model, revision, torch_dtype, low_cpu_mem_usage=True, device_map={"":"cuda:0"}, use_safetensors=True, local_files_only=True, attn_implementation="eager").to(device="cuda", dtype=dtype).eval()`
- 입력: manifest의 recipe 그대로 생성 후 `input_ids` sha256이 manifest 기대값과 같은지 확인
  - 기대값: fp16_01 `d16e737f0b…`, 나머지 두 case `e64f48260f…` (전체 값은 manifest)
  - 입력은 CPU에 두고 매 호출마다 `.to("cuda")` (이전 worker와 동일)
- generate: greedy, `min_new_tokens = max_new_tokens`, `use_cache=True`. 실제 생성 토큰 수를 매 호출 기록, 다르면 실패 처리

## 4. 실행 구성 (각각 fresh process)

| run_id | 그룹 | 단계 (phase × 횟수) | 목적 |
|---|---|---|---|
| R0_aws_replica | 기준 | model_load → performance/finite_check×1 → performance/warmup×1 → performance/measure×5 → memory/between_modes×1 → memory/finite_check×1 → memory/warmup×1 → memory/measure×3 | 이전 AWS 구조 재현 (bench_worker_v3, profile local_core, hybrid, 캐시 유지) |
| R1_controlled | 기준 | model_load → controlled/finite_check×1 → controlled/warmup×1 → controlled/measure×3 | CPU와 맞출 통제 실행 (R0 앞부분과 동일 순서) |
| R2_cold_single | 기준 | model_load → single/measure×1 | 로딩 이력과 반복 요청 이력 분리 |
| D1_cold_single_empty_cache | **진단 전용** | model_load → empty_cache → single/measure×1 | CPU 기준선 초기 상태에 가장 가까운 GPU 조건. 원 조건과 동일하다고 부르지 않음 |

- 3 case × 4 run = 12 프로세스
- `between_modes`: synchronize → `gc.collect()` → **empty_cache 호출 없음** → config/generation_config 복원, CPU/CUDA RNG 복원
- `reset_peak_memory_stats()`는 피크 통계 초기화일 뿐 캐시 해제 아님. measure 블록 직전에만 호출 (이전 worker와 동일)
- 이전 worker의 tokens/components 모드는 memory 이후라 memory 값에 영향 없음 → 생략
- NVML 샘플러 스레드·idle 대기는 allocator를 쓰지 않으므로 생략 가능 (생략 시 run_meta에 기록)

## 5. 필수 기록

### 5.1 원본 allocator 이력
- **모델 로딩 전**(첫 CUDA 할당 전)에 `torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="python", max_entries=충분히 큰 값)`
- 종료 시 `torch.cuda.memory._snapshot()`을 **그대로 pickle 저장** (`memory_snapshot.pickle`). 가공본으로 대체 금지
- 기록 시작 시 `reserved_bytes.all.current == 0`, 누적 할당 카운터 0 확인 → 초기 상태가 빈 allocator임을 증명
- 종료 시 trace의 alloc 이벤트 수 == `allocation.all.allocated` 증가분 확인 (ring buffer 잘림 검출)
- 선택: medium_fp32_12의 R1만 `stacks="all"`(C++ 포함)로 1회 추가 → workspace 블록 출처 식별용

### 5.2 단계 경계 (`boundaries.jsonl`, 한 줄 = 한 경계)
- 각 phase 호출 전/후(`phase_begin`/`phase_end`), `record_start`/`record_end`, empty_cache 직전(`empty_cache_before`)
- 각 경계에서 `torch.cuda.synchronize()` 후 기록:
  - `counters`: `allocation.all.allocated`, `allocation.all.freed`, `segment.all.allocated`, `segment.all.freed` (inuos가 이 값으로 trace 내 경계 위치를 정확히 찾음)
  - `memory_stats` 일부: allocated/reserved/active/inactive_split/requested의 current·peak, segment 수(all/large/small), `num_alloc_retries`, `num_ooms`, `num_device_alloc/free`
  - `time_us` (보조 확인용)
- 포함 관계 값(예: active ⊇ allocated)을 합산하지 말 것
- phase 끝마다 `torch.cuda.memory_snapshot()`(segment 상태) 저장 권장

### 5.3 메타데이터 (`run_meta.json`)
- torch/CUDA/cudnn/transformers/python/kernel 버전, GPU 이름·UUID·총용량, driver
- allocator backend, 위 환경변수 실제값, module loading 조회값, attention 구현, TF32 설정
- 모델 revision, weights 출처(로컬 HF cache 경로가 아닌 repo+revision), dtype, 입력 sha, 생성 설정
- 수집 스크립트 sha256, manifest sha256, 실행 명령, 상태(ok / gpu_oom / error)

### 5.4 호출 결과 (`invocations.jsonl`)
- phase, invocation, 실제 생성 토큰 수, 출력 token id sha256, 소요시간, finite 여부

## 6. 이벤트 의미 (PyTorch v2.6.0 원문 확인 결과)

- `alloc`, `free_requested`, `free_completed`의 `size` = **원요청 크기**(`orig_size` / `requested_size`)
- `segment_alloc`, `segment_free`의 `size` = segment 크기
- `addr`는 재사용됨 → inuos가 세대 ID로 구분. stream 값도 그대로 보존
- 지원되지 않는 필드는 만들지 말고 비워 둘 것 (unknown)

## 7. 하지 말 것

- 모델/입력/생성 설정을 맞추기 위한 임의 보정이나 allocator 설정 변경
- empty_cache를 기준 run(R0/R1/R2)에 섞는 것
- NVML 값으로 reserved 역산
- trace 중간부터 잘라 전달
- 실패 run 삭제 (실패도 로그와 함께 반환)

## 8. 반환 묶음

```
<return_root>/
  common_manifest.json            (받은 그대로)
  collector/                      (실제 사용한 수집 코드 + sha256)
  <case_id>/<run_id>/
    run_meta.json
    boundaries.jsonl
    invocations.jsonl
    memory_snapshot.pickle        (원본)
    segments_*.pickle.gz          (선택)
    collector.log, stdout/stderr
  SHA256SUMS
```
- 모델 weights·대형 tensor는 포함하지 말 것
- pickle은 inuos에서 전역 객체 로드를 거부하는 안전 로더로만 읽음

## 9. 참조 수집기 `gpu_collector_reference.py`

- 위 계약을 구현한 참조 코드. **GPU에서 실행해 본 적 없음** (CPU VM에서 문법 검사만 완료)
- 사용 예: `CUDA_MODULE_LOADING=LAZY python gpu_collector_reference.py --manifest common_manifest.json --case <case_id> --run-id R1_controlled --out <dir>`
- 이전 worker(bench_worker_v3.py, sha256 `4018cd8e…`)와 호출 순서를 맞췄음. 차이가 있으면 run_meta에 남기고 알려 줄 것
- 실행 전 1회 점검 권장: fp16_01 R2로 실행 → `record_start` reserved 0, alloc 수 일치, 생성 토큰 수 64 확인

## 10. inuos가 자료로 할 일 (참고)

- 각 run의 GPU 요청을 xMem에 순서대로 재생 → 요청별 새 segment 여부·reserved를 실제와 비교 → 최초 불일치 지점 보고
- 같은 phase의 CPU trace와 요청 크기·순서·개수 대조 → GPU 전용 요청(workspace 후보) 식별
- GPU 로딩 구간 + CPU 추론 요청을 이어 붙인 hybrid 재생으로 로딩 이력의 기여분 분리

> 레포 경로 대응: VM `/home/inuos/cgf-reserved-diag-20261110/` → 레포 `prediction/local-cpu-vm/03-reserved-diag/`. `runs/`(원시 trace 약 1GB), `inputs/`(GPU 반환 원본), `*.gz`·`*.pickle`은 레포에서 제외. 대응 GPU 쪽: inugpu `04-reserved-diag/`.

- 루트: `/home/inuos/cgf-reserved-diag-20261110` (git 브랜치 `diag/reserved-error-20261110`, 원격 없음, 최신 커밋 `a002c9a`)
- 작성 시점: 2026-11-15 (실험은 11-10 ~ 11-11에 종료)

---

## 1. 개요

### 1.1 목적
- 이전 CPU 예측(342조건, 2026-11-08)의 reserved가 AWS L4 실측보다 크게 작았던 이유를 원인별로 분리
  - 예: gpt2 FP16 b4/768/128 → CPU 928 MiB, L4 1678 MiB
- 원인 후보 3가지를 각각 검증
  - Q1 실행 이력: 초기 캐시 상태와 앞 단계 실행 이력이 reserved를 얼마나 바꾸는가
  - Q2 allocator 정책: 같은 요청을 넣었을 때 xMem과 PyTorch 2.6 allocator의 결정이 같은가
  - Q3 잔여 차이: 이력을 맞춘 뒤 남는 차이가 GPU 전용 요청(workspace 등), 중간 텐서, 해제 시점 중 무엇에서 오는가
- 제약: 기존 production 예측 로직은 수정하지 않고, 원인 확인 전에는 보정식도 넣지 않음

### 1.2 대상 case (이전 342조건 표 기준 case_id)

| 별칭 | case_id | 모델 | dtype | B / L / N |
|---|---|---|---|---|
| fp16_01 | `core__decoder_llm__small__fp16__01` | gpt2 | fp16 | 1 / 256 / 64 |
| fp16_12 | `core__decoder_llm__small__fp16__12` | gpt2 | fp16 | 4 / 768 / 128 |
| medium | `core__decoder_llm__medium__fp32__12` | gpt2-medium | fp32 | 4 / 768 / 128 |

### 1.3 환경
- CPU: Intel Xeon Cascade Lake 24코어, RAM 47 GiB, swap 없음
- 실행 이미지: `cloudgpufit-cpu-validation:20261108` (torch 2.6.0+cu124, transformers 4.52.4, python 3.11.11)
- xMem: `/home/inuos/xmem-work/xMem` @ `1ee4bd08` (lock 파일로 8개 파일 sha256 확인, 무수정)
- GPU 측: INUGPU RTX 3090 24GB, driver 570.211.01, torch 2.6.0+cu124, `CUDA_MODULE_LOADING=LAZY` 확인

---

## 2. 실험 목록 (시간순)

### 2.1 환경·경로 조사, 기준선 재현 (11-10)
- 실제 CPU 예측 경로 확인: `predict_vram.py` → `cpu_tensor_trace.trace_and_estimate` → `inference_trace_runtime.simulate` → xMem `AllocatorSim`
  - predictor 소스: `/home/inuos/CloudGPUFit-cpu-singlepass-20261108` (7f822d7 + 미커밋 변경. 결과 manifest identity와 sha256 일치)
- 이전 CPU 예측의 특징
  - 빈 allocator + 모델·입력 storage를 합성 배치 → generate **1회**만 추적, warmup은 추적 밖
  - 용량 80 GiB, `from_config` 랜덤 가중치
- 이전 L4 측정의 특징 (`bench_worker_v3.py`, profile `local_core`)
  - 한 프로세스에서 모델 로딩 → performance 모드(검증 forward 1 → warmup 1 → measure 5) → `gc.collect()`만 → memory 모드(검증 forward → warmup → measure 3)
  - 캐시 유지(`keep_allocator_cache=True`), `empty_cache()` 없음
- 기준선 결정성: 3 case 모두 다시 추적해도 저장된 예측과 동일 (`runs/baseline/<case>/`)
- xMem 정책 후보 점검 (fp16_01): 같은 timestamp 이벤트 0건, 정확히 10 MiB 요청 0건 → 두 후보 영향 0

### 2.2 GPU 수집 요청 → INUGPU (11-10)
- `handoff/GPU_COLLECTION_REQUEST.md`, `handoff/gpu_collector_reference.py`
- 실행 구성 (case마다 새 프로세스, `torch.cuda.memory._record_memory_history` + snapshot, 카운터 기반 경계)

| run | 내용 |
|---|---|
| R0_aws_replica | 이전 AWS worker 구조 그대로 (performance → memory) |
| R1_controlled | 검증 forward 1 → warmup 1 → measure 3 |
| R2_cold_single | 로딩 → generate 1회 |
| D1_cold_single_empty_cache | 로딩 → `empty_cache()` → generate 1회 |

- 반환: 2026-11-11, RTX 3090, 모든 run이 빈 allocator에서 시작, 잘림 없음, stream 0만 사용

### 2.3 CPU 이력 추적 (11-10 ~ 11-11)
- 도구: `src/cgf_diag/cpu_history_trace.py`. GPU worker의 13단계를 CPU에서 그대로 실행
  - sequence `aws_replica_local_core`: performance(검증 1, warmup 1, measure 5) → between_modes(gc) → memory(검증 1, warmup 1, measure 3)
- 입력: GPU와 같은 방식(`torch.Generator().manual_seed(20261006)`, `randint(3, vocab)`, special→3) → input_ids sha256 GPU와 일치
- **문제와 수정**: medium 추적이 OOM(exit 137)으로 반복 사망
  - 원인: 같은 프로세스에서 두 번째 `torch.profiler` 세션이 decode step마다 RSS를 약 0.6 GB씩 늘림
    - 진단(`diag_oom.py`): profiler 없음 → 정상, memory/nomemory 프로파일·gc on/off → 모두 OOM
  - 수정: **단계마다 새 프로세스**(`--step-index`)로 추적하고 `merge_steps.py`로 병합
    - 동일성 검증: fp16_01 단일 프로세스 추적과 13단계 모두 요청 크기·순서 동일
- 진행: fp16_01·medium·fp16_12 모두 13/13 단계 완료(11-11 12:44), 분석·요약 완료 (`results/SUMMARY_cpu.md`)

### 2.4 GPU 반환 데이터 xMem 재생 (11-11)
- 도구: `src/cgf_diag/gpu_replay.py` (snapshot 안전 로드 → 정규화 → 순서대로 xMem 재생 → 결정 단위 비교)
- 스크립트: `docs/run_gpu_import_phase1.sh` (tmux, 전 run exit 0)

### 2.5 L4 325조건 모집단 확인 (11-11)
- `results/population_l4_vs_cpu/pop2~4.py`: 이전 342조건 CPU 예측과 L4 실측의 325개 공통 성공 조건 비교
- 실행 이력 가설이 3 case 밖에서도 성립하는지 확인

### 2.6 재설계 프롬프트 (11-11)
- `handoff/REDESIGN_EXPERIMENT_PROMPT.md`: 장치 외 차이를 없앤 재실험 설계를 다른 AI에게 맡기기 위한 프롬프트
- 이후 INUGPU가 VRAM 전용 측정(warmup 3 → measure 5, 검증 forward·performance 모드 제거)으로 재설계 → FP16 실험(별도 페이지)

---

## 3. 코드 설명 (`src/cgf_diag/`)

### 3.1 `events.py` (253줄) — 공통 이벤트 스키마
| 줄 | 내용 |
|---|---|
| 33-55 | gz/평문 jsonl 읽기·쓰기 |
| 57-91 | `Pairer`: 주소 재사용을 고려한 alloc/free 짝짓기(세대 id), 짝 없는 free 집계 |
| 93-113 | `cpu_memory_events_from_chrome`: chrome trace의 CPU `[memory]` 이벤트(Bytes, Addr, Ev Idx) 추출, 구간(record_function) 이름 수집 |
| 115-125 | `order_report`: Ev Idx 누락, ts 역전, 같은 ts 이벤트 수 |
| 127-170 | `normalize_cpu`: 구간 필터, phase/invocation 부여, 정규 스키마 `{seq, src, action, addr, size, stream, t, order_key, alloc_id, phase, invocation}` |
| 172-185 | `_SafeUnpickler`/`load_snapshot`: 전역 객체를 거부하는 안전 unpickle |
| 187-241 | `normalize_gpu_snapshot`: memory history → 정규 스키마, 누적 카운터(allocation/segment allocated·freed)로 경계 위치 결정 |
| 243-253 | `gpu_reserved_timeline`: segment 이벤트로 reserved 시계열 |

### 3.2 `replay.py` (244줄) — xMem 순서 재생기
| 줄 | 내용 |
|---|---|
| 23-38 | `verify_xmem`: lock 파일과 xMem 파일 sha256 대조 |
| 40-66 | xMem 클래스 로드. 진단 정책 `PT26AllocSizeBoundary`(PyTorch 2.6 `size < 10MiB` 경계, xMem은 `<=`) |
| 69-200 | `Replayer(capacity_bytes, policy)`: alloc/free/empty_cache를 순서대로 적용, 결정 기록 `(seq, alloc_id, request, block, new_segment, released_segment, reserved_after, allocated_after, segment_id)`, phase별 통계, 용량 초과 시 xMem의 캐시 반환 → 재시도 → OOM 기록 |
| 173-178 | `place_resident`: 기준선과 같은 상주 storage 합성 배치 |
| 203-231 | `time_grouped_xmem`: production 경로(`inference_trace_runtime.simulate`, 같은 ts 안에서 할당 먼저) 그대로 재현 |
| 233-244 | `timestamp_collision_report` |

### 3.3 `cpu_history_trace.py` (257줄) — CPU 이력 추적
| 줄 | 내용 |
|---|---|
| 37-45 | `SEQUENCES`: `aws_replica_local_core`(13단계), `controlled_f1_w1_m3`, `single_generate` |
| 51-59 | `gpu_worker_inputs`: GPU worker와 같은 입력 생성 |
| 76-87 | CLI: `--input-json`, `--sequence`, `--weights from_config|pretrained`, `--out`, `--threads 4`, `--keep-raw-traces`, `--expected-input-sha`, `--step-index`, `--mem-log` |
| 106-107 | `--step-index K` → `<out>/steps/KK/`에 단계 K만 기록 |
| 139 | 자동 GC 비활성 (GPU worker의 gc 지점만 재현) |
| 142-172 | `step_fn`: finite_check(입력 복제 forward + isfinite), generate(GPU와 같은 kwargs, 출력 `.cpu()`, 길이 확인), between_modes(gc.collect) |
| 174-230 | 단계마다 profiler 창 1개(`profile_memory=True`) → 이벤트 정규화. 창 밖에서 live 요청이 없을 때만 `gc.collect()` + `malloc_trim` |
| 232-257 | meta(입력 sha, 상주 크기, 단계 목록, RSS, process_policy) 저장 |

### 3.4 `merge_steps.py` (106줄)
- `steps/*` 병합: 설정 일치 검사(sequence/steps/입력 sha/상주 크기/모델 config sha/가중치) → seq 재번호, `alloc_id = k·1e9 + id`
- `--compare-with`: 단일 프로세스 추적과 단계별 요청 크기·alloc/free 순서 동일성 → `equivalence_vs_single_process.json`

### 3.5 `analyze_cpu.py` (172줄)
- 기준선 A1(production 시간 그룹 재현, 저장값과 일치해야 함), A2(순차 재생), A3(10 MiB 경계 진단)
- 이력: 단계별 요청 동일성, 누적 prefix 재생, 첫 generate 단독, 모드별 측정 구간 피크

### 3.6 `gpu_replay.py` (295줄) — GPU 반환 데이터 재생·비교
| 줄 | 내용 |
|---|---|
| 25-51 | 경계 읽기, `empty_cache` 행 삽입 (marker 자신의 phase 사용) |
| 53-78 | `observed_decisions`: GPU의 실제 segment 결정 복원 (empty_cache 뒤 segment_free는 호출에 귀속) |
| 80-115 | `compare`: xMem 결정과 비교, 최초 불일치와 앞뒤 문맥 |
| 117-139 | `phase_table`: 구간별 GPU/xMem reserved |
| 141-168 | `request_diff`: GPU와 CPU 요청 크기 multiset 차이 (GPU 전용 / CPU 전용) |
| 170-185 | `hybrid_rows`: GPU 로딩 구간 + CPU 요청 |
| 187-219 | `select_cpu_rows`, `CPU_SELECT_PRESETS` (run별로 대응하는 CPU 단계 선택) |
| 221-295 | CLI(`--run-dir`, `--out`, `--capacity-gb 80`, `--cpu-history`, `--gpu-prefix-phases`) → `import_report.json`, `decisions_*.csv.gz` |

### 3.7 기타
- `build_manifest.py`: `manifest/common_manifest.json` (case, run, 설정, L4 참고값)
- `summarize.py`: `results/SUMMARY_cpu.md` 생성
- `diag_oom.py`: profiler 유무·gc on/off별 N회 generate RSS 기록 (OOM 원인 진단)
- `tests/test_replayer_synthetic.py`: 합성 테스트 8개 (크기 경계, 재사용, split/merge, 용량 반환·재시도·OOM, 같은 ts 재정렬, stream, GPU importer·안전 pickle, 최초 불일치) — 전부 통과
- `docs/run_in_image.sh`: 고정 이미지 docker 래퍼 (xmem·predictor ro 마운트, `/tmp` tmpfs, 네트워크 차단)
- `docs/run_cpu_stage_per_step.sh`: 현재 CPU 단계 스크립트 (단계별 프로세스 → 병합 → 분석 → 요약)
- `docs/run_gpu_import_phase1.sh`: GPU run별 import

---

## 4. 결과

### 4.1 같은 GPU 요청을 xMem에 재생 (Q2: allocator 정책)
- reserved 피크 (MiB, GPU 실측 / xMem 재생)

| run | fp16_01 | fp16_12 | medium |
|---|---|---|---|
| 모델 로딩 직후 | 474 / 474 | 474 / 474 | 1380 / 1380 |
| D1 (로딩 → empty_cache → generate 1) | 344 / 344 (구간) | 934 / 932 | 3572 / 3572 |
| R2 (로딩 → generate 1) | 492 / 492 | 920 / 918 | 3572 / 3572 |
| R1 검증 forward까지 | 478 / 478 | 1672 / 1672 | 4570 / 4570 |
| R1 전체, R0 전체 | 492 / 492 | 1678 / 1676 | 4574 / 4574 |

- 결정 불일치: medium 1.15M 요청 중 0건, fp16_12 1건, fp16_01 6건 (1 MiB 요청, 피크 영향 없음)
- 결론: allocator 정책 차이는 오차 원인이 아님

### 4.2 CPU 이력 재생 (Q1: 실행 이력)

| case | CPU 기존 예측 (generate 1회) | CPU 13단계 이력 재생 | L4 실측 |
|---|---|---|---|
| fp16_01 | 330 | 398 (상주 합성, 로딩 캐시 없음) / hybrid(GPU 로딩 + CPU 요청) 490 | 492 |
| fp16_12 | 928 | 1668 (검증 forward 후 1664) | 1678 |
| medium | 3552 | 4574 (검증 forward 후 4568) | 4574 |

- 큰 증가는 **출력 검증 forward**(`model(**inputs)`, 전체 시퀀스 logits) 1회에서 발생
- fp16_01은 모델 로딩 캐시(474)가 주원인 → CPU에서 관측할 수 없음

### 4.3 L4 325조건 모집단 (`results/population_l4_vs_cpu/`)
- L4 325조건 전부: memory 모드 측정 피크 = 측정 시작 직전 reserved (측정 전 단계에서 이미 결정)
- 실측 ≈ max(CPU 예측, 로딩 직후 캐시, 검증 forward 피크(first_logits 모드))

| 분류 | n | 5% 이내: CPU만 → max | 20% 초과: CPU만 → max |
|---|---|---|---|
| decoder LLM | 115 | 37 → 104 | 34 → 0 |
| encoder-decoder | 129 | 21 → 115 | 53 → 0 |
| vision | 81 | 29 → 36 | 18 → 18 |

- vision은 CPU 과대 예측 + allocated 자체가 크게 다름 → 이력과 별개 원인
- decoder LLM bf16 12건: allocated를 CPU가 28~82% 과대 예측 (캐시에 가려져 있었음)

### 4.4 요청 구성 차이 (Q3, fp16_01 generate 1회)
- GPU 29,764 요청 vs CPU 33,091 요청
- GPU 전용: 1 MiB × 3,072 (cuBLASLt, Lt GEMM마다), 4 B × 3,150, 50,257 B × 256, 8,519,680 B 1회(cuBLAS workspace, 첫 forward)
- CPU 전용: 2 B × 6,286, 8 B × 3,586, 512 B × 51
- generate 1회 기준 GPU−CPU reserved 차이: D1 기준 +6~+20 MiB (원인 정량화는 미완)

---

## 5. 결과·로그 파일 위치

| 내용 | 경로 |
|---|---|
| 환경·경로 보고 | `provenance/ENV_AND_PATH_REPORT.md`, `provenance/pytorch_v2.6.0_ref/`, `provenance/xmem-allocator-lock.json` |
| 공통 manifest | `manifest/common_manifest.json` |
| GPU 요청서·수집 코드 | `handoff/` |
| 기준선 재현 | `runs/baseline/<case>/` (레포 제외) |
| CPU 이력 trace | `runs/cpu_history/<case>/from_config_per_step/` (레포 제외) |
| CPU 분석 | `results/cpu/<case>.analysis.json`, `results/SUMMARY_cpu.md` |
| GPU 재생 | `results/gpu/<case>/<run>/import_report.json` |
| 모집단 분석 | `results/population_l4_vs_cpu/` |
| 합성 테스트 | `results/synthetic/synthetic_validation.json` |
| 재설계 프롬프트 | `handoff/REDESIGN_EXPERIMENT_PROMPT.md` |

---

## 6. 한계 / 미확인
- 3 case 모두 gpt2 계열 decoder. encoder-decoder의 로딩 캐시 원인, vision 요청 차이는 요청 단위로 검증하지 않음
- 모집단 분석은 L4 요약값 기반 정황 (요청 단위 아님)
- GPU 재생 2단계(CPU 이력과 요청 diff·hybrid를 fp16_12·medium에 적용)는 실행하지 않음 → FP16 재실험으로 넘어감
- 이전 AWS 실행의 `CUDA_MODULE_LOADING`, `PYTORCH_CUDA_ALLOC_CONF`는 기록 없음

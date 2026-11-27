# 환경·실행 경로 보고 (inuos, 2026-11-10)

## 1. 실험 위치 (기존 코드·결과 미수정)
- 실험 루트: `/home/inuos/cgf-reserved-diag-20261110` (별도 git 저장소, 브랜치 `diag/reserved-error-20261110`)
- 기존 자료는 읽기 전용 mount 또는 복사본으로만 사용

## 2. 실제 CPU 예측 실행 경로 (2026-11-08 342조건 결과 기준)
| 구성 | 실제 위치 / 식별자 | 확인 방법 |
|---|---|---|
| predictor 소스 | `/home/inuos/CloudGPUFit-cpu-singlepass-20261108` (7f822d7 + 미커밋 변경 9파일, detached HEAD) | 5개 파일 sha256이 run manifest `identity`와 일치 |
| 진입점 | `predict_vram.py` → `cpu_tensor_trace.trace_and_estimate` → `inference_trace_runtime.simulate` → xMem `AllocatorSim` | 소스 정독 |
| xMem | `/home/inuos/xmem-work/xMem` @1ee4bd08… | lock 8파일 sha256 일치, import 경로 `/work/xMem/perf_estimator/...` 확인 |
| xMem lock 밖 의존 | `perf_estimator/data_structure`, `utilis` (allocator가 import) | git 상태 무수정 확인, 해시는 `xmem_unlocked_dependencies.sha256` |
| 실행 이미지 | `cloudgpufit-cpu-validation:20261108` (sha256:875290b5…) | torch 2.6.0+cu124, transformers 4.52.4, python 3.11.11, GPU 없음 |
| 메인 저장소 `/home/inuos/CloudGPUFit` | codex/vram-native-inference-20261103 @7f822d7 | predictor 해시 **불일치** → 이번 분석 대상 아님 |
| 결과 패키지 | `/home/inuos/CloudGPUFit_CPU_predictions_20261110` | 원본 prediction/input 342쌍, selected-plan, revisions, lock |
| 이전 L4 실측 | `/home/inuos/cloudgpufit-comparison/aws-actual-20261110/g6-2xlarge-all342-20261102.tar.gz` → `inputs/aws_g6_2xlarge/`에 압축 해제 | worker `bench_worker_v3.py` sha256 4018cd8e… = run-manifest |

## 3. 호스트 자원
- CPU 24코어 (Intel Xeon Cascadelake), RAM 47GiB, swap 없음, 디스크 여유 약 39GiB
- 실험은 jobs=1, 컨테이너 4 threads, 메모리 제한 36GiB, 네트워크 차단(가중치 1회 다운로드 제외)
- tmux 세션 `cgf-diag`에서 실행

## 4. 이전 설명과 대조
| 항목 | 결과 |
|---|---|
| CPU는 forward/generate 텐서 수명 사용 (학습 보정 없음) | **일치** — profiler `[memory]` 이벤트(CPU device)만 사용, gradient/optimizer 항 없음 |
| warmup=0, measured=1, 빈 allocator + 합성 상주 | **일치** — `trace_summary`: cold_single_request, empty_plus_resident_model_and_inputs |
| 워밍업은 profiler 시작 전 수행, 측정 구간만 선택 | **일치** — `cpu_tensor_trace.py:89-99`. 인자만 늘리면 warmup 이력은 trace에 들어가지 않음 |
| GPU: 한 프로세스에서 performance→memory, 캐시 유지 | **일치** — profile `local_core`: warmup 1, performance 5회, memory 3회, `keep_allocator_cache`, 모드 사이 `gc.collect()`만 |
| L4 LLM 115조건 중 100조건 startup=피크, 115조건 모두 측정 샘플 일정 | **일치** (decoder_llm, 양쪽 성공 115조건 재계산) |
| CPU random(from_config) vs GPU pretrained | **일치**. 추가 확인: CPU에서 GPU와 같은 seed/방식으로 입력을 만들면 input_ids sha256이 L4 기록과 동일 → 입력은 완전 통제 가능 |
| 추가 차이 (새로 확인) | CPU 기준선은 입력(input_ids, attention_mask)을 GPU 상주 메모리로 합성 배치. GPU는 입력을 host에 두고 호출마다 `.to("cuda")`. 크기는 4KiB~48KiB로 작음 |
| 이전 AWS의 CUDA_MODULE_LOADING | **기록 없음** — docker 실행 인자에 미지정, 결과 JSON에도 없음. LAZY였다고 확정 불가 |
| PYTORCH_CUDA_ALLOC_CONF 등 | AWS 실행 인자에 미지정(기본값 추정), 결과에 실제값 기록 없음 |

## 5. 추적·집계 의미
- 이벤트: chrome trace `cpu_instant_event` `[memory]`, `Device Type 0`, `Bytes≠0`
  - `Bytes` = CPU 할당 요청 크기(원요청), 음수 = 해제, `Addr` 재사용됨, `Ev Idx` = 기록 순서
- 요청 선택: `cloudgpufit_inference` 구간 안의 할당만, 해제는 구간 밖이어도 짝지음
- 상주 메모리: `model.parameters()+buffers()` 고유 storage 크기 + 입력 storage, 측정 구간 시작 1µs 전 동일 timestamp로 배치
- xMem 시뮬레이션: timestamp별 그룹 → 그룹 안에서 **할당 먼저, 해제 나중** (`simulator.py` `sorted(key=is_allocated)`)
- allocated 피크 = `max(max_usage_changes)` (블록 크기 합), reserved 피크 = `max(max_segment_changes)` (segment 합)
- 용량 인자: `gpu_memory_gb`(기본 80) × 2^30 → `allowed_memory_maximum`. 초과 시 캐시된 전체 segment 반환 후 재시도, 실패 시 OOM
- 원본 요청 단위 trace: 기존 342조건 결과에는 **보존되지 않음** → 이번에 3조건 재실행으로 확보 (`runs/baseline/*/trace.json.gz`)

## 6. PyTorch v2.6.0 원문 대조 (`provenance/pytorch_v2.6.0_ref/`, SHA256SUMS 포함)
- memory history `alloc/free_*` size = orig_size/requested_size, `segment_*` size = segment 크기
- `get_allocation_size`: `size < kMinLargeAlloc(10MiB)` → 20MiB. xMem은 `<=` → 정확히 10MiB(반올림 후) 요청만 차이
- `get_free_block`: 다른 stream 블록이면 false. xMem은 stream 검사 없음 (합성 테스트로 재사용 발생 확인)
- 용량 초과 시: `release_available_cached_blocks`(max_split_size 기본값이면 no-op) → `release_cached_blocks` → 재시도 — xMem도 같은 순서
- cuBLAS workspace: handle·stream당 1회 `CUDACachingAllocator::allocate(4096KiB×2+16KiB×8 = 8.125MiB)` (sm90 제외), 해제되지 않고 유지
- cuBLASLt: 호출마다 `at::empty(1MiB)` 일시 할당 (Lt 경로를 타는 GEMM만)

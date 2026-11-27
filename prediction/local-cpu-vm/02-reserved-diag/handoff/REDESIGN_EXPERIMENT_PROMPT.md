# 실험 재설계 요청: CPU 예측 vs GPU 실측 VRAM 비교 (장치 외 차이 제거)

## 0. 역할과 목표
- 너는 실험 설계자다. **구현·실행 전에 설계안을 먼저 제출**하고 승인을 받는다.
- 목표: CloudGPUFit의 CPU 기반 VRAM 예측과 GPU 실측을 다시 수행하되, 두 결과의 차이가 **오직 장치(CPU 추적+allocator 시뮬레이션 vs 실제 CUDA)** 에서만 나오도록 실험을 설계한다.
- 실험 순서, 측정 대상, 입력, 모델 가중치, 소프트웨어 환경, 측정 구간 경계를 양쪽에서 동일하게 맞춘다.

## 1. 배경 (현재 시스템)
- CPU 예측: `torch.profiler`의 CPU `[memory]` 이벤트로 텐서 할당·해제 요청을 추적 → xMem `AllocatorSim`(PyTorch CUDACachingAllocator 재현)에 재생 → allocated/reserved 피크 산출.
  - 실제 경로: `predict_vram.py` → `cpu_tensor_trace.trace_and_estimate` → `inference_trace_runtime.simulate` → xMem `AllocatorSim`
  - 상태: 빈 allocator + 모델·입력 storage 합성 배치 → generate **1회**만 추적 (warmup은 추적 밖), `from_config` 랜덤 가중치
- GPU 실측(이전): AWS g6.2xlarge(L4), `bench_worker_v3.py`, profile `local_core`, 342조건 계획 `benchmark_plan_v2.json`
- 소프트웨어: torch 2.6.0+cu124, transformers 4.52.4, python 3.11.11, attention eager

## 2. 진단으로 확인된 문제 (2026-11-10~11, 재설계의 근거)
- **이전 GPU 측정 절차** (한 조건 = 한 프로세스, 그 안에서 연속 실행, 캐시 유지)
  1. 모델 로딩 (`from_pretrained`, `device_map=cuda`, `low_cpu_mem_usage`)
  2. performance 모드(속도 측정): 출력 검증 forward(`model(**inputs)`, 전체 시퀀스 logits) → warmup 1 → generate 5
  3. 모드 사이 `gc.collect()`만, `empty_cache()` 없음
  4. memory 모드(VRAM 측정): 검증 forward → warmup 1 → `reset_peak_memory_stats()` → generate 3 측정
- `reset_peak_memory_stats()`는 피크 통계만 초기화하고 캐시는 그대로 → 측정 피크 = max(측정 구간 실제 필요량, 앞 단계가 잡아 둔 캐시)
- 근거
  - L4 325조건 전부: memory 모드 측정 피크 = 측정 시작 시점 reserved
  - decoder LLM·encoder-decoder: 큰 reserved 오차(20%↑)가 전부 GPU>CPU, allocated는 대부분 일치. 실측 ≈ max(CPU 예측, 로딩 직후 캐시, 검증 forward 피크)로 두면 20%↑ 오차 0건
  - LLM: 검증 forward의 전체 logits가 주원인 (예: gpt2 fp16 b4/768/128 — 예측 928, 검증 forward 1672, 실측 1678 MiB)
  - encoder-decoder small: 모델 로딩 캐시가 주원인 (예: 예측 132, 로딩 직후 272, 실측 274 MiB)
  - RTX 3090에서 memory history를 받아 같은 요청을 xMem에 재생 → 실측과 0~2 MiB. allocator 정책 차이는 원인 아님
  - CPU에서 GPU와 같은 13단계 이력을 재생 → gpt2-medium fp32 4574 vs 실측 4574 MiB
- **이력과 별개로 남은 차이** (재설계 후에도 측정·보고해야 함)
  - vision_cls: CPU가 과대 예측, allocated도 크게 다름 (bf16 다수) → 요청 자체 차이 추정
  - decoder LLM bf16: allocated를 CPU가 28~82% 과대 예측 (이전엔 캐시에 가려짐)
  - GPU 전용 요청: cuBLAS workspace(handle·stream당 8.125 MiB, 유지), cuBLASLt 1 MiB 일시 할당(Lt GEMM마다), cuDNN workspace(vision 추정)
  - CPU 전용 요청: 작은 스칼라 텐서 등
  - 이전 AWS 실행의 `CUDA_MODULE_LOADING`, `PYTORCH_CUDA_ALLOC_CONF` 미기록
- **CPU 추적 구현 주의**: 같은 프로세스에서 두 번째 `torch.profiler` 세션은 RSS가 단계당 ~0.6 GB 증가해 OOM → 추적 단계마다 새 프로세스 사용 (요청 순서·크기 동일성 검증 완료)

## 3. 설계 요구사항
1. **공통 프로토콜 1개**
   - 단계·순서·반복 수·입력 생성·generate 인자를 하나의 정의(코드/설정)로 만들고, `device=cuda`(실측)와 `device=cpu`(추적)가 같은 정의를 실행
   - 각 측정 구간의 시작·끝 경계를 양쪽에서 같은 지점으로 정의
2. **격리**
   - 측정 지표마다 allocator 시작 상태를 명시적으로 정의하고 양쪽에서 동일하게 만든다
     - 예: 로딩 후 `synchronize → gc.collect → empty_cache → reset_peak_memory_stats` 후 시작, 시작 시점 allocated/reserved를 양쪽 모두 기록
   - 서로 다른 지표(로딩, 첫 추론, warmup 후 정상상태 등)는 별도 프로세스, 또는 같은 프로세스라도 경계와 캐시 처리를 양쪽 동일하게
   - 속도(latency) 측정은 VRAM 측정과 allocator를 공유하지 않게 분리
   - 출력 검증(finite check)은 측정 구간 뒤 또는 별도 프로세스. 측정 앞에 두려면 CPU 쪽 프로토콜에도 똑같이 포함
   - 조건 간 간섭 없음: 조건마다 새 프로세스, 시작 시 GPU 사용량 0 확인
3. **"앞 단계 이력"을 지표로 정의**
   - 다음을 각각 별도 지표로 측정·예측하도록 설계
     - (a) 모델 로딩 후 상주·캐시
     - (b) 빈 캐시에서 첫 추론 1회
     - (c) warmup N회 후 추론
   - 어떤 지표를 제품 예측값으로 쓸지는 사용자 결정 사항으로 남김
4. **동일 조건**
   - 같은 가중치(양쪽 pretrained 또는 양쪽 from_config 중 선택하고 이유 제시), 같은 revision pin
   - 같은 입력 생성(seed `20261006`, `torch.Generator().manual_seed`, `randint(3, vocab)`, special token→3 방식), 입력 sha256 기록·대조
   - 생성 길이 고정 (EOS 조기 종료 방지, 실제 생성 토큰 수 기록)
   - 같은 dtype·attention 구현·TF32 off·`cudnn.benchmark` 설정
   - 같은 Docker 이미지 digest·패키지 버전. 환경변수 고정·기록 (`CUDA_MODULE_LOADING`, `PYTORCH_CUDA_ALLOC_CONF` 포함)
   - 입력 위치(host→`.to(device)` 시점)도 양쪽 동일하게
5. **요청 단위 비교 가능성**
   - GPU: 최소 파일럿 조건에서 `torch.cuda.memory._record_memory_history` + snapshot을 경계 마커와 함께 수집
   - CPU 요청 trace 원본 보존
   - GPU 전용·CPU 전용 요청을 분리 집계해서, 장치 고유 차이(workspace, 연산 경로)를 숨기지 말고 정량 보고
6. **측정 지표**
   - 지표별 allocated 피크, reserved 피크, 시작 시점 값
   - NVML 프로세스 사용량은 별도 열 (allocated와 직접 비교 금지)
   - 단위는 bytes 정수
7. **GPU 기종**: 이전과 같은 L4(g6.2xlarge) 사용을 기본으로. 다른 기종이면 영향(cuBLAS workspace 등 아키텍처 의존 항목)을 명시
8. **조건 범위**
   - 1차 파일럿: 아래 대표 조건
     - gpt2 fp16 b1/256/64
     - gpt2 fp16 b4/768/128
     - gpt2-medium fp32 b4/768/128
     - encoder-decoder small fp16 1개
     - vision_cls bf16 1~2개
     - decoder LLM bf16 1개
   - 2차: 기존 342조건 계획 전체
9. **제약**
   - 기존 코드·결과 수정 금지 (별도 경로·브랜치)
   - 설계 승인 전 production 예측 로직 변경·보정식 추가 금지
   - 가짜 GPU 데이터 금지, 대형 모델 무단 다운로드 금지
   - 장시간 작업은 tmux에서 실행

## 4. 참고 자료 (inuos VM)
- 진단 실험 루트: `/home/inuos/cgf-reserved-diag-20261110`
  - `README.md`, `provenance/ENV_AND_PATH_REPORT.md`
  - `src/cgf_diag/cpu_history_trace.py`, `gpu_replay.py`, `merge_steps.py`
  - `handoff/gpu_collector_reference.py` (memory history 수집 참고)
  - `results/population_l4_vs_cpu/` (325조건 분석)
- 이전 GPU worker·계획: `inputs/aws_g6_2xlarge/bench_worker_v3.py`, `run_bench_v3.py`, `benchmark_plan_v2.json`
- 이전 CPU 예측 결과: `/home/inuos/CloudGPUFit_CPU_predictions_20261110`
- predictor 소스: `/home/inuos/CloudGPUFit-cpu-singlepass-20261108` (실험 기준 버전)

## 5. 제출물 (설계 단계)
1. 공통 프로토콜 정의: 단계, 순서, 경계, 캐시 처리, 프로세스 분리 방식 (표 1개 + 단계 다이어그램)
2. 지표 정의표: 지표별 GPU 측정 방법 ↔ CPU 예측 방법 대응
3. 동일성 체크리스트: 환경·입력·가중치·설정. 실행 시 자동 검증 방법 포함
4. 남는 장치 고유 차이 목록과 정량화 방법
5. 파일럿 합격 기준
   - 예: 같은 GPU 요청을 xMem에 재생 시 0~2 MiB
   - 예: 지표 (b)에서 CPU-GPU 차이를 요청 단위로 전부 설명
6. 구현 계획: 수정·신규 파일, GPU runner·CPU runner, 결과 스키마
7. 실행 시간·비용 추정 (CPU 추적은 조건에 따라 generate 1회 수 분~수십 분)
8. 사용자 결정이 필요한 항목 목록 (예: 제품 예측 대상 지표, 가중치 종류, GPU 기종)

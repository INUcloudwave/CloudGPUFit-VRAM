# INUOS(CPU 예측) 쪽 맥락 — 로컬 분석 AI용 (2026-11-15 작성, 11-20 갱신: §4.3~4.6 추가)

> **보고서를 쓰려면 먼저 루트의 `../REPORT_CONTEXT_KO.md`를 읽는다.** INUGPU·INUOS·AWS 전체를 통합해 왜·어떻게·결과·반영·코드 변화를 정리한 문서다. 이 문서는 INUOS 쪽 세부다.

이 문서는 `../AI_CONTEXT_KO.md`(INUGPU 작성)의 §7 "INUOS 쪽 정보(11-12 시점)"를 **최신 상태로 대체**합니다.
먼저 `../AI_CONTEXT_KO.md`를 읽고, 이어서 이 문서를 읽으세요.

- 표기
  - "확인됨": 코드·로그·결과 파일로 확인한 사실
  - "추정": 근거는 있지만 직접 검증하지 않은 해석
  - "미확인": 아직 검증하지 않은 내용
- 단위: 결과 파일 원값은 byte. MiB = 2^20 B

---

## 1. INUOS가 하는 일

- GPU 없이 CPU VM에서 GPU VRAM(PyTorch CUDA caching allocator의 allocated/reserved)을 예측합니다.
- 방법
  1. CPU에서 실제 추론을 실행하면서 `torch.profiler`로 CPU 텐서 할당·해제 요청(`[memory]` 이벤트)을 순서대로 기록
  2. (FP16 실험) FP32로 실행하고 요청 크기를 FP16 기준으로 환산 (CPU에 FP16 하드웨어가 없어 네이티브 FP16이 매우 느림)
  3. xMem(PyTorch CUDA caching allocator 시뮬레이터, `1ee4bd08`, 무수정)에 요청을 순서대로 재생
  4. INUGPU 실측과 같은 키(`summary.first_measure_peak_reserved` 등)로 대조
- VM: Intel Xeon Cascade Lake 24 vCPU, RAM 47 GiB, swap 없음. FP16/AMX 명령 없음
- 실행 이미지: `cloudgpufit-cpu-validation:20261108` (`sha256:875290b5…`), python 3.11.11, torch 2.6.0+cu124, transformers 4.52.4

## 2. 실험 이력 요약 (상세는 `docs/`)

| 기간 | 실험 | 상태 | 레포 경로 | 상세 문서 |
|---|---|---|---|---|
| 10-26 | baseline9: v2 휴리스틱 예측기 (FP32 prefill ×0.5 + 보정 상수 + KV 해석식) | 완료 9/9, 이후 폐기 | `00-early-eval-recovery/cloudgpufit-eval/` | `docs/01` §3.1 |
| 11-03 ~ 11-05 | native v3 (native dtype `from_config`, full generate, warmup 1 + 측정 3) all342 | 중단: 244 ok / 17 host OOM / 81 미실행 | `01-cpu-predictor-342/cloudgpufit-predictions-runs/native-v3-t4-j6/` | `docs/01` §3.4 |
| 11-08 | recovery(스트리밍 파서, 컨테이너별 RAM 제한) → singlepass v4 (warmup 0 + 측정 1) | 검증 완료 | `01-cpu-predictor-342/predictor-singlepass-20261108/` | `docs/01` §3.5~3.8 |
| 11-08 ~ 11-10 | **342조건 CPU 예측 v4** | 325 ok / 17 실패 | `01-cpu-predictor-342/CloudGPUFit_CPU_predictions_20261110/` | `docs/01` §3.9~3.10 |
| 11-10 ~ 11-11 | **reserved 오차 원인 분리** (342 예측 vs AWS L4) | 완료 | `02-reserved-diag/` | `docs/02` |
| 11-11 ~ 11-12 | FP16 680 조건표, CPU 프로토콜 v1, FP16 shadow 환산, v2(GPU 프로토콜 정렬) | 완료 | `03-fp16-cpu-prediction/` | `docs/03` §2.1~2.13 |
| 11-12 ~ 11-15 | main395 (구 L1024 조건표) CPU 예측 | **사용자 요청으로 중단**: 388 결과 (375 ok / 6 cpu_oom / 7 timeout) | `03-fp16-cpu-prediction/runs/main395/` | `docs/03` §2.14 |
| 11-15 | GPU-CPU 정렬표, 용량 재생, L256 전환, 스레드 확인 | 완료 | `03-fp16-cpu-prediction/handoff_codex/`, `tools/` | `docs/03` §2.15~2.18 |
| 11-15 19:18 ~ 11-16 09:44 | **main395_L256 CPU 예측** (새 151개 실행 + 244개 재사용) | 완료: 395 결과 (386 ok / 9 cpu_oom) | `03-fp16-cpu-prediction/runs/main395_l256/`, `results/main395_l256/` | 아래 §4.1 |
| 11-16 | GEMM probe / exact (CPU FP16 GEMM 경로 점검) | 실행 기록만 추가 | `03-fp16-cpu-prediction/runs/gemm_probe/`, `runs/gemm_exact/` | - |
| 11-16 | reserved 예측 v3 (내림차순 주소 + cuBLAS 작업 공간 + 실효 용량), 원인 확인 실험 | 완료 | `03-fp16-cpu-prediction/results/reserved_v3/` | §4.2 |
| 11-17 | T5 환산 규칙 v2 재실행 (효과 없음), 합성곱 모델 층 분해 (정확 3/3), 합성곱 방침 | 완료 | `results/reserved_v3/rule2_t5/`, `confirm/hybrid/` | §4.3 |
| 11-17 | cuBLAS 작업 공간을 GPU 모델명 → 세대로 계산 (`gpu_rules.py`), GPU 프로필 확장 | 완료 | `src/cgf680/gpu_rules.py`, `manifest/gpu_profiles.json` | §4.4 |
| 11-17 ~ 11-18 | AWS r7i.8xlarge 실제 FP16 CPU 예측 395개 (사용자 실행) | 완료 395/395 ok | `../aws/cpu_fp16_real_r7i/` | §4.5 |
| 11-18 ~ 11-20 | AWS GPU 5종 실측으로 주소 규칙·cuBLAS·outside 확인, CPU 실행 방식 비교 | 완료 | `results/reserved_v3/aws_gpu_check/` | §4.6 |

## 3. 지금까지의 핵심 결론

1. **이전 342 예측 vs AWS L4의 큰 reserved 오차 = 실행 이력 차이** (확인됨, `docs/02`)
   - 이전 GPU 측정은 같은 프로세스에서 로딩 → performance 모드(검증 forward, warmup, 5회) → memory 모드를 캐시 유지로 이어서 실행
   - 특히 출력 검증 forward(전체 시퀀스 logits)와 모델 로딩 캐시가 reserved를 키움
   - CPU 예측은 빈 allocator에서 generate 1회만 봄 → 서로 다른 상황을 잰 것
   - 같은 GPU 요청을 xMem에 넣으면 실측과 0~2 MiB → allocator 정책은 원인 아님
   - 이 결과로 INUGPU가 측정을 재설계함 (VRAM 전용, 로딩 → warmup 3 → measure 5, 검증 forward 제거)
2. **재설계 후(FP16, RTX 3090 L1024 비교, 둘 다 ok 374조건)** (확인됨, `docs/03` §4)
   - 입력 sha256 375/375 일치, attention 구현 이름 374/374 일치, 로딩 직후 reserved 374/374 오차 0
   - LLM·T5 allocated 피크: 모델별 중앙값 오차 −0.1 ~ −1.5% (CPU가 약 6~10 MiB 작음, GPU 전용 workspace 추정)
   - LLM·T5 reserved 오차 20% 초과 15건: 전부 CPU 재생 용량을 1 TiB로 둬서 GPU(24 GB)에서 일어나는 캐시 반환이 재현되지 않은 것
     - GPU 용량(24,979,374,080 B)으로 재생하면 ±5% 9건, 5~10% 3건, 크게 어긋남 3건
   - CNN(resnet-18/50, mobilenet_v2): allocated부터 −16% / −9% / +28% → conv 연산 경로 차이로 추정, 미확인
   - GPU OOM 조건은 CPU도 RAM 부족(42 GB)으로 예측 자체가 실패 (FP32 실행이라 메모리 2배 이상)

## 4. 현재 진행 상태 (2026-11-15 19시)

| 항목 | 상태 |
|---|---|
| GPU L256 본실험 (INUGPU) | 17:00 시작, 진행 중 (18:45 기준 measure 107/395 ok) |
| CPU main395 (L1024) | 중단·보존. 결과는 `runs/main395/cases/*/result.json` |
| CPU L256 전환 | 조건표 교체, 겹치는 244개 결과 재사용(심볼릭 링크) 완료 |
| CPU L256 새 151개 실행 | **완료 (11-16 09:44)**. §4.1 참고 (아래 표는 11-15 19시 기준 그대로 둠) |
| 용량 재생 (395 전체) | 도구 완성(`tools/replay_capacity.py`, 1 TiB self-check 5/5 일치), 전체 실행 전 |
| CNN GPU trace 요청 | 요청서 작성(`handoff_codex/INUGPU_CNN_TRACE_REQUEST_KO.md`), GPU L256 종료 후 실행 요청 예정 |

- CPU L256 실행 설계 (`tools/run_l256.py`, worker `cpu_worker2.py`는 무수정)
  - P단계: 3개 병렬, cpuset 분리(0-6 / 7-13 / 14-20, 7스레드), 조건마다 새 컨테이너
  - 메모리 감시: 호스트 가용 메모리 < 5 GiB이고 2개 이상 실행 중이면 **가장 늦게 시작한 작업**을 종료(`preempted_low_mem`)
  - S단계: P단계가 모두 끝난 뒤, 메모리 실패·시간 초과 작업을 순차 실행(20스레드, 제한 없음). S 결과로 확정
  - 예상 소요 20~30시간 (추정)

### 4.1 main395_L256 CPU 예측 결과 (2026-11-16 추가)

- 실행: 11-15 19:18 ~ 11-16 09:44 (tmux `cgf-l256`, 스케줄러 `tools/run_l256.py`)
  - 실제 배치: 큰 작업 슬롯 0-12(13스레드) + 작은 작업 슬롯 13-17 / 18-22(5스레드), 메모리 추정 기반 진입 제어
- 결과: 395 조건 모두 결과 있음 — **386 ok / 9 worker_killed_cpu_oom**
  - 실패 9개: gpt2-xl b128 3개(in128/new512, in512/new128, in512/new512), t5-3b b128 6개 (모두 trace 단계 RAM 부족)
- 파일
  - 조건별 예측 모음: `03-fp16-cpu-prediction/results/main395_l256/cpu_predictions.csv` (`.jsonl` 동일 내용)
  - 실측(RTX 3090) 비교 일부: `results/main395_l256/pilot_vs_rtx3090.csv` (26개 조건)
  - 상태 요약·실패 목록·시도 이력: `results/main395_l256/status_summary.json`
  - 조건별 원본: `runs/main395_l256/cases/<condition_id>/result.json`
    - 244개는 `../../main395/cases/` 를 가리키는 심볼릭 링크(재사용, 목록은 `runs/main395_l256/reuse_manifest.json`)
  - 스케줄러 기록: `runs/main395_l256/{attempts,scheduler_events,suite_invocations}.jsonl`, 콘솔 로그 `runs/main395_l256.console.log`
- 전체 395개 GPU 실측과의 비교표는 아직 만들지 않음 (GPU 결과는 `../inugpu/05-fp16-vram/` 쪽)
- reserved 예측 v3는 §4.2 참고 (11-16 저녁 추가)

### 4.2 reserved 예측 v3 (2026-11-16 추가)

- 목적: GPU `first_measure_peak_reserved`를 CPU 정보만으로 예측한다 (INUGPU `RESERVED_PREDICTION_GUIDE_KO.md` §0).
- 바꾼 것: 주소 규칙(새 세그먼트는 아래 주소), cuBLAS 작업 공간 8,519,680 B를 첫 GEMM 직후에 삽입(CPU 프로브로 위치 확인), 용량 = cuda_total − outside 312 MiB
- 결과 (RTX 3090, 비교 519개, 허용 오차 100 MB)

| 그룹 | 개수 | v3 | v2 |
|---|---|---|---|
| 일반 | 436 | 96.8 % | 86.9 % |
| 재시도 | 23 | 60.9 % | 0 % |
| CNN | 60 | 68.3 % | 65.0 % |
| 전체 | 519 | 91.9 % | 80.5 % |

  - 재시도 판정은 23/23, GPU OOM 판정은 3/3이 맞았다.
- 조건별 GPU 실측은 예측에 쓰지 않았다. 출처는 `03-fp16-cpu-prediction/handoff_codex/reserved_prediction_inputs_KO.md`
- 보고서: `handoff_codex/reserved_prediction_report.md`, 조건별 결과: `handoff_codex/reserved_prediction_per_condition.csv`
- 원인 확인 실험: `results/reserved_v3/confirm/`
  - CNN: 합성곱 내부 버퍼가 장치마다 다름 (확정)
  - qwen: GPU 트레이스 요청 (`handoff_codex/INUGPU_TRACE_REQUEST_20261116_KO.md`)
  - 재시도: 미확정
- 코드: `src/cgf680/{replay3,gemm_probe,gemm_shadow}.py`, `tools/{reserved_predict,compare_reserved,run_gemm_exact_all}.py`
- 실험 과정과 단계별 스크립트·결과: `results/reserved_v3/PROCESS_KO.md` (가이드 검증 `guide_verification/`, 원인 확인 `confirm/`)
- 합성곱 모델(resnet, mobilenet) 오차 원인 확정(11-17): 라이브러리 층 차이. mobilenet은 CPU 버퍼 제거로 해결, resnet은 cuDNN 작업 공간(범위로 보고). `results/reserved_v3/confirm/hybrid/`, `handoff_codex/reserved_conv_models_policy_KO.md`

### 4.3 T5 환산 규칙 v2, 합성곱 모델 층 분해 (2026-11-17)

- T5 24/48 MiB 차이 원인: `torch.isinf` 안의 임시 텐서 `abs(x)`가 bool 출력 태그를 물려받아 FP32 크기로 남음 (확인됨, `results/reserved_v3/confirm/T5_DOUBLE_SIZE_KO.md`)
  - 수정: 환산 규칙 v2 (`src/cgf680/rule2_shadow.py`, `cpu_worker2.py --shadow-rule v2`). 생산 규칙 v1은 그대로 둠
  - t5-small/base 90개 재실행 + v3 재생: reserved 변화 4개 조건 ±2 MiB → 사실상 효과 없음 (`results/reserved_v3/rule2_t5/COMPARE_KO.md`)
- 합성곱 모델 오차 원인 (확인됨, `results/reserved_v3/confirm/hybrid/README_KO.md`)
  - 요청 목록을 모델 층(연산 출력)과 라이브러리 층(CPU oneDNN 버퍼 / GPU cuDNN·cuBLAS 작업 공간)으로 나눔
  - CPU 모델 층 == GPU 모델 층 (해제 순서까지, 3/3). CPU 모델 층 + GPU 라이브러리 층 재생 = GPU 실측 (3/3)
  - MobileNet 과대예측 = CPU 버퍼 → 제거하면 해결 (6580 = 6580 MiB). ResNet 과소예측 = cuDNN 작업 공간 → CPU로는 알 수 없음
- 방침 (구현 전): `handoff_codex/reserved_conv_models_policy_KO.md`

### 4.4 cuBLAS 작업 공간 계산과 GPU 프로필 (2026-11-17 ~ 18)

- `src/cgf680/gpu_rules.py`: GPU 이름 → compute capability → PyTorch 규칙(Hopper 32 MiB, 나머지 8,519,680 B, `CUBLAS_WORKSPACE_CONFIG` 우선)
- `tools/reserved_predict.py --gpu rtx3090,t4,...`: 여러 GPU를 한 번에 예측 (첫 GPU가 main, 나머지는 `by_gpu`). 3090 결과는 이전과 바이트 동일
- `manifest/gpu_profiles.json`: rtx3090 + AWS 6종(t4, a10g, l4, l40s, rtxpro4500, rtxpro6000). 11-18에 AWS 실측으로 5종의 cuda_total_memory, outside, 실효 용량, driver를 채움 (rtxpro6000은 미측정)

### 4.5 AWS 실제 FP16 CPU 예측 (2026-11-17 16:08 ~ 11-18 02:21 KST)

- `cpu_worker2.py --dtype-mode fp16_real` (FP32 실행 + 환산 대신 실제 FP16 실행, 같은 실행에서 첫 GEMM 위치 기록), 스케줄러 `tools/run_aws.py`, 번들 `aws/`
- r7i.8xlarge (32 vCPU, 247 GiB), INUOS와 같은 이미지·조건표·스레드 수, 코어만 줄인 병렬
- 395/395 ok (INUOS 메모리 부족 9개 포함), v3 재생 395줄 오류 0
- 결과: `../aws/cpu_fp16_real_r7i/`, 설명 `../aws/README_KO.md`

### 4.6 AWS GPU 5종으로 확인 (2026-11-18 ~ 20)

- `results/reserved_v3/aws_gpu_check/README_KO.md`
  - 내림차순 주소 규칙: A10G·L4·L40S·Blackwell에서 3090과 같은 동작 (판별용 조건 48/52, 48/52, 48/52, 46/48, 나머지는 ±2 MiB). T4는 SDPA 모델의 요청 목록 차이로 판별 불가
  - cuBLAS 작업 공간: 5종 모두 8,519,680 B (디코더 190~192개 조건)
  - outside: T4 130, A10G 308, L4 232, L40S 510, RTX PRO 4500 320 MiB (중앙값). LLM·T5는 상수, 합성곱 모델은 +2~4 MiB
- `aws_vs_inuos_rinf.txt`: INUOS 환산 방식과 AWS 실제 FP16 실행의 정확도 비교. LLM 같음, T5 AWS가 약간 나쁨, CNN은 AWS 과대예측 큼
- PPT 예시 수치: `../REPORT_CONTEXT_KO.md` §8.3

## 5. 열린 문제 (분석 AI가 판단해 줄 것)

1. **torch 스레드 수에 따라 CPU 요청 trace가 달라짐** (확인됨, `03-fp16-cpu-prediction/results/thread_check/`)
   - 4조건 × 스레드 1/5/7/20: 요청 trace 완전 동일 2/12, 예측값(summary) 동일 12/12
   - 스레드 수에 비례하는 CPU 전용 임시 버퍼(예: gpt2 248,320 B = 20 × 12,416 B)
   - 기존 245개는 5 또는 20스레드, 새 151개는 7(P) 또는 20(S)스레드 → 섞임. 큰 조건 영향 미확인
2. **용량 근처 reserved**: 캐시 반환 시점에 민감. reserved 수치 대신 OOM 판정(들어가는가)을 1차 목표로 둘지
3. ~~CNN allocated 오차 원인~~ → **해결 (11-17, §4.3)**: MobileNet은 CPU oneDNN 버퍼, ResNet은 cuDNN 작업 공간
4. ~~T5 요청 2배 문제~~ → **원인 확인 (11-16, §4.3)**, reserved 영향 없음. 원래 기록: GPU 8 KB ↔ CPU 16 KB, GPU 1 MiB ↔ CPU 2 MiB. shadow 태그에서 FP32 고정(`f`)으로 잘못 분류됐을 가능성(추정). T5 allocated 오차는 −0.1~−1.5%로 작음
5. **GPU 전용 할당**: cuBLAS 작업 공간은 v3에서 반영 (§4.2, §4.4). cuBLASLt 1 MiB(일시)는 reserved에 영향 없음
6. **스레드·용량 외 정렬 항목 전체**: `03-fp16-cpu-prediction/handoff_codex/gpu_cpu_alignment.md` (15항목 판정표)

## 6. GPU 결과와 비교하는 방법

- CPU 결과: `03-fp16-cpu-prediction/runs/main395/cases/<condition_id>/result.json`
  - `summary.first_measure_peak_reserved`, `first_measure_peak_allocated`, `first_measure_start_reserved`, `after_load_reserved` (byte, GPU와 같은 키)
  - `variant_with_cublas_workspace`: 예측값 아님(cuBLAS 8.125 MiB 1회 주입 변형)
  - `inputs.<name>.sha256`: GPU `input_sha256`과 대조
  - 모아 본 표: `03-fp16-cpu-prediction/results/main395/cpu_predictions.csv`
- GPU 결과: `../inugpu/05-fp16-vram/results/gpu_rtx3090_main395_L1024_per_condition.csv` (byte, `m1_peak_reserved` 등, `input_sha256`은 `name=sha;name=sha`)
- 비교 도구
  - `03-fp16-cpu-prediction/handoff_codex/alignment_20261115/compare_l1024.py` (L1024, 374조건)
  - `03-fp16-cpu-prediction/tools/compare_gpu_l256.py` (L256용, GPU CSV + 용량 재생 결과)
  - `03-fp16-cpu-prediction/tools/check_gpu_overlap.py` (GPU L1024 ↔ L256 겹치는 245개 동일성)
- 조건 ID는 GPU `condition_id`와 같습니다 (`manifest/cases395*.jsonl`의 `case_id`).

## 7. 코드 지도 (FP16 실험, `03-fp16-cpu-prediction/`)

| 파일 | 역할 |
|---|---|
| `src/cgf680/protocol2.py` | GPU worker(`vram_worker.py` d06f0f15)와 같은 입력·생성 설정·호출 (장치 무관 부분) |
| `src/cgf680/cpu_worker2.py` | **현재 CPU worker**. FP16 모델 구조로 로딩 요청 합성 → FP32 모델로 호출 1회 추적 → shadow 태그로 FP16 크기 → xMem에 로딩 + 호출 ×4(warmup 3 + measure 1) 재생 |
| `src/cgf680/load_emu.py` | GPU `from_pretrained(device_map=cuda)` 로딩 요청 순서 합성 (13/13 모델 RTX 3090과 일치) |
| `src/cgf680/shadow.py` | FP32 실행에서 FP16 요청 크기를 계산하는 TorchFunctionMode 태그, T5 FP16 분기 패치 |
| `src/cgf680/cpu_worker.py`, `protocol.py` | v1 (eager 고정, 4회 실제 추적). 현재 미사용 |
| `tools/run395.py` | main395 스케줄러 (A 4병렬 / B 순차 / C 재시도) |
| `tools/run_l256.py` | L256 스케줄러 (3병렬 + 저메모리 시 최근 시작 작업 종료 → 순차 재실행) |
| `tools/replay_capacity.py` | 저장된 요청 trace를 GPU 용량으로 재생 (`manifest/gpu_profiles.json`) |
| `tools/collect395.py` | 조건별 result.json → `results/<run>/cpu_predictions.csv` |
| `tools/link_overlap.py` | 조건표가 겹치는 결과 재사용 |
| `tools/compare_requests_npz.py` | 두 요청 trace 완전 동일성 비교 |

- 재사용 코드: `02-reserved-diag/src/cgf_diag/replay.py`(`Replayer`, xMem 순서 재생), `events.py`
- 줄 단위 설명: `docs/03_fp16_cpu_prediction_KO.md` §3, `docs/02_reserved_diag_KO.md` §3, `docs/01_cpu_predictor_342_KO.md` §4

## 8. 레포에 넣지 않은 것

- 실험 코드·입력·결과·로그는 모두 포함했습니다 (요청 trace npz, 원시 trace, GPU 반환 snapshot 포함).
- 뺀 것은 `EXCLUDED_LARGE_FILES.md`: 50MB 초과 파일 11개(sha256 기재), 모델 가중치, 호스트 커널 로그·docker inspect, 같은 레포 커밋 7f822d7과 동일한 앱 파일, zip 사본
- `03-fp16-cpu-prediction/runs/main395_l256/cases/`의 244개는 `../../main395/cases/<id>`를 가리키는 심볼릭 링크입니다.

## 9. 운영 제약 (사용자 지시)

- production 예측 로직과 과거 결과는 보존. 원인 확인 전 보정식 추가 금지
- 실패 조건을 낮은 배치·다른 dtype으로 몰래 바꾸지 않음
- RAM 부족으로 실패한 조건은 재시도하지 않고 "CPU 예측 불가"로 기록 (사용자 결정, 11-15)
- 장시간 실험은 tmux에서 실행. 프로세스 종료는 사용자 확인 후
- 키·토큰·개인정보 기록 금지

# 02. outside 메모리 분석 (2026-11-03 ~ 11-10)

> 레포 경로 대응: VM `cloudgpufit-outside/` → 레포 `gpu-vm/02-outside-analysis/`. 대용량 제외 파일은 `EXCLUDED_LARGE_FILES.md` 참고.


- 대상 디렉터리: `cloudgpufit-outside/` (이 문서의 모든 경로는 이 디렉터리 기준 상대경로)
- 근거: `NOTION_SUMMARY_20261108.md`, `HANDOFF_KO_20261108.md`, `AWS_NEXT_STEP_20261110.md`, `outside-lab/*.md`, 각 `results/*/REPORT_KO.md`, `summary.json`, 로그·종료코드 파일
- 표기: **[미확인]** = 파일로 확인되지 않는 항목, **[주의]** = 문서 사이의 기록 충돌이나 정정 사항

---

## 1. 개요

### 1-1. 목적
- CloudGPUFit: GPU를 빌리기 전 CPU에서 추론 VRAM을 예측하고 AWS 인스턴스를 추천함
  - 최종 추정 = xMem 예측 reserved + **outside**
- 이번 단계는 학습 단계가 아님. outside에 영향을 주는 요인을 확인하고, 그 결과로 상수 / GPU별 표 / 예측식 중 무엇이 필요한지 판단함
- 출발점: 기존 AWS 벤치(bench-v3) 원본 6개 아카이브의 CPU 재분석 결과 (`outside-lab/EXPERIMENT_DESIGN.md`, `outside-lab/analysis/`)

| GPU | 첨부 기준 인스턴스 | 유효 조건 수 | 조건별 outside 중앙값 (범위) |
|---|---|---:|---|
| T4 | g4dn.2xlarge | 228 | 130MiB (128–132) |
| A10G | g5.2xlarge | 342 | 308MiB (302–312) |
| L4 | g6.xlarge / 2xlarge / 4xlarge | 각 342 | 232MiB (226–236) |
| L40S | g6e.2xlarge | 342 | 510MiB (504–514) |

- 기존 자료의 한계: 같은 조건을 반복한 실행이 없음(repeat=0), 측정 시점이 첫 초기화가 아님(`model_reused=true`), driver 595 사용(로컬 3090은 570), manifest에 인스턴스 라벨 충돌이 있음(`g6.2xlarge` 잔존)

### 1-2. outside 정의와 측정 규칙
- `outside = NVML 해당 PID 메모리 − torch.cuda.memory_reserved` (같은 조회 구간에서 짝지어 측정)
- 금지 사항: allocated 차감 / 서로 다른 시점의 최대값끼리 차감 / xMem 오차 흡수 / 음수·결측의 0 치환 / 다른 프로세스나 디바이스 전체 used로 대체
- 조회 순서는 TNT(torch→NVML→torch)가 기본. 주 outside는 첫 NVML 값과 첫 reserved 값의 차이
- 안정 관측: checkpoint마다 `synchronize` 후 50ms 간격으로 21회 측정
- 품질 필터(제외만 하고 원본은 보존): 주 조회 > 5ms / reserved bracket 변화 / 다른 GPU PID / 음수 / API 오류
- CUDA 초기화 전 PID 부재는 예상된 결측으로 처리하며 실패로 보지 않음
- plateau 진단 기준: clean 샘플 10개 이상, 마지막 10개의 span ≤ 8MiB
- 집계 단위: 시간 샘플 → process-stage 중앙값 → 독립 프로세스 반복 → GPU 환경
  - 프로세스를 여러 번 반복해도 GPU 환경 표본은 1개로 셈
- 주요 단계: `after_cuda_init`, `after_model_load`, `after_first_forward`, `final_idle`
  - 보조 단계: `after_first_inference`, `after_warmup3`, `after_repeat_block0~2`, `after_performance5`

### 1-3. 가설

| # | 가설 |
|---|---|
| H0 (설계 문서에서 추가) | 계측 자체(pre-init API 호출, 50ms 샘플링, 조회 순서)가 측정 대상을 바꾸는가 |
| H1 | 같은 GPU·환경에서 새 프로세스로 반복해도 outside가 안정적인가 |
| H2 | 모델·배치·길이보다 GPU·환경의 영향이 큰가 |
| H3 | 초기화 직후부터 일정한가, 아니면 사용한 연산에 따라 단계적으로 증가하는가 |
| H4 | 소프트웨어·backend·모듈 로딩이 영향을 주는가 |
| H5 | GPU 특성(용량·아키텍처·SM 등)이 GPU 간 차이를 설명하는가 |
| H6 | 미측정 GPU를 예측 시점 정보만으로 실용적으로 추정할 수 있는가 |

### 1-4. 판정 기준
- 원래 기준 `outside-lab/criteria.json` (`outside-pilot-20261103-v1`, 결과 확인 후 변경 금지)
  - 반복 범위·의미 있는 차이 16MiB, 확인 실험 10회 이상
  - 향후 예측 목표: held-GPU MAE 32 / p95 64 / 최대 과소추정 128MiB, 용량 판정 보류 ±256MiB
- 사용자 실용 기준 `outside-lab/practical-outside-criteria.json`: 차이 **≤ 100MiB면 거의 고정**
  - 앞선 결과를 본 뒤 정한 기준. 원래 기준과 함께 보고하며 통계적 동등성 주장이 아님
- 두 문서 모두 시간 샘플로 CI나 p값을 만들지 않음

### 1-5. 기간과 환경
- 기간: 2026-11-03(설계·smoke·core-01) ~ 2026-11-10(로컬 마무리, g5 실행·회수)
- 로컬: RTX 3090 24GiB, driver 570.211.01, 이미지 `cloudgpufit-bench:cu124-v1` (`sha256:457da255…`, torch 2.6.0+cu124, transformers 4.52.4, Python 3.11.11)
  - probe.py SHA256 `7df9a02c…`와 모델 revision 고정. worker마다 새 프로세스 (11-08 이후는 새 컨테이너)
- AWS: g6.xlarge (L4, 11-08), g5.xlarge (A10G, 11-10). 둘 다 us-east-1, driver 595.112.07

---

## 2. 실험 목록

### 요약표

| suite | 날짜 (KST) | GPU | 설계 | worker | 결과 |
|---|---|---|---|---:|---|
| smoke-01 / 02 | 11-03 | 3090 | init smoke | 1 / 1 | 01 실패(UUID 형식 불일치) → probe 수정 → 02 성공 |
| core-01 | 11-03 | 3090 | 10절차 × 3반복 | 30 | init 254. 연산별 272/258/312, 혼합 314 |
| models-01 | 11-04 | 3090 | 3모델 × 2이력 × 3반복 | 18 | 254→296→312 (ResNet 314) |
| large-01 | 11-04 | 3090 | 8모델 × 2이력 × 3반복 | 48 | 전 모델 312 (Vision 314) |
| qwen-smoke-01 | 11-08 | 3090 | Qwen FP16 smoke | 1 | 실패(nonfinite logits) |
| qwen-bf16-smoke-01 / qwen-bf16-01 | 11-08 | 3090 | 2모델 × 2이력 × 3반복 | 1 / 12 | 254→296→312 |
| loading-sensitivity-01 | 11-08 | 3090 | 3블록 × LAZY/EAGER × 2이력 | 12 | +728MiB (EAGER 1040) |
| loading-confirmation-01 | 11-08 | 3090 | 10블록 × 2모드 | 20 | +728MiB가 10블록 모두에서 재현 |
| qwen-loading-01 | 11-08 | 3090 | 3블록 × 2모드 | 6 | +754MiB (EAGER 1066) |
| qwen-attention-01 | 11-08 | 3090 | 3블록 × eager/SDPA (+진단 2) | 6 | 차이 0 |
| qwen-shape-01 | 11-08 | 3090 | 3블록 × 5형상 | 15 | 차이 0 |
| software-s1~s4 | 11-08 | 3090 | 각 3블록 × {S0,Sx} × 2모델 | 48 | 차이 0 |
| local-closeout-20261110 | 11-10 | 3090 | 5조건 × 1회 + 전체 재집계 | 5 | 312. 상수 최대오차 2/26MiB |
| aws g6.xlarge 파일럿 | 11-08 | L4 | 3블록 × {S0,S4} × 2모델 | 12 | 186→216→232 |
| aws g5.xlarge | 11-10 | A10G | 3블록 × {S0,S4} × 2모델 | 12 | 12/12 exit0. **outside 미분석** |

### 2-1. smoke-01 / smoke-02 (11-03)
- 실행: `outside-lab/container_run.py smoke` (`probe.py --scenario init`)
- smoke-01 (21:12): `RuntimeError: CUDA/NVML UUID mismatch`
  - 원인은 torch와 NVML의 UUID 표기 형식 차이
- 이후 probe에 `canonical_gpu_uuid` 정규화를 추가함. 수정 전 파일은 `outside-lab/probe.py.before-uuid-fix`로 보존
- smoke-02 (21:19): status ok
- 이후 모든 suite는 수정 후 probe(`7df9a02c…`)를 사용함

### 2-2. core-01 (11-03 21:20~21:29) — H0, H1, H3
- 설계: 10절차 × 새 프로세스 3회 = 30 job, 반복 블록 안 순서는 seed 20261103으로 섞음
  - 절차: init 대조 / pre-init API 호출 / gemm·conv·eager attention 단독 / 혼합 순서 3종 / gemm endpoint 샘플링 / gemm NTN 조회
- 실행: `container_run.py pilot --tier core` → 컨테이너 안에서 `pilot.py --execute` → `probe.py`
  - 로그 `core-01.log`, 아카이브 `outside-core-01.tar.gz`
- 환경: CUDA_MODULE_LOADING 미설정. 실제 late CUDA API 모드는 LAZY
- 결과(stable process 중앙값, 반복 범위 전부 0MiB, plateau 미달 0)

| 절차 | 첫 호출 후 → final_idle outside |
|---|---|
| init / init_api_probe | 254 → 254 |
| gemm (TNT / NTN / endpoint) | 272 (세 변형 동일) |
| conv | 258 |
| attention (eager) | 312 |
| gemm→conv→attention | 272 → 274 → 314 |
| conv→attention→gemm | 258 → 314 → 314 |
| attention→gemm→conv | 312 → 312 → 314 |

- 해석
  - API 호출, 샘플링 방식, 조회 순서에 따른 차이는 없음 (H0)
  - 연산별 증가분은 단순 합산되지 않음 (H3)

### 2-3. models-01 (11-04 09:08~09:16) — H1–H3
- 설계: GPT-2 / ResNet-50 / T5-small, FP16 기본 조건 × {direct, performance_first} × 3회 = 18
- 실행: `container_run.py pilot --tier models --cache …`. 로그 `models-01.log`, 아카이브 `outside-models-01.tar.gz`
- 결과: 세 모델 모두 254 → 296 → 312MiB (ResNet-50만 첫 forward 이후 314). 이력 차이 0, 반복 범위 0

### 2-4. large-01 (11-04 09:31~10:02) — H2
- 설계: FP16, B1, eager, TF32=false 조건의 8모델 × 2이력 × 3회 = 48 (`outside-lab/large-plan.json`, timeout 600s)
  - 모델: GPT-2 / medium / XL, T5-small / base / 3B (encoder 256, 생성 64), ResNet-50, ViT-large (224×224)
- 실행: `outside-lab/container_run_large.py pilot`. 로그 `large-01.log`, 아카이브 `outside-large-01.tar.gz`
- 결과: 텍스트 모델 전부 254 → 296 → 312MiB, Vision 2종 314MiB, 반복 그룹 200개 범위 전부 0

### 2-5. Qwen 확장: qwen-smoke / bf16 (11-05 준비, 11-08 실행) — H2, H4
- 준비 (11-05)
  - `CloudGPUFit_outside_qwen_extension_20261105.zip`: `container_run_qwen.py`, `qwen-plan.json`, `qwen-revisions.json`, 테스트 포함
  - Qwen2.5-7B revision `272aa6d3…` 다운로드 약 43분 (`qwen-download.log`)
- qwen-smoke-01 (11-08 15:31, FP16): `first_forward_finite_probe`에서 `nonfinite logits`로 실패
  - 원본은 보존. 재시도나 정밀도 변경으로 감추지 않음
  - 실패 이전 관측은 `qwen_fp16_failed_partial.csv`로 따로 보관
- qwen-weight-check-01: CPU에서 가중치 검사
  - BF16 텐서 339개, 비유한 값 없음, FP16 범위 초과 없음, 최대 |w| 414
  - 실패 원인은 확정하지 못함
- **[주의]** QWEN_README의 FP16 pilot(`results/qwen-01`)은 실행되지 않음 (해당 디렉터리 없음)
- qwen-bf16-smoke-01 (15:44): ok
- qwen-bf16-01 (15:46~15:55)
  - 구성: GPT-2·Qwen BF16 × 2이력 × 3회 = 12, LAZY 명시
  - 실행: `container_run_qwen_bf16.py pilot --env CUDA_MODULE_LOADING=LAZY`. 로그 `qwen-bf16-01.log`
- 결과
  - 12/12 exit0, PID 12개
  - 두 모델 모두 254 → 296 → 312MiB, 반복 그룹 50개 범위 0
  - 이력 72쌍 차이 0
  - large-01 GPT-2 FP16과 비교한 공통 75쌍 차이 0 (날짜가 다른 suite 간 비교)
  - Qwen reserved 14,878MiB, NVML 15,190MiB (각각의 최대값을 기술한 것이며 outside 계산에 쓰지 않음)
- 품질: raw 7,802개, 예상 preinit 결측 504개, running 제외 2개. 기존 분석 240행과 독립 감사가 일치

### 2-6. loading-sensitivity-01 (11-08 16:26~16:33) — H4
- 설계: `outside-lab/LOADING_SENSITIVITY_DESIGN_KO.md`
  - GPT-2 BF16, 3블록 × {LAZY, EAGER} × {direct, performance_first}
  - 컨테이너 6개, worker 12개, seed 20261108
  - 바꾼 것은 CUDA_MODULE_LOADING 하나 (DATA_LOADING은 미설정)
- 실행: `run-loading-sensitivity-01.sh` → `prepare_loading_sensitivity.py --execute`
  - 종료코드 0
  - **[주의]** 최상위 `loading-sensitivity-01.log`는 0바이트. 실제 로그는 `results/loading-sensitivity-01/00.log~05.log`에 있음
- 결과 (MiB)

| 단계 | LAZY | EAGER | 차이 |
|---|---:|---:|---:|
| after_cuda_init | 254 | 824 | +570 |
| after_model_load | 296 | 824 | +528 |
| after_first_forward ~ final_idle | 312 | 1040 | +728 |

- stable reserved와 allocated는 두 모드에서 같았고, NVML만 달랐음
- 품질: raw 5,970개, 추가 제외 5개 (query 4, bracket 1)

### 2-7. loading-confirmation-01 (11-08 17:11~17:21) — H4 확인
- 설계: GPT-2 BF16 direct, 10블록 × 2모드 = 20 worker/컨테이너
  - 순서는 LAZY→EAGER 5블록, EAGER→LAZY 5블록으로 균형 (seed 20261109)
- 실행: `run-loading-confirmation-01.sh` → `run_loading_confirmation.py --execute`. 종료코드 0
- 결과
  - LAZY 254/296/312, EAGER 824/824/1040
  - 첫 forward 이후 차이 +728MiB가 10블록 모두에서 같음
  - mode/stage 24그룹 범위 0
- 품질: raw 8,879개, 추가 제외 3개 (query > 5ms)
- 탐색 결과를 본 뒤 정한 확인 설계이므로 완전한 사전등록이 아님 (보고서에 명시)

### 2-8. qwen-loading-01 (11-08 17:32~17:38) — H4
- 설계: Qwen BF16 direct, 3블록 × 2모드 = 6 (seed 20261110, 완전 균형 아님)
- 실행: `run-qwen-loading-01.sh` → `run_qwen_loading.py --execute`. 종료코드 0
- 결과
  - LAZY 254/296/312, EAGER 824/824/1066
  - 첫 forward 이후 차이 +754MiB가 3블록 모두에서 같음
- GPT-2 EAGER 1040과 26MiB 차이가 남. suite가 달라 모델 효과로 확정하지 않음
- 품질: raw 4,310개, 추가 제외 1개

### 2-9. qwen-attention-01 (11-08 18:03~18:10) — H4
- 설계: Qwen BF16 LAZY direct, `--attention eager` 대 `sdpa`, 3블록 × 2 = 6 (seed 20261111)
  - 측정 후 CPU profiler 진단을 attention별로 1회씩(총 2회) 별도 실행
- 실행: `run-qwen-attention-01.sh` → `run_qwen_attention.py --execute` (진단은 `diagnose_attention_backend.py`)
- 결과
  - 두 attention 모두 254 → 296 → 312MiB
  - 추론 이후 42개 process-stage 전부 312, 블록 차이 0
- 진단
  - SDPA: `aten::_scaled_dot_product_flash_attention`이 첫 forward에서 28회, generate64에서 1792회
  - eager: softmax가 28회 / 1792회
  - 별도 프로세스에서 측정했으므로 본 측정 worker의 각 호출을 직접 추적한 것은 아님

### 2-10. qwen-shape-01 (11-08 18:31~18:59) — H2
- 설계: Qwen BF16 LAZY eager direct, 3블록 × 5조건 = 15 (seed 20261112)
  - 조건: base B1/T256/N64, b16, t2048, n512, b4t1024n256 (복합)
- 실행: `run-qwen-shape-01.sh` → `run_qwen_shape.py --execute`. 종료코드 0
- 결과
  - 모든 조건 254 → 296 → 312, 조건과 base의 차이 전부 0MiB
  - running 범위 296~312
  - reserved 최대: base 14,878 / n512 14,916 / t2048 17,938 / b16 19,832 / b4t1024n256 20,686MiB
  - 워크로드 메모리가 약 5.8GiB 늘어도 outside는 변하지 않음
- device used − process NVML = 전 샘플 448.9MiB (nvidia-smi FB Reserved 442MiB에 대응)
- 품질: raw 24,618개, 제외 13개 (모두 running)

### 2-11. software-s1 ~ s4 (11-08 19:38~21:03) — H4
- 설계: `outside-lab/SOFTWARE_DESIGN_KO.md`
  - 각 변형마다 3블록 × {S0, Sx} × {GPT-2, Qwen} BF16 LAZY = 12, 총 48
  - S0는 매 suite의 동시 대조군

| 변형 | 바뀐 것 | torch / runtime (실측) | cuBLAS / cuDNN | Sx−S0 (주요 42쌍) |
|---|---|---|---|---:|
| S1 | base 이미지 (python slim, conda 없음) | 2.6.0+cu124 / 12.4 | 12.4.5 / 9.1.0 | 0 |
| S2 | CUDA wheel 묶음 | 2.6.0+cu126 / 12.6 | 12.6.4 / 9.5.1 | 0 |
| S3 | torch 버전 | 2.5.1+cu124 / 12.4 | 12.4.5 / 9.1.0 | 0 |
| S4 | 최신 스택 | 2.8.0+cu128 / 12.8 | 12.8.4 / 9.10.2 | 0 |

- 실행
  - S1: 빌드한 이미지 사용
  - S2~S4: `outside-lab/images/install_variant.sh`로 호스트 `pkgs/sX`에 설치하고, python slim 이미지에 읽기 전용으로 마운트
  - 그다음 `run_software.py Sx --execute`, `analyse_software.py Sx` 순서로 실행
  - S3·S4는 `run-software-s3-s4.sh` / `run-software-s4.sh`로 연쇄 실행 (로그 `software-s3-s4.log`, `software-s4-chain.log`)
- 결과
  - 모든 변형 254 → 296 → 312, 반복 범위 0
  - device gap은 전 suite에서 448.9MiB
- **[주의]**
  - S2 첫 설치에는 CUDA13 패키지가 섞여 폐기 후 재설치함
  - S4 첫 설치는 cross-device 오류로 재설치함 (둘 다 측정 전)
  - S1 실행 후 `run_software.py`에 마운트 모드를 추가함 → S1 분석을 다시 돌리면 source hash가 맞지 않음
  - 설치 디렉터리 `pkgs/s2~s4`는 측정 후 삭제했고, 설치 로그만 남음

### 2-12. local-closeout-20261110 (11-10 18:49~18:56)
- 설계: `outside-lab/LOCAL_CLOSEOUT_DESIGN_KO.md`
  - Qwen BF16 LAZY direct, 5조건 × 1회 (seed 20261110): eager/base, sdpa/base, sdpa/b16, sdpa/t2048, sdpa/n512
  - 기존 전체 raw 재집계 포함
- 실행: `run-local-closeout-20261110.sh` → `run_local_closeout.py --execute`. 감사는 `audit_local_closeout.py`
- 새 5회 결과
  - 모두 첫 forward 이후 312MiB
  - 11-08 같은 기본형상(qwen-attention-01)과의 차이 0
  - 조건별 1회이므로 반복 안정성은 주장하지 않음
- 전체 재집계: 로컬에서 성공한 model worker 190개, 추론 후 stable process-stage 1,375개

| 모드 | 고정 상수 | 프로세스 | MAE | p95 | 최대 오차 / 과소추정 | 100MiB 초과 |
|---|---:|---:|---:|---:|---:|---:|
| LAZY | 312 | 171 | 0.211 | 2 | 2 (Vision 314) | 0 |
| EAGER | 1040 | 19 | 4.105 | 26 | 26 (Qwen 1066) | 0 |

- 상수는 이전 GPT-2 결과에서 정해 두었고 재적합하지 않음. 다만 이미 본 데이터에 대한 사후 기술 평가임
- 품질: 새 raw 6,253개, 제외 3개 (n512 실행 중 query > 5ms, 최대 179.3ms)

### 2-13. AWS g6.xlarge (L4) 파일럿 (11-08, UTC 12:54~13:13)
- 번들: `aws-bundle.tar.gz` (`setup.sh`, `run.py`, Dockerfile.s0/s4, lab/)
- 실행: 인스턴스에서 `./setup.sh` → `python3 run.py`
- 설계: 3블록 × {S0 2.6.0+cu124, S4 2.8.0+cu128} × {GPT-2, Qwen} BF16 LAZY eager direct = 12 (seed 20261201)
- 환경: us-east-1, AMI `ami-0d220f0a0f14edbef`, driver 595.112.07, L4 23,034MiB, FB Reserved 469MiB
- 결과 (`results/aws-g6-xlarge-01/REPORT_KO.md`)
  - 12/12 exit0
  - outside 186 → 216 → 232MiB, S4−S0 0, 모델 차이 0, 반복 범위 0
  - 기존 AWS L4 값 232(226–236)와 일치
  - device gap 476.9MiB
  - 품질 필터 제외 2샘플
- 분석: `python3 outside-lab/analyse_aws.py aws-results/extracted/g6.xlarge-20261108T125436Z` (stdout만 출력)

### 2-14. AWS g5.xlarge (A10G) (11-10, UTC 10:41~10:56)
- 패키지: `cloudgpufit-aws-manual-20261110.zip` (= `aws-manual-20261110/`)
  - 사용자가 직접 실행: `verify_package.py` → `nohup sudo -n bash run_all.sh g5.xlarge-01`
- 설계: g6와 같음 (12 worker, timeout 1200s)
  - 상태, UUID, 실제 LAZY, 케이스 검사를 강화함
- 환경 (execution 아카이브의 env 기록): us-east-1, AMI `ami-0bc11c94e8c4894e4`, driver 595.112.07, A10G 23,028MiB, FB Reserved 439MiB
- 회수 검증 (`aws-results/g5-xlarge-01/retrieval-verification.json`): suite complete, 12/12 exit0, child ok 12, 고유 PID 12, samples 파일 12개, `run.exitcode` = 0
- **outside 분석은 수행되지 않음**
  - `"outside_quality_analysis_pending": true`
  - 압축 해제 디렉터리, 분석 CSV, REPORT가 없음

---

## 3. 코드 설명

### 3-1. 공통 구조
- 측정: `probe.py` 하나를 해시로 고정하고, 모든 실행기가 이를 컨테이너 안에서 호출함
- 11-08 이후 실행기(`run_*.py`, `prepare_loading_sensitivity.py`)의 공통 흐름
  1. 기본은 dry-run(스케줄 JSON 출력)이며 `--execute`일 때만 GPU를 사용함
  2. 이미지 ID, probe hash, source/input hash를 검사함
  3. seed로 블록 순서를 정함
  4. worker마다 `docker run --rm --network none --gpus device=<UUID> --pid=host`를 실행함 (lab은 ro, HF cache는 ro로 마운트)
  5. 종료 후 manifest status와 실제 late CUDA 모드를 검사함
  6. 실패나 timeout이 나면 중단함 (자동 재시도 없음, 자체 생성한 컨테이너만 stop)
  7. `orchestration.json`을 기록함
- 분석: `analyse_probe.py`(suite 공통 요약)와 suite별 독립 분석기(`analyse_*`, `audit_*`)로 이중 검증함

### 3-2. 측정·실행 계층 (`outside-lab/`)

#### probe.py
- 역할: 단일 worker 측정기
- 흐름
  1. NVML 초기화. 다른 PID, MIG, MPS가 있으면 중단
  2. torch import → CUDA init → UUID 정규화 검사
  3. scenario 실행
     - `init` / `gemm` / `conv` / `attention` / `ops`: first, warmup3, repeat 10×3, release
     - `model`: 로딩 → 첫 forward(유한성 확인) → generate → warmup → (performance5) → 반복 블록 3
  4. checkpoint마다 sync 후 stable 21회, action 중에는 50ms running 샘플. 샘플마다 TNT/NTN 조회, `outside_bytes`, bracket, device memory 기록
  5. 마지막에 late CUDA 진단(실제 module loading 모드, runtime) 기록
- 입력 → 출력: CLI 인자(`--scenario`, `--plan`, `--case-id`, `--attention`, `--history` 등) → `manifest.json`, `samples.jsonl`, `events.json`
- `probe.py.before-uuid-fix`: smoke-01 당시 수정 전 버전 (보존용)

#### pilot.py
- 역할: suite 스케줄러
  - core: 10절차
  - models: case × {direct, performance_first}
  - 블록마다 seed 20261103으로 셔플
- 동작: dry-run이면 스케줄과 hash 출력. `--execute`면 job마다 새 Python 프로세스로 probe 실행, 실패 시 중단
- 출력: `suite-manifest.json` + run 디렉터리

#### preflight.py / prepare_cache.py / reanalyse.py
- preflight: 호스트를 읽기 전용으로 점검 (nvidia-smi, Docker 이미지, MPS). torch는 import하지 않음
- prepare_cache: 고정 revision HF 캐시 검사(`--check-only`) 또는 다운로드 (CPU 전용)
- reanalyse: 기존 AWS bench-v3 JSON을 CPU에서 재분석 → `outside-lab/analysis/legacy_*.csv`, `legacy_audit.json`

#### container_run.py / container_run_large.py / container_run_qwen.py / container_run_qwen_bf16.py
- 공통
  - 고정 이미지 래퍼. phase: check, cache-check, cache-download, plan, smoke, pilot
  - 이미지 ID가 다르면 중단
  - smoke와 pilot만 단일 UUID GPU를 노출
  - `--env` 허용 목록 사용 (CUDA_MODULE_LOADING 등)
- 파일별 차이와 사용 suite

| 파일 | 차이 | 사용 suite |
|---|---|---|
| container_run.py | 기본 래퍼 | smoke-01/02, core-01, models-01 |
| container_run_large.py | large-plan, timeout 600, probe hash 검사 추가 | large-01 |
| container_run_qwen.py | 별도 revision 파일, 디스크 25GiB 가드, 실제 모델 smoke, timeout 900 | qwen-smoke-01 (FP16) |
| container_run_qwen_bf16.py | qwen과 같고 smoke case와 plan만 BF16 | qwen-bf16-smoke/01. loading-sensitivity에서 child로도 사용 |

#### prepare_loading_sensitivity.py
- 역할: LAZY/EAGER 탐색 실행기
  - 3블록 × 2모드 (seed 20261108)
  - 각 child는 `container_run_qwen_bf16.py pilot --plan loading-sensitivity-plan.json --repeats 1 --env CUDA_MODULE_LOADING=<mode>`로 2이력을 실행
- 출력: `results/loading-sensitivity-01/`

#### run_loading_confirmation.py / run_qwen_loading.py / run_qwen_attention.py / run_qwen_shape.py / run_local_closeout.py
- 구조는 3-1의 공통 흐름과 같고 seed, 조건, 출력 경로만 다름
  - confirmation: 10블록 균형, seed 20261109
  - qwen_loading: seed 20261110
  - qwen_attention: seed 20261111. 측정 후 진단 컨테이너 2개를 추가 실행
  - qwen_shape: `qwen-shape-plan.json`, seed 20261112
  - local_closeout: `local-closeout-plan.json`, seed 20261110. docstring은 shape 실행기 문구가 복사된 채 남아 있음
- 출력: 각각 `results/<suite>/`

#### run_software.py
- 역할: S0 대 Sx(S1~S4) 비교
  - `images/variants.json`의 mode에 따라 빌드 이미지 또는 패키지 마운트로 실행
  - 변형 lock 파일과 image inspect 저장
- 출력: `results/software-sX-01/`

#### diagnose_attention_backend.py
- 역할: 별도 프로세스에서 Qwen first_forward / generate64를 CPU profiler로 관찰
  - aten attention operator 이름과 횟수, Transformers 분기 소스 hash를 `diagnostic.json`에 저장
- 메모리 값은 outside 분석에 쓰지 않음

#### images/ (Dockerfile.variant, install_variant.sh, variants.json)
- Dockerfile.variant: S1 빌드용 (python 3.11.11 slim + 고정 패키지)
- install_variant.sh: S2~S4 패키지를 호스트 디렉터리에 `pip --target`으로 설치하고 lock 생성 (디스크 절약)
- variants.json: 변형별 torch, index, mode, base digest

### 3-3. 분석·감사 계층 (`outside-lab/`)

#### analyse_probe.py
- 역할: suite 공통 요약기
  - 샘플별 품질 플래그
  - process-stage 통계 (clean 수, 중앙값, last10 span)
  - suite 내부 반복 그룹
- 출력: `process_stage.csv`, `quality_flags.csv`, `repeat_groups.csv`, `audit.json`
- 한계: child suite 단위로 묶기 때문에, 블록마다 컨테이너가 나뉜 실험은 suite별 분석기가 따로 필요함

#### audit_qwen_bf16.py
- 역할: qwen-bf16-01 독립 감사
  - raw 재계산, 품질과 provenance 검사
  - analyse_probe 결과와 대조
  - FP16 실패 partial 분리
- 출력: `results/qwen-bf16-01-independent-audit/`
- 기타
  - 다른 분석기들이 이 파일의 `ROOT`, `sha`, `flags`, `audit`를 재사용함
  - 출력 디렉터리가 이미 있으면 중단함

#### analyse_loading_sensitivity.py / analyse_loading_confirmation.py / analyse_qwen_loading.py / analyse_qwen_attention.py
- 공통 틀
  - orchestration의 block, mode, history를 보존해 블록 안에서 대응 비교
  - 출력: paired_process_stage, repeat_summary, block_mode_comparison, mode_effect_summary, quality_audit, run_provenance, evidence_sha256, summary.json
- 파일별 추가 출력
  - sensitivity: history_comparison
  - confirmation: pilot_confirmation_comparison, input_verification
  - qwen_loading: cross_model_comparison
  - attention: diagnostic_attention_operators, practical_assessment

#### analyse_qwen_shape.py / analyse_software.py
- shape: 블록 안 (조건 − base) 비교, 100MiB 판정, device_gap
  - 같은 디렉터리 모듈을 import하므로 `python3 -I`로는 실행 불가
- software: 인자 Sx → (Sx − S0) 블록·모델·단계별 비교 → `results/software-sX-01-analysis/`

#### analyse_aws.py
- 역할: AWS 간이 분석기 (약 20줄)
  - 압축 해제 디렉터리의 samples.jsonl에 품질 필터 적용
  - (이미지, 모델, 단계)별 블록 중앙값과 범위, S4−S0, device gap 계산
- **출력은 stdout뿐**이며 파일을 남기지 않음

#### audit_local_closeout.py
- 역할: 로컬에서 성공한 모든 model worker 재집계 + 새 5회 감사
  - LAZY 312 / EAGER 1040 상수 적용 오차
  - 날짜 간 비교
- 출력: `results/local-closeout-20261110-analysis/`

#### test_protocol.py / test_large_launcher.py / test_qwen_launcher.py
- CPU 전용 단위 테스트
  - protocol 6개: PID 격리, 음수 clamp 금지, pre-init API 미호출, 무작위 블록, 프로세스 단위 그룹
  - large 4개: plan 전달, 48개 스케줄
  - qwen 4개: 디스크 가드, pin 일치, 스케줄

### 3-4. 최상위 스크립트

| 파일 | 내용 |
|---|---|
| `run-loading-sensitivity-01.sh` | `prepare_loading_sensitivity.py --execute --output results/loading-sensitivity-01` 실행, `.log`와 `.exitcode` 기록 |
| `run-loading-confirmation-01.sh` | `run_loading_confirmation.py --execute` 실행, 로그와 종료코드 기록 |
| `run-qwen-loading-01.sh` | `run_qwen_loading.py --execute` |
| `run-qwen-attention-01.sh` | `run_qwen_attention.py --execute` |
| `run-qwen-shape-01.sh` | `run_qwen_shape.py --execute` |
| `run-software-s3-s4.sh` | S3·S4 연쇄 실행: install_variant → run_software → analyse_software → pkgs 삭제 (`set -e`) |
| `run-software-s4.sh` | S4 단독 재실행 (S4 첫 설치 실패 이후) |
| `run-local-closeout-20261110.sh` | `run_local_closeout.py --execute` |

- 모든 셸 래퍼는 tmux 지속 세션 안에서 실행하도록 설계됨. 래퍼 자체에는 tmux 명령이 없음

### 3-5. AWS 패키지

#### aws-bundle/ (`aws-bundle.tar.gz`, 10개 파일, g6 파일럿용)
- `setup.sh`
  1. nvidia-smi와 Docker 정보, IMDSv2 인스턴스 메타데이터 기록
  2. S0/S4 이미지 빌드 (base digest 고정)
  3. VRAM < 22,000MiB이면 GPT-2만 실행하는 plan 사용
  4. 고정 revision 캐시 다운로드
- `run.py`
  1. probe hash 확인, 다른 GPU 프로세스가 없는지 확인
  2. 3블록 셔플 (seed 20261201)
  3. worker마다 새 컨테이너 실행 (LAZY, timeout 1200)
  4. `results/<label>-<UTC>.tar.gz` 생성
- 자동 중지(shutdown, StopInstances) 코드는 없음. 이전 요약의 "자동 shutdown" 문구는 정정됨

#### aws-manual-20261110/ (`cloudgpufit-aws-manual-20261110.zip`, g5용)
- `README_KO.md`: EC2 설정(us-east-1, g5.xlarge, 150GiB gp3), 전송, 실행, 회수, 수동 Stop 절차
- `verify_package.py`: `package-manifest.json`의 SHA256 대조
- `run_all.sh`
  - 실행 순서: verify → setup → dry-run 스케줄 → run
  - 재실행 방지(`.run-started`)
  - 종료 시 `run.exitcode`와 execution 아카이브 생성
- `select_ami.sh`: 사용자가 지정한 AMI를 읽기 전용으로 조회함. 다른 AMI로 자동 대체하지 않음
- `run.py` / `setup.sh`: aws-bundle 대비 강화된 점
  - 단일 GPU 확인, label 검증
  - 매 worker 전 입력 hash 재검사
  - manifest status, UUID, LAZY, dtype, attention 검사

### 3-6. 기타
- `handoff/`: 별도 후속 요청 (inuos 작성, 11-10)
  - 내용: xMem과 PyTorch allocator의 reserved 오차 원인을 분리하기 위한 GPU allocator trace 수집 요청
  - 파일: GPU_COLLECTION_REQUEST.md, gpu_collector_reference.py, common_manifest.json, xmem-allocator-lock.json, SHA256SUMS
  - outside suite가 아니며, 이 폴더 안에서 실행된 기록은 **[미확인]**
- `pkgs/`: S2~S4 pip 설치 로그만 남음 (`install-s2/s3/s4.log`)

---

## 4. 결과 요약

### 4-1. 단계별 outside (3090, LAZY, 모든 모델 공통)

| 단계 | outside |
|---|---:|
| CUDA 초기화 후 | 254MiB |
| 모델 로딩 후 | 296MiB |
| 첫 forward 이후 ~ 종료 | **312MiB** (Vision 314) |

### 4-2. 요인별 차이 (첫 forward 이후)

| 요인 | 차이 | 100MiB 기준 |
|---|---:|---|
| 모델 크기 GPT-2(124M) ~ Qwen 7B, T5-3B | 0 | 고정 |
| Vision (ResNet, ViT) | +2 | 고정 |
| FP16 ↔ BF16 (서로 다른 suite) | 0 | 고정 |
| direct ↔ performance_first | 0 | 고정 |
| 배치 16 / 입력 2048 / 생성 512 / 복합 | 0 | 고정 |
| attention eager ↔ SDPA | 0 | 고정 |
| base 이미지, torch 2.5.1~2.8.0, CUDA 12.4~12.8 (S1~S4) | 0 | 고정 |
| 날짜 (11-08 ↔ 11-10) | 0 | 고정 |
| **CUDA_MODULE_LOADING LAZY → EAGER** | **+728 (GPT-2) / +754 (Qwen)** | **고정 아님** |
| EAGER에서 GPT-2 → Qwen | +26 | 고정 |
| GPU 간 (L4 232 / 3090 312 / 기존 A10G 308, T4 130, L40S 510) | 최대 380 | 고정 아님 (driver 등이 함께 달라 인과 확정 불가) |

### 4-3. 기타 관측
- 반복 재현성: 모든 suite에서 같은 조건의 새 프로세스 간 범위 0MiB
- device used − process NVML: 3090 448.9MiB, L4 476.9MiB로 일정
  - 이는 드라이버·펌웨어 예약분이며 outside와는 별개
  - 용량 판정 시 명목 total에서 빼야 하는지 검토 필요 (기존 벤치 guard는 명목 total 사용)
- 상수 적용 오차 (사후 기술): LAZY 312는 최대 2MiB, EAGER 1040은 최대 26MiB, 100MiB 초과 0

---

## 5. 가설별 판정

| 가설 | 근거 | 판정 |
|---|---|---|
| H0 계측 영향 | init_api_probe, gemm endpoint/NTN 모두 대조군과 같음 | 영향 없음 (core-01 범위) |
| H1 반복 안정성 | 전 suite 반복 범위 0, 10블록 확인 | 부합 |
| H2 워크로드 대 환경 | 워크로드 0~26MiB ≪ 모듈 로딩 728~754, GPU 간 130~510 | 부합 |
| H3 일정 대 단계 증가 | 254→296→312 단계 증가 후 plateau. EAGER는 824→1040 | 부분 상이 (첫 추론 이후 값을 사용해야 함) |
| H4 소프트웨어 영향 | 모듈 로딩만 큼. attention, 정밀도, 이미지, torch, CUDA는 0 | 요인별로 다름 |
| H5 GPU 특성 | 3090 + L4 신규 1종뿐. g5 미분석 | 보류 (AWS 표본 부족) |
| H6 미측정 GPU 예측 | holdout 평가 없음 | 보류 |

- 종합
  - 같은 GPU·driver·LAZY 조건에서 조사한 범위의 추론 후 outside는 312~314MiB였음
  - 미검증 조건까지 고정된다고 보장하지는 않음
- 제품 측면
  - xMem과 분리해 GPU별 outside를 쓰는 구조가 타당함
  - 초기화 직후 값(254)이 아니라 첫 추론 이후 값(312)을 써야 함
  - 모듈 로딩 정책을 조건으로 명시해야 함

---

## 6. 결과 파일 위치

| 구분 | 경로 |
|---|---|
| 요약·인계 | `NOTION_SUMMARY_20261108.md`, `HANDOFF_KO_20261108.md`, `AWS_NEXT_STEP_20261110.md` |
| 설계 | `outside-lab/EXPERIMENT_DESIGN.md`, `outside-lab/*_DESIGN_KO.md`, `outside-lab/LARGE_MODELS_README.md`, `outside-lab/QWEN_README.md` |
| 기준 | `outside-lab/criteria.json`, `outside-lab/practical-outside-criteria.json` |
| plan·revision | `outside-lab/*-plan.json`, `revisions.json`, `qwen-revisions.json`, `*-schedule.json`, 최상위 `*-dry-run.json`, `qwen-bf16-schedule.json` |
| 기존 AWS 재분석 | `outside-lab/analysis/` |
| raw (11-03~04) | `results/{smoke-01,smoke-02,core-01,models-01,large-01}/` + 아카이브 `outside-{core,models,large}-01.tar.gz` |
| raw (11-08~10) | `results/<suite>/` (`orchestration.json`, child `session/*/manifest.json`, `samples.jsonl`, `events.json`, `container-inspect.json`) |
| 공통 분석 | `results/<suite>-analysis/` (core, models, large, qwen-bf16), `results/<suite>-probe-analysis/` |
| 보고서 | `results/qwen-bf16-01-independent-audit/REPORT_KO.md`, `results/loading-sensitivity-01-analysis/REPORT_KO.md`, `results/loading-confirmation-01-analysis/REPORT_KO.md`, `results/qwen-loading-01-analysis/REPORT_KO.md`, `results/qwen-attention-01-analysis/REPORT_KO.md`, `results/qwen-shape-01-analysis/REPORT_KO.md`, `results/software-summary-01/REPORT_KO.md`, `results/local-closeout-20261110-analysis/REPORT_KO.md`, `results/aws-g6-xlarge-01/REPORT_KO.md` |
| software 개별 | `results/software-s{1..4}-01-analysis/` (REPORT 없음, summary.json과 CSV만 있음) |
| 실행 로그·종료코드 | 최상위 `*.log`, `*.exitcode` |
| AWS g6 | `aws-results/g6.xlarge-20261108T125436Z.tar.gz`, `aws-results/extracted/…`, `aws-results/{run,setup}.log` |
| AWS g5 | `aws-results/g5-xlarge-01/g5.xlarge-01-20261110T104149Z.tar.gz` (raw, 116 항목), `…-execution-20261110T105641Z.tar.gz` (logs, env, exitcode), `retrieval-verification.json` |
| AWS 패키지 | `aws-bundle/`, `aws-bundle.tar.gz`, `aws-manual-20261110/`, `cloudgpufit-aws-manual-20261110.zip` (+ `.sha256`) |

---

## 7. 한계와 미확인 사항

- **[미확인] g5.xlarge (A10G) 결과는 회수만 되었고 분석되지 않음**
  - `retrieval-verification.json`에 `outside_quality_analysis_pending: true`로 기록됨
  - 압축 해제본, 분석 CSV, REPORT가 없음
  - 기존 A10G(g5.2xlarge) 308MiB와 일치하는지는 아직 알 수 없음
- **[미확인] g6 파일럿 분석 결과는 REPORT 요약(7줄)뿐임**
  - `analyse_aws.py`가 stdout으로만 출력하는 구조라 파생 CSV나 독립 감사 결과가 파일로 남지 않음
- **[주의] AMI 기록 충돌**
  - g6 tar의 기록은 `ami-0d220f0a0f14edbef`
  - 사용자가 이후 지정한 AMI는 `ami-0bc11c94e8c4894e4` (표시 이름은 사용자 설명 기준이며 API로 검증하지 않음)
  - g5 실행은 후자를 사용함
  - 같은 AMI라고 볼 근거가 없으므로 실행별로 구분해야 함
- **[주의] 자동 중지 문구 정정**
  - 이전 요약에는 "자동 shutdown"이라고 적혀 있었으나 실제 코드에는 중지 코드가 없음
  - Stop은 사용자가 콘솔에서 직접 수행
- 단일 GPU 환경 문제
  - 3090 결과는 GPU 하나에서 프로세스만 반복한 것이므로 GPU 일반화의 근거가 아님
  - 독립 GPU 환경 표본은 3090, L4 신규, (A10G 미분석)
- driver 효과 미분리
  - 3090은 570, AWS는 595
  - 같은 GA102 칩인 A10G와 3090을 비교해도 driver 효과를 분리했다고 주장할 수 없음
- 미검증 범위
  - HF transformers 추론만 확인함
  - vLLM 등 다른 프레임워크, transformers 버전, cuDNN benchmark=true, driver A–B–A 비교는 미검증
- 계측 해상도: 50ms 샘플링과 checkpoint 해상도 밖의 transient는 놓칠 수 있음. 모델 로딩 중에는 연속 샘플이 없음
- 사후 기준과 사후 평가
  - 100MiB 실용 기준은 결과를 본 뒤 정함
  - closeout의 상수 오차 평가는 사후 기술 평가이며 holdout 검증이 아님
- 실패 원인 미해명: Qwen FP16 nonfinite logits 원인 미확정. FP16 Qwen pilot(`qwen-01`)은 실행되지 않음
- 순서 균형: 3블록 실험(qwen-loading, qwen-attention, loading-sensitivity)은 순서가 완전히 균형되지 않음
- 재현성 문제
  - S1 분석을 다시 돌리면 source hash가 맞지 않음 (`run_software.py`를 나중에 수정했기 때문)
  - S1 이미지와 `pkgs/s2~s4`는 삭제되어 같은 환경을 다시 만들려면 재설치가 필요함
- `loading-sensitivity-01.log`(최상위)는 0바이트임. 실제 진행 로그는 `results/loading-sensitivity-01/*.log`
- 다음 단계 (문서상 계획이며 미실행)
  - g5 분석
  - g4dn.xlarge (T4) → g6e.xlarge (L40S) → p3 / p4d / p5 순서
  - 이후 leave-one-GPU-out 평가

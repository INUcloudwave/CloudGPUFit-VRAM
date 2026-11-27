# CloudGPUFit outside 분석 — 진행 정리 (2026-11-08)

## 1. 목적

- CloudGPUFit: GPU 대여 전 CPU에서 추론 워크로드의 VRAM을 예측하고 AWS 인스턴스를 추천
  - 최종 추정 = xMem 예측 reserved + **outside**
- **outside 정의**: 같은 조회 구간의 `NVML 해당 프로세스 메모리 − torch.cuda.memory_reserved`
  - allocated 차감 금지, 서로 다른 시점의 최대값끼리 차감 금지, xMem 오차 흡수 금지
  - 음수·결측을 0으로 바꾸지 않음, 다른 프로세스 메모리 제외
- 이번 단계 목표: 학습이 아니라 **outside에 영향을 주는 요인 확인** → 상수 / GPU별 표 / 예측식 중 무엇이 필요한지 판단

### 검증 가설

| # | 가설 |
|---|---|
| H1 | 같은 GPU·환경에서 새 프로세스로 반복해도 outside가 안정적인가 |
| H2 | 모델·배치·길이보다 GPU·환경 영향이 큰가 |
| H3 | 초기화 직후부터 일정한가, 사용한 연산에 따라 단계적으로 증가하는가 |
| H4 | 소프트웨어·backend·모듈 로딩이 영향을 주는가 |
| H5 | GPU 특성(용량·아키텍처·SM 등)이 GPU 간 차이를 설명하는가 |
| H6 | 미측정 GPU를 예측 시점 정보만으로 실용적으로 추정할 수 있는가 |

### 판정 기준

- 사용자 실용 기준: 비교 대상 간 outside 차이 **≤100MiB면 거의 고정**
- 기존 사전 기준(16MiB)은 덮어쓰지 않고 함께 보고

## 2. 기존 AWS 자료 (참고)

| GPU | 인스턴스 | 조건별 outside 중앙값 (범위) |
|---|---|---|
| T4 | g4dn.2xlarge | 130MiB (128–132) |
| A10G | g5.2xlarge | 308MiB (302–312) |
| L4 | g6.xlarge / 2xlarge / 4xlarge | 232MiB (226–236) |
| L40S | g6e.2xlarge | 510MiB (504–514) |

- 342개 워크로드 조건 안에서 GPU별 범위는 2~10MiB
- 한계: 조건 안에서 모델 재사용, 같은 조건의 반복 없음, driver 595 (로컬 3090은 570)

## 3. RTX 3090 실험 환경

- RTX 3090 24GiB, driver 570.211.01, 이미지 `cloudgpufit-bench:cu124-v1` (torch 2.6.0+cu124, transformers 4.52.4)
- probe.py 해시 고정, 모델 revision 고정, worker마다 새 컨테이너·새 프로세스
- 안정 관측: 체크포인트마다 50ms × 21회, 품질 필터(조회 5ms 초과 / reserved 변화 / 다른 PID / 음수 제외), 원본 보존
- 모든 suite를 독립 재계산으로 검증

## 4. 3090에서 수행한 실험

| 실험 | 내용 | worker 수 | 관련 가설 |
|---|---|---:|---|
| core-01 | 초기화, 측정 API 영향, GEMM/conv/attention 단독·혼합 | 30 | H1, H3 |
| models-01 | GPT-2·ResNet-50·T5-small, 실행 이력 2종 | 18 | H1–H3 |
| large-01 | 8개 모델 (GPT-2 ~ T5-3B, ViT), FP16 | 48 | H2 |
| qwen-bf16-01 | Qwen2.5-7B BF16 | 12 | H2, H4 |
| loading-sensitivity / confirmation / qwen-loading | CUDA 모듈 LAZY vs EAGER | 38 | H4 |
| qwen-attention-01 | eager attention vs SDPA(Flash) | 6 | H4 |
| qwen-shape-01 | 배치 16 / 입력 2048 / 생성 512 / 복합 | 15 | H2 |
| software-s1~s4 | 이미지·torch·CUDA 버전 | 48 | H4 |

## 5. 주요 결과

### 5-1. 단계별 outside (LAZY 기준, 모든 모델 공통)

| 단계 | outside |
|---|---:|
| CUDA 초기화 후 | 254MiB |
| 모델 로딩 후 | 296MiB |
| 첫 forward 이후 ~ 종료 | **312MiB** (비전 모델 314) |

- 연산별 초기 증가: GEMM 272 / conv 258 / attention 312 / 혼합 314MiB (증가분을 단순 합산할 수 없음)

### 5-2. 워크로드 변화 (H2)

| 변화 | outside 차이 |
|---|---:|
| 모델 크기 GPT-2(124M) ~ Qwen 7B | 0MiB |
| 비전 모델 (ResNet·ViT) | +2MiB |
| FP16 ↔ BF16 | 0MiB |
| 배치 1→16, 입력 256→2048, 생성 64→512, 복합 | 0MiB |
| EAGER 모드에서 GPT-2 → Qwen | +26MiB |

- qwen-shape-01에서 reserved 최대는 14,878 → 20,686MiB로 늘었지만 outside는 312MiB로 변하지 않음

### 5-3. 소프트웨어·설정 변화 (H4)

| 변화 | outside 차이 | 100MiB 기준 |
|---|---:|---|
| CUDA_MODULE_LOADING LAZY → EAGER | **+728 ~ +754MiB** | 고정 아님 |
| attention eager ↔ SDPA(Flash) | 0MiB | 고정 |
| S1: base 이미지 (python slim, conda 없음) | 0MiB | 고정 |
| S2: CUDA 묶음 cu124 → cu126 (cuDNN 9.1 → 9.5) | 0MiB | 고정 |
| S3: torch 2.6.0 → 2.5.1 | 0MiB | 고정 |
| S4: torch 2.8.0 + CUDA 12.8 (cuDNN 9.10) | 0MiB | 고정 |

- PyTorch는 환경변수가 없으면 LAZY로 동작 → EAGER는 사용자가 따로 설정한 경우에만 해당

### 5-4. 재현성 (H1)

- 모든 실험에서 같은 조건의 새 프로세스 반복 간 범위 0MiB

### 5-5. 드라이버 예약 메모리 (새로 확인)

- `디바이스 used − 프로세스 NVML` = 모든 샘플에서 **448.9MiB**로 일정
  - `nvidia-smi -q`의 FB Memory Reserved(442MiB)에 해당하는 드라이버/펌웨어 예약분
- outside와는 별개의 디바이스 단위 값이며, outside 측정(프로세스 NVML)에는 포함되지 않음
- 기존 AWS 벤치의 NVML peak도 프로세스 기준 값이었음
- 용량 판정 시 시사점: 실제 사용 가능 메모리 ≈ 명목 총량 − 예약분
  - 기존 벤치의 메모리 가드는 명목 total을 사용 → 경계 근처에서 약 0.4GiB 과대평가 가능

## 6. 가설별 판정

| 가설 | 결과 | 판정 |
|---|---|---|
| H1 반복 안정성 | 반복 간 범위 0MiB | ✅ 부합 |
| H2 워크로드 vs 환경 | 워크로드 0~26MiB ≪ 모듈 로딩 728~754MiB, GPU 간 130~510MiB | ✅ 부합 |
| H3 일정 vs 단계 증가 | 254 → 296 → 312 단계 증가 후 고정 | ⚠️ 부분 상이 (첫 추론 이후 plateau 값을 써야 함) |
| H4 소프트웨어 영향 | 모듈 로딩만 큼. attention·정밀도·이미지·torch·CUDA 버전은 0 | ⚠️ 요인별로 다름 |
| H5 GPU 특성 | 3090 단일 GPU로는 검증 불가 | ⏸ AWS 필요 |
| H6 미측정 GPU 예측 | 검증 불가 | ⏸ AWS 필요 |

## 7. 종합 해석

- 같은 GPU·driver·LAZY에서 **조사한 워크로드·소프트웨어 조건의 추론 후 outside는 312~314MiB**였습니다. 미검증 조건까지 고정을 보장하지 않습니다.
- 통제 비교에서 큰 차이가 확인된 요인은 **CUDA 모듈 로딩 설정**입니다. GPU 간 차이도 관측됐지만 driver·환경이 달라 하드웨어만의 인과효과로 확정하지 않습니다.
- 미검증 소프트웨어 후보에는 **driver**, 다른 프레임워크·라이브러리 설정 등이 있습니다. driver가 유일한 원인 후보라는 뜻은 아닙니다. 3090 공유 VM의 driver는 변경하지 않습니다.
- 제품 측면
  - xMem 예측과 분리해 GPU별 outside 값으로 다루는 현재 구조가 타당
  - 조사한 범위에서는 워크로드별 추가 보정 필요성이 작음(사용자100MiB 기준)
  - 초기화 직후 값(254)이 아닌 첫 추론 이후 값(312)을 사용해야 함 (58MiB 과소추정 방지)
  - 드라이버 예약분을 용량에서 차감할지 검토 필요
- 참고: VRAM이 같은 24GB여도 L4 232 / A10G 308 / 3090 312MiB → 용량 비례 아님. A10G와 3090은 같은 GA102 칩이지만 driver가 달라 비교 확정 불가

## 8. 한계

- 3090 결과는 단일 GPU 환경에서 프로세스만 반복한 것 → GPU 일반화 근거 아님
- HF transformers 추론만 확인. vLLM 등 다른 프레임워크, cuDNN benchmark, transformers 버전은 미검증
- 50ms 샘플링 해상도 밖의 순간 변화는 관측 불가

## 9. 다음 단계: AWS

### 목표

- GPU별 plateau outside와 드라이버 예약분 측정 (H5·H6 표본 확보)
- 3090 결과(소프트웨어 영향 없음)가 다른 GPU에서도 성립하는지 S0 vs S4로 확인
- driver 영향 분리: 가능하면 같은 GPU에서 driver만 통제해 비교. AMI 변경은 다른 시스템 요소도 바뀔 수 있으므로 driver 단독 효과로 자동 귀속하지 않음

### 진행 방식

- 번들 `aws-bundle.tar.gz`: probe(해시 고정), plan, Dockerfile 2개(S0 동일 레시피, S4), `setup.sh`, `run.py`
  - 3090에서 end-to-end 검증 완료 (worker 6개 정상, outside 312MiB 일치)
- 인스턴스에서: `./setup.sh` → `python3 run.py` → 결과 `results/*.tar.gz`
  - 3블록 × {S0, S4} × {GPT-2, Qwen 7B} (GPU 메모리 22GB 미만이면 GPT-2만)
  - **정정:** 현재 setup.sh/run.py에는 자동 중지 코드가 없습니다. 결과 회수 후 AWS 콘솔에서 Stop을 명시적으로 수행합니다. 종료·볼륨 삭제는 보존 확인과 별도 승인 후 처리합니다.

### 인스턴스 순서

| 순서 | 인스턴스 | GPU | 목적 |
|---|---|---|---|
| 1 | g6.xlarge | L4 | 파일럿, 기존 L4 값 재현 확인 |
| 2 | g5.xlarge | A10G | 3090과 같은 칩 → driver 효과 분리 |
| 3 | g4dn.xlarge | T4 | Turing 아키텍처 |
| 4 | g6e.xlarge | L40S | Ada, 48GB |
| 이후 | p3 / p4d / p5 | V100 / A100 / H100 | 새 아키텍처 (비용 검토 후) |

- 인스턴스당 약 1시간, 약 $1~3 예상 (리전 요금 확인 필요)
- 파일럿 결과 확인 후 driver 비교와 추가 인스턴스 진행

## 10. 결과 파일 위치 (inugpu)

- 루트: `~/cloudgpufit-outside`
- 인계 노트: `HANDOFF_KO_20261108.md`
- 보고서
  - `results/qwen-bf16-01-independent-audit/REPORT_KO.md`
  - `results/loading-sensitivity-01-analysis/REPORT_KO.md`
  - `results/loading-confirmation-01-analysis/REPORT_KO.md`
  - `results/qwen-loading-01-analysis/REPORT_KO.md`
  - `results/qwen-attention-01-analysis/REPORT_KO.md`
  - `results/qwen-shape-01-analysis/REPORT_KO.md`
  - `results/software-summary-01/REPORT_KO.md`
- 설계 문서: `outside-lab/*_DESIGN_KO.md`
- AWS 번들: `aws-bundle.tar.gz`

## 2026-11-10 로컬 마무리 업데이트

- SDPA큰배치16·입력2048·생성512 + 당일기본형상(eager/SDPA) 5회모두정상, 추론후outside312MiB. 11월8일기본형상대비차이0. 각새조건1회점검이며반복안정성확인으로과장하지않음.
- 기존전체local 성공모델190개/추론후stable1375행: LAZY312MiB상수최대오차2MiB(171프로세스), EAGER1040MiB상수최대26MiB(19프로세스).100MiB초과0. 동일GPU의사후기술평가이며미측정GPU성능아님.
- 로컬검증은여기서마무리하고AWS독립GPU환경확대로이동. 기존L4파일럿12회는완료상태이며다음후보A10G(g5). 인스턴스는사용자가직접준비하기로함.
- 보고서: results/local-closeout-20261110-analysis/REPORT_KO.md
- AWS접속후실행/회수/중지안내: AWS_NEXT_STEP_20261110.md. 기존번들10파일/hash검사통과, 자동중지코드없다는문서오류정정.

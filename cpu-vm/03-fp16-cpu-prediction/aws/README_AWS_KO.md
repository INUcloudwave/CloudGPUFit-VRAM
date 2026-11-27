# AWS FP16 CPU 실행 안내 (L256 395개 조건, 2026-11-17 INUOS)

## 1. 무엇을 하는가

INUOS에서 돌린 CPU 예측(main395_L256)을 FP16을 지원하는 AWS CPU에서 다시 돌린다. 이번에는 모델과 입력을 **실제 FP16**으로 실행해서, FP32로 돌린 뒤 FP16 크기로 환산하는 단계를 없앤다.

INUOS와 같은 것 / 바뀌는 것:

| 항목 | INUOS (main395_l256) | AWS |
|---|---|---|
| 도커 이미지 | `cloudgpufit-cpu-validation:20261108` (PyTorch 2.6.0, transformers 4.52.4) | 같은 이미지 파일 (`docker load`) |
| 작업자, 프로토콜 | `cgf680.cpu_worker2` (로드 합성, warmup 3 + measure 1, xMem 고정본) | 같음 |
| 조건표 | `manifest/cases395_l256.jsonl` 395개 | 같음 |
| 모델 | 13개, 리비전 고정 | 같은 리비전과 파일 (`aws/model_files.json`, blob 해시 확인) |
| 조건별 torch 스레드 수 | 5 / 13 / 20 | **INUOS 값 그대로** (`aws/inuos_case_stats.json`) |
| 실행 dtype | FP32 실행 + FP16 크기 환산 (shadow v1) | **`--dtype-mode fp16_real`** (실제 FP16 실행, 환산 없음) |
| 작업 공간 위치 | 별도 CPU 프로브 | 같은 실행 중에 첫 GEMM 출력 직후 위치를 기록한다 (INUOS 프로브와 같은 값, gpt2 54 확인) |
| 병렬 배치 | 슬롯 3개 (13 / 5 / 5코어) | 조건별 코어·메모리를 INUOS 기록으로 정한다 (아래) |

- 선택 모드: `MODE=fp32_shadow_v2`는 INUOS 방식에 T5 환산 규칙 수정을 더한 것이다.
  - fp16_real 결과와 비교하고 싶을 때만 쓴다. 기본 실행에는 들어가지 않는다.

## 2. 서로 영향을 주지 않게 하는 방법

- 조건마다 새 컨테이너를 띄운다. 네트워크는 없고, 출력 폴더는 조건별로 따로 쓴다.
- 코어는 겹치지 않게 나눈다(`--cpuset-cpus`).
- 메모리는 컨테이너마다 상한을 건다(cgroup). 상한의 합이 (전체 RAM − 12 GiB)를 넘지 않을 때만 새 조건을 시작한다.
- **스레드 수는 INUOS와 같고, 배정 코어 수만 줄인다.**
  - 요청 목록은 스레드 수에 따라 달라지지만(CPU 커널의 스레드별 작업 버퍼), 배정 코어 수와는 무관하다.
  - INUOS에서 확인: gpt2 b1을 5스레드로 코어 1개에서 돌린 결과가, 저장된 5스레드·5코어 결과와 바이트까지 같았다.
  - 따라서 병렬도가 바뀌어도 실험 결과는 같고, 실행 시간만 바뀐다.

## 3. 실행 계획 (순서와 자원)

INUOS 기록(395개, 조건별 실행 시간·트레이스 시간·최대 메모리·스레드)으로 정했다. `tools/run_aws.py`가 자동으로 처리한다.

1. **1단계 P: INUOS에서 성공한 386개**
   - 오래 걸린 조건부터 넣는다(LPT). 코어나 메모리가 남으면 뒤쪽의 작은 조건이 빈자리를 채운다.
   - 코어 수 = 스레드 × (INUOS 실행 시간 중 트레이스 비율) × 0.5. 5분 이상 걸린 조건은 최소 2코어, 5분 미만은 1코어다.
     - 실행 시간 대부분이 단일 스레드 파이썬(프로파일러 처리, 매칭, xMem 재생)이기 때문이다.
     - 분포: 1코어 261개, 2코어 45개, 3~9코어 80개
   - 메모리 = 1.6 × INUOS 최대 + 2 GiB (최소 6 GiB, 최대 66 GiB)
   - 맨 앞에 들어가는 조건 (INUOS 시간 기준)

     | 조건 | 시간 | 스레드 | 코어 |
     |---|---|---|---|
     | t5-3b b32/128/512 | 99분 | 20 | 2 |
     | t5-3b b32/256/512 | 87분 | 13 | 2 |
     | t5-3b b128/128/256 | 70분 | 20 | 3 |
     | qwen b128/512/512 | 62분 | 20 | 9 |
     | qwen b128/256/512 | 60분 | 13 | 6 |
     | tinyllama b128/512/512 | 56분 | 20 | 9 |

   - 시간 제한: max(2시간, 6 × INUOS 시간)
2. **2단계 S: P가 모두 끝난 뒤 실행**
   - 대상: INUOS에서 CPU 메모리 부족으로 실패한 9개(gpt2-xl b128 3개, t5-3b b128 6개)와, AWS의 P 단계에서 메모리 부족·시간 초과·도커 오류로 실패한 조건
   - 자원: 96 GiB, 12코어, 스레드는 INUOS 값. 512 GiB 기준 동시에 5개
3. **3단계: reserved v3 재생**
   - 내림차순 주소 + 작업 공간 + RTX 3090 용량으로 R_inf, R_at, R_min을 계산한다.
   - 48개 병렬, 작업 공간 위치는 각 조건의 result.json에서 가져온다.
4. **결과 묶기**
   - `<bundle root>/aws_fp16_l256_results.tar.gz`로 만든다. 원본 trace.json은 빼고, result.json, npz, 로그만 넣는다.

### 예상 시간 (r7i.16xlarge, 64 vCPU, 512 GiB)

- INUOS FP32 시간으로 시뮬레이션했다. 트레이스가 코어 수에 따라 얼마나 빨라지는지에 따라 범위가 생긴다.
  - P 단계: **약 3.7~6.5시간**
  - S 단계: 약 0.5~1.5시간 (INUOS에서 끝까지 돈 적이 없어 불확실)
  - 3단계: 약 1시간
- Sapphire Rapids가 INUOS(Cascade Lake)보다 코어당 빠르면 시간이 줄어든다. CPU의 FP16 커널 속도는 아직 재 본 적이 없다.
- **스모크 테스트 3개의 시간으로 다시 추정한다.**

## 4. 인스턴스

- **권장: r7i.16xlarge** (Intel Sapphire Rapids, AVX512-FP16, 64 vCPU, 512 GiB)
  - 대안: r7i.24xlarge (96 vCPU, 768 GiB). 같은 계획으로 약 2.5~4시간이다.
  - 사용 전에 서울 리전의 가격과 재고를 확인한다.
- 디스크: gp3 **250 GB** 이상. 구성: 이미지 3.8 GB, 모델 28 GB, 결과, 큰 조건의 임시 trace.json(수 GB).
  - 남은 디스크가 30 GiB 아래로 내려가면 새 조건을 시작하지 않는다.
- OS: Ubuntu 22.04/24.04 또는 Amazon Linux 2023
  - 필요 패키지: docker, tmux, zstd, python3 (호스트 스크립트는 표준 라이브러리만 쓴다)
- 네트워크: 모델 다운로드 때만 필요하다(Hugging Face). 실험 컨테이너는 `--network none`으로 돈다.

## 5. 절차

```bash
# 0) 인스턴스 준비 (Ubuntu 예)
sudo apt-get update && sudo apt-get install -y docker.io tmux zstd python3
sudo usermod -aG docker $USER && newgrp docker
# 1) 번들 업로드·풀기 (INUOS에서 scp)
mkdir -p ~/cgf && cd ~/cgf && tar -xzf cgf_aws_bundle_20261117.tar.gz
# 2) 이미지 로드 + 모델 다운로드 + xMem 잠금 확인 + 스모크 테스트 3개
bash cgf-fp16-680-20261111/aws/setup.sh ~/cgf/cloudgpufit-cpu-validation_20261108.tar.zst
# 3) 본 실행 (tmux)
tmux new -s cgf 'bash cgf-fp16-680-20261111/aws/run.sh 2>&1 | tee cgf-fp16-680-20261111/runs/aws_run.console.log'
# 진행 확인: tail -f cgf-fp16-680-20261111/runs/aws_fp16_l256/scheduler_events.jsonl
# 4) 결과 가져오기: ~/cgf/aws_fp16_l256_results.tar.gz -> INUOS. 끝나면 인스턴스를 종료한다.
```

- 중간에 멈춰도 같은 명령으로 다시 시작하면 된다. 끝난 조건(result.json에 최종 상태)은 건너뛴다.
- 비교와 분석은 INUOS에서 한다.
  - GPU 비교: `tools/compare_reserved.py`
  - INUOS shadow 결과와 요청 목록 비교

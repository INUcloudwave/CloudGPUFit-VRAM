# 로컬 측정 최적화 시험

**자동 중단과 조건 축소는 없습니다.** 2시간은 향후 AWS 실험의 목표입니다. 로컬에서는 실제 시간과 기존 결과 유지 여부를 확인합니다. 이 ZIP에는 시간 제한 실행기나 105개 조건 계획이 포함되지 않습니다.

## 기본 설정

- 기존 9개 모델, 342개 core 조건, 3개 정밀도 유지
- 조건마다 새 프로세스·컨테이너를 사용하고, 같은 조건의 모드 사이에서만 모델 재사용
- 기본적으로 36°C 냉각 대기를 하지 않음. 이전 프로세스의 메모리와 남은 GPU 작업은 확인
- 첫 비교는 기존 core의 워밍업 3회, 일반 측정 20회, 구성 분석 1회 유지
- 온도·SM 클록·thermal slowdown은 측정 전후 경계에서 기록. 구간 중의 모든 변화를 관측하는 것은 아님
- 단계별 시간·전체 시간·진행률·추정 남은 시간 기록. 추론 측정 루프 안에서는 출력하지 않음

## 노트북 → 연구실 컴퓨터 → VM

AnyDesk 파일 관리자로 노트북의 ZIP을 연구실 컴퓨터의 `~/yeonjun`으로 업로드합니다. 그다음 **연구실 컴퓨터 터미널**에서 실제 포트와 주소로 바꿔 실행합니다.

```bash
cd ~/yeonjun
scp -P 실제포트번호 -i <키파일 경로> \
  ./cloudgpufit-bench-local.zip \
  inugpu@실제VM주소:/home/inugpu/cloudgpufit-bench/
```

포트 옵션은 대문자 `-P`입니다.

## VM에서 기본 조건 9개 먼저 실행

```bash
cd ~/cloudgpufit-bench
python3 -m zipfile -e cloudgpufit-bench-local.zip .
python3 test_protocol.py

python3 run_bench_v2.py \
  --profile local_core --case-set baseline9 --threads 4 \
  --no-temperature-wait \
  --reference results/20261006T083514Z_e0e9d733
```

기존 Docker 이미지와 `hf-cache`를 사용하므로 **기존 cloudgpufit-bench 폴더에** 압축을 풉니다. 원래 `run_bench.py`, `bench_worker.py`, `benchmark_plan.json`, 결과 폴더는 변경하지 않습니다.

`baseline9`는 기존 core의 모델별 첫 FP16 조건이며, smoke 조건과 다릅니다. 적용 측정 39개를 모델 로딩 9회로 수행합니다. 온도 대기 비활성화는 manifest에서 `maximum_start_temperature_c=0`으로 기록됩니다.

완료 후 표시되는 새 Results 폴더로 비교합니다.

```bash
python3 compare_results.py \
  --reference results/20261006T083514Z_e0e9d733 \
  --candidate results/새결과폴더
```

`comparison.csv`, `comparison-summary.json`은 초기 회귀 점검이며 통계적 동등성 증명이 아닙니다. 환경·측정 범위·횟수 불일치와 관측된 thermal slowdown을 별도 issue로 기록합니다. 세부 허용 기준은 JSON과 코드에 명시되어 있습니다.

## 기본 비교 후 반복 수 축소 시험

우선 위 결과를 확인한 뒤 사용합니다. 일반 성능 20회는 유지하고 메모리·토큰 측정만 3회로 바꿉니다.

```bash
python3 run_bench_v2.py \
  --profile local_core --case-set baseline9 --threads 4 \
  --no-temperature-wait --memory-iterations 3 --token-iterations 3 \
  --reference results/20261006T083514Z_e0e9d733
```

이 경우 memory/tokens의 샘플 수가 달라 비교기에 `iteration_count_changed`가 표시됩니다. 3개 표본의 토큰 p90은 진단용입니다. `--warmup`, `--repeats`, `--performance-iterations`, `--first-logits-iterations`도 선택적으로 조절할 수 있습니다.

로딩 공유와 온도 대기 해제를 먼저 함께 시험하므로, 차이가 나면 원인 분리를 위해 `--isolation separate-mode`로 재실행하거나 `--max-start-temp 36`으로 온도 대기를 다시 켤 수 있습니다. 반복 수를 바꾸기 전에 기존 결과와 비교합니다.

## 기록과 검증

`summary.csv`, 조건별 JSON, `host-timings.json`, `case-timings.json`, `events.jsonl`, `run-manifest.json`이 저장됩니다. `--progress-seconds 0`은 실시간 표시만 끕니다. 첫 완료 전 ETA는 calibrating이고, 이후 값도 추정치입니다.

재사용 모드의 `startup_metrics_scope=reused_resident_model`은 새 모델 로딩 피크를 뜻하지 않습니다. 메모리·토큰 진단을 일반 성능 측정에 합치지 않았습니다. CUDA 문맥과 workspace는 모드 사이에 남을 수 있어 기존 결과와 비교가 필요합니다.

Python 문법·CPU 프로토콜 검사를 통과했습니다. 실제 추론 측정 함수의 AST가 원본과 동일한지, 조건 342개 유지, 모드별 반복 설정, RNG/config 복원, 조건 간 재사용 금지, 온도 대기 비활성화를 검사합니다. GPU 실행·결과 동등성은 아직 확인하지 않았습니다.

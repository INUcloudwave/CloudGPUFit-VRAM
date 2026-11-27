# CloudGPUFit outside lab

먼저 `EXPERIMENT_DESIGN.md`를 읽으시면 됩니다.

GPU를 사용하기 전 CPU에서 원본 JSON을 재분석하고 신규 측정을 준비하는 별도 패키지입니다. CloudGPUFit 저장소나 실행 중인 xMem을 변경하지 않습니다. 모델 학습 코드가 아닙니다. 기존 GPU 실측 이미지를 재사용하며 의존성을 자동 설치하지 않습니다.

## 현재 확인 상태

- AWS 원본 6개 압축 파일 재분석 완료. 원본은 이 ZIP에 중복 포함하지 않았습니다.
- CPU 측정 회계/품질/반복단위 테스트, Python 구문 검사, core/models 계획 dry-run 완료.
- 실제 GPU·Docker 통합 실행, 로컬 이미지/캐시 존재 확인은 아직입니다.
- 이후 단계 명령은 로컬 환경 결과를 확인한 후 2–3개씩 안내합니다.

## 포함 파일

| 파일 | 역할 |
|---|---|
| EXPERIMENT_DESIGN.md | 기존 결과, 한계, H0–H6 설계, 고정 기준, 단계별 확장 및 학습 판단 |
| criteria.json | 앞으로의 실험 판단 기준 |
| reanalyse.py | 기존 bench-v3 추출 디렉터리의 CPU 재분석 |
| analysis/ | 실제 6개 아카이브의 분석 CSV·원본 해시·manifest 감사 |
| preflight.py | 호스트의 nvidia-smi·Docker 이미지·MPS 프로세스 읽기 전용 확인 |
| container_run.py | 이미지 ID 검증, CPU 검사/캐시 준비/계획과 GPU smoke/pilot 분리 실행 |
| pilot.py | 3개 무작위 반복 블록, 각 job을 새 Python 프로세스로 순차 실행 |
| probe.py | import/init/첫 할당/연산/모델 단계, raw 조회 구간·오류 보존 |
| prepare_cache.py | GPU 없이 pinned revision 모델 가중치 캐시 검사/다운로드 |
| pilot-plan.json / revisions.json | GPT-2·ResNet-50·T5-small 기존 FP16 기본 조건과 고정 revision |
| analyse_probe.py | 신규 측정의 프로세스·단계·독립 반복 요약 |
| test_protocol.py | GPU 없이 실행하는 측정 정의·단위 검증 |

## CPU에서 재현 가능한 명령

패키지 디렉터리에서 다음 두 명령으로 테스트와 실행 계획을 확인할 수 있습니다. 이 단계는 torch를 import하거나 GPU에 접근하지 않습니다.

```bash
python3 -m unittest discover -s . -p test_protocol.py -v
python3 pilot.py
```

기존 원본 아카이브를 안전하게 풀어둔 경로를 대상으로 `python3 reanalyse.py --input <추출_상위_경로> --out <새_분석_경로>`를 실행합니다. 디렉터리 구조는 `추출_상위/원본첨부명/results/실행ID/run-manifest.json`입니다. 현재 6개 첨부명 대응은 코드에 명시되어 있고 원래 manifest 라벨은 보존합니다. 신규 데이터 분석은 `python3 analyse_probe.py --input <신규_결과_상위> --out <새_분석_경로>`입니다.

## GPU 실행 전 순서

1. 호스트 GPU UUID/driver/다른 프로세스와 Docker 이미지 ID를 확인합니다.
2. `container_run.py check`로 GPU 없는 컨테이너 import·pip check를 합니다. 기본 이미지 ID가 다르면 멈추므로 다른 환경을 원본 환경으로 오인하지 않습니다. Docker 권한이 필요하면 `--sudo`를 지정합니다.
3. 모델 단계 전에 기존 HF_HOME 디렉터리를 찾아 `container_run.py cache-check --cache <경로>`로 검사합니다. 없으면 GPU 옵션 없이 `cache-download`로 세 모델만 준비할 수 있습니다. read-only check와 download를 구분합니다.
4. `container_run.py plan`은 GPU를 노출하지 않습니다. `smoke`와 `pilot`만 지정한 단일 UUID를 노출합니다. 둘 다 `--gpu-uuid`와 새 `--output` 경로를 명시해야 합니다. 모델 pilot에는 `--tier models --cache <경로>`가 필요합니다.

`pilot`은 core 기본 30개 fresh-process job, models 기본 18개입니다. 실행 중 오류가 있으면 중지하고 원본을 남깁니다. 중복 실행이나 자동 재시도는 하지 않습니다. 환경 sweep은 자동 실행하지 않습니다. `--env CUDA_MODULE_LOADING=LAZY` 등은 새 output을 사용한 별도 실험으로만 사용합니다.

CPU checker는 실제 GPU 연산 호환성까지 확인하지 못합니다. 이 ZIP의 코드가 GPU 검증을 통과했다고 해석하지 마십시오. smoke 결과와 host 환경을 확인한 뒤 파일럿을 진행합니다.

## 데이터 해석 제한

기존 1,938조건과 신규 raw 시간 샘플을 독립 GPU 표본으로 세지 않습니다. init 전 결측은 0이 아닙니다. outside는 NVML PID 메모리−reserved이며 xMem 오차를 흡수하지 않습니다. 다른 GPU PID가 있는 파일럿은 단독 프로세스 결과와 섞지 않습니다. 단계별 checkpoint에는 동기화 효과가 포함되고 새로운 모델 프로토콜은 원본 benchmark의 완전한 이력 복제와 구분됩니다.

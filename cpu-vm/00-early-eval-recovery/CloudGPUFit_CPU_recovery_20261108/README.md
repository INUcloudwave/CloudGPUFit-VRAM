# CloudGPUFit CPU 예측 복구 패치

이 패키지는 기존 결과와 실행 코드를 보존하고, 별도 작업 폴더에서 수정본을 검증하기 위한 것입니다.
기준 커밋은 `7f822d76af13bea39191beb1852ec67513b18f15`입니다.
원격 GitHub에는 업로드되지 않았습니다.

## 확인 결과

- 기존 manifest 기준 43.01시간, 성공 244개, 처리 오류 16개입니다.
- 마지막 수동 중단 조건까지 포함한 파일 기준으로 오류는 17개이며, 미실행은 81개입니다.
- 커널 OOM 16건과 각 오류 JSON의 저장 시각이 0~2초 차이로 일대일 대응합니다.
  기존 실행기가 조건별 host PID를 기록하지 않았으므로 시각 기반 대응이라는 한계가 있습니다.
- GPT-2 XL 두 조건은 단일 Python 프로세스가 약 46.25GiB에 도달해 종료되었습니다.
- CPU FP16의 전체 실행시간도 길었습니다. 기록 처리와 실제 CPU 연산의 비중은 새 로그로 확인해야 합니다.

## 수정 내용

- xMem 원본 allocator와 3회 추론을 유지하고, 시각화용 상세 이력만 선택적으로 생략합니다.
- JSON을 순차 파싱해 필요한 메모리 이벤트만 보관하고, profiler 참조 생존 구간을 분리합니다.
- 단계별 시간·RAM·PID·예외 traceback을 기록합니다.
- 별도 작업 컨테이너에 RAM 상한·시간제한을 적용하고 OOM/timeout/수동 취소를 구분합니다.
- 기존 성공값과 allocated/reserved를 각각 byte 단위로 비교하며 차이가 나면 다음 작업을 중단합니다.

39개 회귀 검사와 작은 모델 5종×3정밀도=15개 통합 검사를 통과했습니다.
검증 환경은 Python 3.12 / torch 2.6.0+cpu / Transformers 4.52.4이며 VM 환경과 구분합니다.
이 환경에는 Docker daemon이 없어 실제 컨테이너 메모리 제한·mount·취소 동작은 VM에서 검증해야 합니다.
원래 크기의 모델에 대한 RAM 감소량·속도 개선 배수는 아직 측정하지 않았습니다.

## 1. 별도 작업 폴더 준비

압축을 풀고, inuos의 기존 CloudGPUFit 저장소 폴더에서 실행합니다.
`prepare_recovery.py`의 실제 위치에 맞게 아래 경로를 사용하세요.

```bash
python3 ~/CloudGPUFit_CPU_recovery_20261108/prepare_recovery.py \
  --repo . --target ~/CloudGPUFit-cpu-recovery-20261108
```

이 명령은 로컬 `git worktree`만 만들고 그 안에 패치를 적용합니다.
기존 checkout·컨테이너·결과를 수정하거나 원격 저장소에 접속하지 않습니다.
대상 폴더가 이미 있으면 덮어쓰지 않고 종료합니다.

## 2. 성공했던 작은 조건 두 개만 검증

기존 `xmem-cpu` 컨테이너가 실행 중이며 예측 프로세스는 없는 상태에서 실행합니다.
`--prepare`는 사용하지 않습니다. 기존 컨테이너의 패키지 상태를 로컬 이미지로 복제한 뒤
원본 코드·캐시는 읽기 전용으로 연결하고 새 코드를 별도 읽기 전용 mount로 사용합니다.
필요한 패키지를 기존 컨테이너에 설치하거나 원본 코드를 동기화하지 않습니다.

```bash
cd ~/CloudGPUFit-cpu-recovery-20261108
python3 -u tools/run_prediction_recovery.py \
  --output-root ~/cloudgpufit-predictions/recovery-check-01 \
  --case-id core__decoder_llm__small__fp32__01 \
  --case-id core__decoder_llm__small__fp16__01 \
  --compare-with ~/cloudgpufit-predictions/native-v3-t4-j6 \
  --threads 4 --memory-gib 32 --host-reserve-gib 6 --timeout-seconds 3600
```

32GiB는 작업의 RAM 필요량 예측이 아니라 VM 보호용 상한이며, 1시간도 완료 예상시간이 아닌 운영 제한입니다.
수정 후 사용량을 확인해 조정합니다. output-root가 이미 있으면 새 이름을 사용하세요.
이전 실패 조건이나 전체 98개 조건을 아직 시작하지 마세요.

## 3. 결과 확인

```bash
python3 - <<'PY'
import json
from pathlib import Path
p=Path.home()/'cloudgpufit-predictions/recovery-check-01'
print((p/'manifest.json').read_text())
for f in sorted(p.glob('*.prediction.json')):
    d=json.loads(f.read_text())
    print(f.name, d['status'], d.get('error_kind'), d.get('reference_comparison'))
PY
```

manifest와 각 조건의 `.log`·`.prediction.json`을 공유하면 VM 결과를 확인할 수 있습니다.
정상 완료와 기존값 일치를 확인한 뒤 실패 조건 1개, T5 및 Vision 경로로 검증 범위를 넓힙니다.
자세한 근거·조건 목록은 동봉된 `diagnosis.md`와 패치 적용 후
`docs/cpu_prediction_recovery_20261108.md`를 참고하세요.

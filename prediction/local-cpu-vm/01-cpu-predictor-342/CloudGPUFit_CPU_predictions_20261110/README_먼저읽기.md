# CloudGPUFit CPU 예측 결과 — 2026-11-10 정리

## 바로 확인할 파일

- `CPU_predictions.xlsx`: 요약, 성공325, 전체342, 실패17, 모델/정밀도별, GPU별 요구량, 컬럼설명, 원본 출처 시트. 첫 행 고정과 자동 필터를 적용했습니다.
- `cpu_predictions_success.csv`: AWS 실측과 우선 비교할 정상 예측 325개.
- `cpu_predictions_all.csv`: 전체 조건 342개. 실패 17개도 보존.
- `cpu_predictions_failures.csv`: cgroup_oom 16개 + timeout 1개.
- `cpu_predictions.json`: CSV와 같은 정리 결과. 실패의 예측값은 null.
- `raw_predictions_with_requests.json`: 조건·요청·모델 config·원본 결과를 한 JSON으로 묶었습니다.
- `raw/predictions/`: 원본 prediction/input JSON 342쌍. 기존 비교 스크립트의 --predictions 입력으로 사용할 수 있습니다.
- `raw/logs/`: 선택된 조건의 최초/solo 재시도 로그. 원본 파일을 수정하지 않았습니다.
- `prediction_provenance.json`, `raw/runs/`: 각 조건의 원본 위치·hash와 두 실행의 manifest/scheduler 기록.
- `source/`: 실제 predictor hash와 일치하는 코드·계획·revision·allocator lock. runner/scheduler는 후속 332개 실행과 hash가 일치하는 버전입니다.
- `commands/retry_oom_unlimited.sh`: 사용자 직접 실행용 명령. 이 패키지를 만들면서 실행하지 않았습니다.

## 집계 범위

앞선 실행 `parallel-all342-20261108-180550`에서 성공한 10개와, 이를 제외한 후속 실행 `parallel-all342-20261108-183250`의 332개 결과를 case_id로 합쳤습니다. 중복/누락은 없습니다. 후속 실행은 성공315/실패17이고, 전체는 **성공325 / 실패17 / 미실행0**입니다.

첫 실행 manifest는 중단으로 runner_failed여도, 이 패키지에는 해당 실행의 실제 정상 결과 10개만 채택했습니다. 후속 실행의 재시도 전 오류는 최종 결과의 previous_attempt 및 원본 로그에 남아 있습니다. 예전 native-v3 결과 244개나 작은 검증용 모델 결과는 섞지 않았습니다. 제가 직접 시작했다가 사용자 요청으로 중단한 11월10일 RAM 무제한 시도도 포함하지 않았습니다.

두 실행의 predictor 5개 파일 identity, selected-plan 및 revisions SHA-256은 동일합니다. 호스트 실행기/병렬·보호 설정은 달랐으며 원본 manifest에서 확인할 수 있습니다. 성공 결과 모두 prediction_version=4-native-inference-single-request, warmup=0, measured=1입니다. manifest의 git_commit은 작업 기준 7f822d7이며, 실제 변경 소스는 identity hash를 기준으로 확인해야 합니다.

## AWS 실측과 비교하는 방법

1. case_id, model, dtype, revision, batch/길이/해상도/max_new_tokens 및 actual_input_shapes를 대응합니다. case_id 번호만 보고 입력을 추정하지 마세요.
2. CPU `predicted_allocated_bytes` ↔ GPU `inference_peak_allocated_bytes`를 비교합니다.
3. CPU `predicted_reserved_bytes` ↔ GPU `inference_peak_reserved_bytes`를 별도로 비교합니다.
4. NVML 비교가 필요하면 기존 GPU별 보정 범위를 확인하고 `reserved + outside` 추정값을 GPU `nvml_process_sampled_peak_bytes`와 비교합니다. allocated를 NVML과 직접 같은 지표처럼 비교하지 마세요.
5. GPU 실측도 정상인 조건만 오차 통계에 사용하고, CPU 실패/실측 미지원/누락은 별도로 집계하세요. T4 BF16 미지원과 L40S 보정 미포함을 주의하세요.

CPU는 임의 초기화 from_config 모델, GPU는 pretrained 모델이며 입력 난수 값까지 동일한 것은 아닙니다. CPU는 단일 요청을 빈 allocator+상주 storage에서 재생하지만, 기존 GPU hybrid memory 측정은 선행 performance/warmup 캐시를 이어받았습니다. reserved 차이를 모두 추론 텐서 오차 또는 모두 캐시 효과라고 단정하면 안 됩니다. 이 파일에는 **AWS 실측이나 CPU-GPU 오차 통계는 포함하지 않았습니다**.

## 단위와 실행시간

- 정확한 비교는 *_bytes 정수로 하세요. MiB=2^20 bytes, GiB=2^30 bytes입니다.
- CPU RSS, cgroup peak, job_ram_limit_GiB는 CPU 실행 자원입니다. GPU VRAM 예측값이 아닙니다.
- max_logged_*는 로그에 남은 수치입니다. OOM/timeout 직전 값은 기록되지 않을 수 있어 실패 조건의 실제 peak라고 해석하지 마세요.
- final_attempt_seconds는 최종 시도 wall time입니다. 재시도 포함 시간은 recorded_attempts_seconds이며 대기시간은 제외됩니다. 병렬 조건 시간을 모두 합쳐 전체 실험 wall time이라고 부르지 마세요.
- GPU별 required_vram은 기존 보정표 기반 파생 추정입니다. 전 GPU에 보장되는 상한이나 새 실측값이 아닙니다.

## 직접 OOM 재실행

현재 VM에 SSH로 접속한 tmux에서 다음 명령을 직접 실행하세요. 이 명령은 기존 최종 OOM 16개만, 한 번에 한 조건씩, RAM hard limit/저메모리 강제종료 보호 없이 실행합니다. 조건당 6시간 제한은 남습니다. 약47GiB 물리 RAM을 초과하면 커널 OOM이 발생할 수 있습니다.

```bash
bash /home/inuos/cloudgpufit-predictions/retry_oom_unlimited_20261110.sh
```

다른 노트북에서는 패키지의 결과 파일을 확인하시면 됩니다. 재실행 스크립트는 준비된 INUOSVM-1 경로·Docker snapshot이 있는 VM용입니다.

## 무결성

`SHA256SUMS.txt`에 결과·원본·코드 파일의 SHA-256을 넣었습니다. 각 원본 prediction/input의 SHA-256은 prediction_provenance.json에도 기록했습니다. 원본 실험 폴더를 덮어쓰지 않았습니다.

# cuDNN 대체 계획(fallback) 시뮬레이터 반영 — replay5 (2026-11-23)

- 코드: `src/cgf680/replay5.py` (replay3는 그대로 둠)
  - 합성곱 호출마다 작업 공간 후보 목록(3090 probe의 candidates)을 넣는다.
  - PyTorch 2.6 동작을 따른다.
    - 형태별로 고른 계획을 기억한다.
    - 작업 공간 할당이 실패하면(할당기가 캐시를 비우고 다시 시도한 뒤에도) 다음 후보로 넘어가고, 그 후보를 기억한다.
- 검사 (`fallback_eval.py` → `fallback_eval.json`)
  - A) 메모리가 충분하면: replay5 = replay3 (1순위 작업 공간), CNN 108/108 조건 동일, fallback 0회
  - B) 메모리가 부족하면: OOM 없이 도는 최소 용량

| 조건 | 대체 없음 | 대체 반영 | 차이 |
|---|---|---|---|
| resnet-50 b128 r512 | 5,142 MiB | 4,628 MiB | −514 |
| regnet-y-040 b128 r512 | 5,896 | 4,870 | −1,026 |
| resnet-18, convnext, mobilenet b128 r512 | 변화 없음 | | 최소 용량이 텐서로 정해짐 |

- **GPU 검증 (2026-11-23): 15/15 지점 일치**
  - 예측은 측정 전에 기록했다: `cap_predictions_preregistered.jsonl` (17:39, sha256 dc1e887c…)
    - `cap_predict.py`로 만들었다. replay3와 replay5를 같은 상한에서 계산한다.
  - 측정: RTX 3090, `set_per_process_memory_fraction` 상한 (INUGPU `run_conditions_cap.py` / `cap_worker.py`, 작업자 d06f0f15)
    - resnet-50: 11-16 최소 용량 탐색 결과를 그대로 사용 (`gpu_check/resnet50_capsearch_20261116.jsonl`)
    - regnet: 11-23 INUOS가 INUGPU에서 실행 (사용자 승인, `gpu_check/`)

| 조건 | 상한 (MiB) | GPU 실측 (reserved / allocated MiB) | replay5 | replay3 |
|---|---|---|---|---|
| resnet-50 b128 r512 | 6,170 | 6,168 / 4,091.1 | 같음 | 같음 |
| | 5,130, 4,870, 4,740, 4,675, 4,642 | 4,630 / 3,835.1 (재시도 40회) | 4,630 / 3,835.1 | OOM ✗ |
| | 4,610 | OOM | OOM | OOM |
| regnet-y-040 b128 r512 | 7,946 | 7,944 / 5,873.8 | 같음 | 같음 |
| | 5,900 | 5,896 / 5,873.8 (재시도 1회) | 같음 | 같음 |
| | 5,640, 5,380, 5,130, 4,900, 4,872 | 4,870 / 4,849.7 (재시도 13~14회) | 4,870 / 4,849.7 | OOM ✗ |
| | 4,840 | OOM | OOM | OOM |

- 해석
  - 메모리가 빠듯하면 cuDNN이 작업 공간이 작은 계획으로 바꾼다. replay5는 바뀐 뒤의 reserved와 allocated까지 바이트 단위로 맞힌다.
  - 11-16에 원인을 모르던 "최소 용량 예측 5,142 vs GPU 4,642 MiB"(resnet-50) 차이가 이것으로 설명된다.
  - 대체 계획을 모르면 최소 필요 메모리를 과대평가한다(resnet-50 +514, regnet +1,026 MiB).
- 수정: allocated 최대치에 작업 공간이 빠지던 버그를 고쳤다(11-23, 예측 기록 전). reserved 결과에는 영향이 없다.
- 한계
  - 조건 2개, RTX 3090(cuDNN 9.1)만 확인했다.
  - 측정이 끝난 지점(end = small_pool_limit 등) 뒤의 후보는 모른다. 마지막으로 측정한 후보에서 멈춘다.

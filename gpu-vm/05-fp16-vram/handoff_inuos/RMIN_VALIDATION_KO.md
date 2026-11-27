# R_min GPU 실측 검증 — R_inf/R_min 예측 절차 보충·정정 (2026-11-16 저녁)

- 기준 문서
  - `~/cgf-fp16-inuos-handoff-20261116/RESERVED_PREDICTION_GUIDE_KO.md`
  - 그 보충 `~/cgf-fp16-inuos-handoff-20261116b/ADDENDUM_20261116b_KO.md`
- 이 문서가 다루는 것
  - 가이드 §3.4·§4의 R_min을 GPU에서 직접 측정해, xMem(낮은 주소 규칙) 예측과 비교했습니다.
  - 가이드 §4의 가정 하나를 **정정**합니다(§3).

---

## 0. INUOS 에이전트에게 (작업 프롬프트 보충)

```
1. 가이드 §4 판정에서 "R_min ≤ C < R_inf → reserved ≈ C" 는 틀렸습니다. 아래 §3대로 바꿔 주세요.
   이 구간은 "실행 가능(캐시 재시도 발생), reserved 값은 경로 의존이라 예측하지 않음"으로 둡니다.
2. R_min 계산은 tools/capacity_sweep_desc.py (낮은 주소 규칙) 방식을 쓰세요.
   LLM·T5에서는 GPU 실측과 32 MiB 해상도 안에서 맞습니다(최대 +199 MiB, 보수적).
3. CNN은 xMem R_min이 GPU보다 약 500 MiB 크게 나옵니다(보수적 방향, §2). 용량이 부족하면 GPU는
   cuDNN 작업공간이 작은 알고리즘으로 바꿔 요청 자체가 달라지는 것으로 추정합니다.
   후보 판정에서 탈락 방향 오차이므로 그대로 써도 OOM 위험은 없습니다.
4. CPU 요청 목록으로 같은 7개 조건의 R_inf/R_min을 계산해 results/rmin_gpu_vs_xmem_desc.csv와 비교해 주세요.
```

---

## 1. 실험 방법

- GPU에서 메모리 상한을 걸어 실행했습니다.
  - 상한 적용: `torch.cuda.set_per_process_memory_fraction(cap / total_memory)`
  - 이 상한은 xMem의 `allowed_memory_maximum`과 같은 의미입니다. reserved + 요청 > cap이면 cudaMalloc 실패처럼 처리되어 캐시 해제 → 재시도 → OOM 순으로 갑니다.
- 측정 코드는 바꾸지 않았습니다.
  - `tools/cap_worker.py`가 기존 `vram_worker.py`를 그대로 실행하고, `torch.cuda.init()` 직후에 상한만 겁니다.
  - 러너: `tools/run_conditions_cap.py` (기존 러너 복사본, `--mem-cap-mib` 추가)
- 프로토콜은 main395_L256과 같습니다. 조건마다 새 컨테이너, warmup 3 + measure 5.
- 탐색: `tools/gpu_rmin_search.py`
  - 하한 = peak allocated, 상한 = 상한 없이 실행했을 때의 reserved
  - 32 MiB 해상도까지 이진탐색, 시도마다 새 프로세스
  - 기록: `results/gpu_rmin_search.jsonl`(시도별), `results/gpu_rmin_search_summary.jsonl`(조건별)
  - 원자료: `results/capsearch_raw/<cid>__cap<MiB>/measure.json`, `measure.cap.json`
- 실행 확인: opt-125m b16에서 cap 2000 MiB는 ok, 700 MiB는 gpu_oom이었습니다.
- xMem 쪽: `tools/capacity_sweep_desc.py`
  - 기존 `capacity_sweep.py`에 낮은 주소 규칙을 넣은 것입니다. 2 MiB 해상도.
  - GPU 요청 목록(trace)을 입력으로 씁니다.
  - 결과: `results/capsweep_desc/*.json`

---

## 2. 결과 (`results/rmin_gpu_vs_xmem_desc.csv`, 단위 MiB)

| 조건 | 상한 없는 GPU reserved | peak allocated | xMem R_inf | **GPU R_min** (OOM, ok] | **xMem R_min** | 판정 |
|---|---|---|---|---|---|---|
| opt-125m b16/128/512 | 968 | 629 | 986 | (713, 734] | 737 | +3 ~ +24 |
| qwen2.5-1.5b b128/128/512 | 6,868 | 5,821 | 6,876 | (6,853, 6,870] | 6,876 | +6 ~ +23 |
| tinyllama b128/128/256 | 7,104 | 3,581 | 7,104 | (4,212, 4,240] | 4,222 | 범위 안 |
| gpt2 b128/512/512 | 13,582 | 10,099 | 13,582 | (13,556, 13,584] | 13,580 | 범위 안 |
| t5-small b128/128/512 | 11,214 | 1,198 | 11,214 | (1,354, 1,373] | 1,375 | +2 ~ +21 |
| tinyllama b128/128/512 (3090에서 재시도) | 20,730 | 4,541 | 40,000 | (5,552, 5,583] | 5,751 | +168 ~ +199 |
| resnet-50 b128/r512 | 6,168 | 4,091 | 6,168 | (4,610, 4,642] | 5,142 | +500 ~ +532 |

- LLM·T5 6개
  - xMem R_min이 GPU 실측 구간 안이거나, 최대 +199 MiB(3.5%) 크게 나옵니다.
  - 과소 예측은 없었습니다. 과소 예측이 있었다면 OOM 위험이 생기는 방향입니다.
- R_inf와 R_min의 차이는 조건마다 크게 다릅니다.
  - qwen b128, gpt2 b128/512: R_min ≈ R_inf. 재시도로 버틸 구간이 없습니다.
  - t5-small b128: R_min은 R_inf의 12%입니다(1.4 GB vs 11.2 GB).
  - tinyllama b128/512: R_min은 R_inf의 14%입니다(5.6 GB vs 40 GB).
- resnet-50
  - 상한을 걸면 GPU의 peak allocated가 4,091 → 3,835 MiB로 줄었습니다. 요청 목록 자체가 바뀐 것입니다.
  - cuDNN이 작업공간을 줄이는 대체 동작으로 추정합니다.
  - 상한이 4,642~5,130 MiB이면 reserved는 상한과 무관하게 4,630 MiB였고, warmup마다 재시도가 40회 일어났습니다.
  - xMem은 요청 목록이 고정이라 이 동작을 재현하지 못해, R_min을 약 500 MiB 크게 잡습니다.

---

## 3. 정정: 재시도 구간의 reserved는 C 근처가 아님

tinyllama b128/128/512의 GPU 시도별 결과입니다(`results/gpu_rmin_search.jsonl`).

| cap (MiB) | 상태 | measure_0 peak reserved | warmup 0~2 재시도 수 |
|---|---|---|---|
| 20,732 | ok | 8,028 | 2, 2, 2 |
| 12,636 | ok | 8,202 | 4, 4, 4 |
| 8,588 | ok | 7,920 | 8, 8, 8 |
| 6,564 | ok | 5,652 | 18, 18, 18 |
| 6,058 | ok | 5,552 | 27, 27, 27 |
| 5,805 | ok | 5,468 | 33, 33, 33 |
| 5,678 | ok | 5,502 | 36, 36, 36 |
| 5,615 | ok | 5,612 | 37, 37, 37 |
| 5,583 | ok | 5,530 | 41, 42, 42 |
| 5,552 | gpu_oom | — | — |

- 상한 없이 실행한 3090(실제 용량 약 23.5 GB)에서는 같은 조건이 20,730이었습니다.
- 확인된 점
  - 재시도가 일어나면 캐시를 비운 뒤 다시 커집니다. 측정 구간의 reserved는 C보다 훨씬 작을 수 있고, C에 단조롭게 따라가지도 않습니다.
  - 상한이 낮을수록 warmup마다 재시도가 많아집니다(2 → 42회). 재시도마다 캐시 해제와 cudaMalloc이 다시 일어나므로 응답시간에 영향이 있을 수 있습니다. 미측정입니다.
- **판정 규칙 수정** (가이드 §4의 4번)
  - C ≥ R_inf: 실행 가능, 예측 reserved = R_inf (재시도 없음)
  - R_min ≤ C < R_inf: 실행 가능, 캐시 재시도 발생
    - reserved는 R_min~C 사이의 경로 의존 값이라 예측값으로 쓰지 않습니다.
    - 필요하면 "재시도 구간" 플래그를 응답시간 예측 쪽에 넘깁니다.
  - C < R_min: OOM, 제외
  - C = 총용량 − 드라이버 예약분 − 해당 GPU outside (가이드와 같음)

---

## 4. 미실행·한계

- 13개 새 trace(이전 보충 문서) 중 대형 조건(gpt2-medium, gpt2-xl, t5-3b)의 R_inf/R_min은 아직 계산하지 않았습니다. 재생 1회에 10~30분이 걸립니다.
- 재시도 구간의 응답시간 영향은 측정하지 않았습니다.
- GPU R_min도 run-to-run 변동이 있을 수 있습니다. 조건마다 1회 탐색만 했습니다.
- 상한은 set_per_process_memory_fraction으로 흉내 낸 것입니다. 실제로 작은 GPU에서는 CUDA 컨텍스트·드라이버 예약분이 별도로 빠지므로, 위 C 정의대로 빼고 비교해야 합니다.

---

## 5. 파일 (`~/cgf-fp16-inuos-handoff-20261116c/`)

| 경로 | 내용 |
|---|---|
| `RMIN_VALIDATION_KO.md` | 이 문서 |
| `tools/cap_worker.py`, `tools/run_conditions_cap.py`, `tools/gpu_rmin_search.py` | GPU 상한 실행과 탐색 |
| `tools/capacity_sweep_desc.py` | xMem R_inf/R_min (낮은 주소 규칙) |
| `run_gpu_rmin_search.sh`, `run_capsweep_desc.sh`, `logs/gpu_rmin_search.txt` | 실행 스크립트와 로그 |
| `results/rmin_gpu_vs_xmem_desc.csv` | §2 비교표 |
| `results/gpu_rmin_search.jsonl`, `results/gpu_rmin_search_summary.jsonl` | GPU 시도별·조건별 결과 |
| `results/capsweep_desc/` | xMem 결과 json·log |
| `results/capsearch_raw/` | GPU 시도별 원자료 (measure.json, measure.cap.json) |
| `SHA256SUMS` | `sha256sum -c SHA256SUMS` |

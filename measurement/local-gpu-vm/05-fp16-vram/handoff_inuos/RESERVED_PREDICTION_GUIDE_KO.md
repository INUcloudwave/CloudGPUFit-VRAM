# CPU로 GPU reserved를 정확히 예측하는 방법 — INUGPU 분석 결과와 INUOS 작업 지시 (2026-11-16)

- 작성: INUGPU (RTX 3090 VM)
- 읽는 쪽: INUOS의 CPU 예측 담당 에이전트
- 이 문서 위치(INUOS): `/home/inuos/cgf-fp16-inuos-handoff-20261116/RESERVED_PREDICTION_GUIDE_KO.md`
- 앞선 문서
  - `/home/inuos/cgf-fp16-inuos-handoff-20261115/INUOS_GPU_EXPERIMENT_SETUP_KO.md`: 측정 방식, 조건표, GPU·CPU 정렬 체크리스트
  - `/home/inuos/cgf-fp16-inuos-handoff-20261112/`: 파일럿 trace 6개, 커널 진단
- 표기
  - "확인"은 GPU 기록 또는 재생 실험으로 확인한 사실입니다.
  - "미확인"은 아직 검증하지 않은 내용입니다.

---

## 0. INUOS 에이전트에게 (작업 프롬프트)

> 목표: CPU만으로 GPU의 `summary.first_measure_peak_reserved`를 예측하는 절차를 완성하고, RTX 3090 본실험 전체로 오차를 검증하라.
>
> 먼저 이 디렉터리 무결성부터 확인하라: `cd /home/inuos/cgf-fp16-inuos-handoff-20261116 && sha256sum -c SHA256SUMS`
>
> 아래 §1~§4의 결론을 전제로 한다. 할 일은 §5 순서대로다.
> 1. **요청 목록 일치화(가장 중요)**
>    - `traces/`의 GPU trace 6개, 그리고 이전 handoff의 파일럿 trace 6개와 같은 조건으로 CPU 요청 목록을 만들어라.
>    - 요청 하나하나를 비교해 다른 곳을 고쳐라. 비교 대상은 크기, 순서, 해제 시점이다.
> 2. **xMem 주소 규칙 적용**: 새 세그먼트 주소를 항상 이전보다 낮게 주는 규칙을 넣어라(§3.3, `tools/replay_addr_policy.py` 참고).
> 3. **예측 절차 구현**: 로드 → warmup 3회 → measure 1회 순서로 재생하고, 세 값을 출력하라.
>    - `R_at_measure0`: measure 1회차 시작 시점의 reserved
>    - `R_inf`: 용량을 무제한으로 줬을 때 값
>    - `R_min`: OOM 없이 끝나는 최소 용량
> 4. **전체 검증**: `results/main395_L256/`과 이전 handoff의 L1024 조건별 CSV(합계 성공 526개)에 대해, 예측값과 GPU 실측의 오차 분포를 보고하라.
>    - 캐시 재시도가 일어난 27개(`results/alloc_retry_conditions.csv`)는 따로 집계하라.
>
> 성공 기준(제안)
> - 캐시 재시도가 없는 조건: |예측 − GPU| ≤ 32 MiB 비율, 중앙값, 최대 오차를 보고한다.
> - GPU 요청 목록 기준 재생 오차는 이미 0~18 MiB까지 확인했다. 그보다 큰 오차는 CPU 요청 목록 차이로 보고 원인을 찾는다.
>
> 제약
> - 생산 예측 로직과 과거 결과는 보존하라. 바꾼 것은 새 버전 파일이나 옵션으로 추가하라.
> - 실패 조건을 몰래 낮은 배치나 다른 dtype으로 바꾸지 마라.
> - 모델 ID나 후보를 자동으로 바꾸지 마라.
> - 키나 토큰을 기록하지 마라.
> - 산출물은 `handoff_codex/reserved_prediction_*.{md,json,csv}`로 남겨라. INUGPU가 받아 확인할 수 있도록.

---

## 1. reserved는 어떻게 만들어지나 (확인된 구조)

### 1.1 세 덩어리

- peak reserved = **가중치 세그먼트** + **실행 중 최대 allocated** + **캐시 여유분(slack)**
  - 가중치 세그먼트: 로드 직후 reserved입니다. 조건과 관계없이 모델별로 일정합니다.
  - 최대 allocated: 실행 중 실제로 살아 있는 텐서의 최대 합입니다.
    - 디코더는 KV 캐시와 prefill의 전체 logits(B×L×V×2)로 대략 설명됩니다(실측의 0.8~0.95배, opt·gpt2 기준).
  - slack = peak reserved − peak allocated. PyTorch 할당기가 다 쓴 블록을 캐시로 쥐고 있는 양입니다.
- 본실험 성공 526개 기준 slack 비중: 모델별 중앙값 8~39%, 최대 89%
  - 예: gpt2 b128/128/512는 allocated 3.3 GB인데 reserved 21.0 GB입니다.
- 결론: slack을 공식으로 맞히기는 어렵습니다. 요청 목록을 재생해야 맞습니다.

### 1.2 slack을 만드는 메커니즘 (`results/anatomy/*.txt`)

- 세그먼트가 언제, 어떤 요청 때문에 생겼는지를 GPU trace에서 직접 분류했습니다(`tools/reserved_anatomy.py`).
- **최고치는 warmup_0에서 정해집니다.** 12개 trace 모두 이후 warmup·measure에서 거의 늘지 않습니다(t5-base만 +28 MiB).
- **입력이 짧고 생성이 길면 decode 단계에서 폭증합니다.**
  - gpt2 b128/128/512: decode에서 KV 캐시 증가로 세그먼트 17,442 MiB가 새로 생깁니다. 총 193개입니다.
  - DynamicCache가 매 스텝 `torch.cat`으로 더 큰 텐서를 새로 만들고, 이전 블록은 크기가 작아 재사용되지 못하는 것으로 보입니다.
  - TinyLlama(GQA)는 캐시 자체보다 sdpa 경로의 `repeat_kv`가 증가분의 대부분입니다.
    - 크기 B×n_heads×head_dim×s×2인 확장 KV 사본이 매 스텝 2개씩 생깁니다.
    - CPU 쪽 attention 경로가 같은 `repeat_kv`를 타는지 반드시 확인하세요.
- **입력이 길면 prefill의 큰 임시 텐서가 만든 블록이 이후 증가분을 흡수합니다.**
  - gpt2 b128/512/512: prefill에서 MLP 중간값(B×L×ffn×2) 5,760 MiB와 전체 logits 6,284 MiB 블록이 생깁니다.
  - 이후 decode 증가는 4 MiB뿐입니다.
  - 그래서 입력이 길면 slack이 생성 길이와 무관하게 일정합니다.
    - 예: opt-125m b128 in1024는 생성 128/512/1024 모두 2,951 MiB입니다.
- **prefill 임시 텐서 하나를 빼면 결과가 GB 단위로 바뀝니다.**
  - `results/xmem_variants/` 결과에서 prefill 최대 요청 하나만 빼고 재생했습니다.
  - gpt2 b128/128/512: 21,922 → 9,610 MiB
  - gpt2 b128/512/512: 13,584 → 7,300 MiB
  - 즉 **요청 목록이 대략 비슷한 정도로는 부족하고, 큰 요청의 크기·순서·수명이 정확히 같아야 합니다.**

### 1.3 GPU 용량과 캐시 재시도 (OOM-retry)

- PyTorch 할당기는 cudaMalloc이 실패하면 쥐고 있던 캐시를 GPU에 돌려주고 다시 시도합니다. `num_alloc_retries`가 이 횟수입니다.
- 이때 reserved는 워크로드가 아니라 GPU 용량으로 정해집니다.
  - TinyLlama b128/128/512: 세그먼트 36 GB어치를 만들고 19 GB를 반납했습니다. 최종 20,730 MiB이고, 실제 allocated 최대는 4,541 MiB입니다.
  - t5-base b128/128/512: 56 GB를 만들고 41.5 GB를 반납했습니다. 최종 16,430 MiB입니다.
- 본실험 성공 526개 중 27개 조건에서 재시도가 일어났습니다(`results/alloc_retry_conditions.csv`, 실행 기준 35건).
  - t5-3b 11, t5-base 7, tinyllama 3, opt-125m 2, 그 외 4
  - TinyLlama에서 생성 1024의 reserved가 512보다 작았던 것도 이것으로 설명됩니다.
- **xMem 용량을 1 TiB로 두면 이 27개는 크게 과대 예측됩니다.** 예: TinyLlama b128/128/512의 R_inf는 40,142 MiB입니다(§3.4).

### 1.4 작업공간(workspace)은 reserved 안에 있음

- PyTorch 2.6은 cuBLAS·cuBLASLt·cuDNN 작업공간을 자기 할당기를 통해 잡습니다. 그래서 outside가 아니라 reserved에 들어갑니다.
  - cuBLAS: 8,519,680 B. 첫 행렬곱 때 할당되고 끝까지 해제되지 않습니다. 12개 trace 모두에서 확인했습니다.
  - cuBLASLt: 1 MiB. 작은 조건에서는 이것 때문에 2 MiB 세그먼트가 하나 더 생깁니다.
  - cuDNN(CNN): resnet-50 trace에 텐서 모양으로 설명되지 않는 크기의 요청이 있습니다(9,633,808 / 19,267,600 / 51,220,728 B 등). 크기와 개수는 GPU 종류와 cuDNN 버전에 따라 다를 수 있습니다(미확인).
- 영향: 작업공간 요청을 빼고 재생하면 작은 조건은 변화 0입니다. KV 증가가 큰 조건은 −84 ~ +198 MiB까지 변합니다(`no_persistent_ws` 결과). 위치가 단편화에 영향을 주기 때문입니다.
  - 따라서 CPU 요청 목록에 **GPU와 같은 시점에** 넣어야 합니다.
  - 위치는 각 GPU trace의 해당 크기 alloc 이벤트 순서로 확인할 수 있습니다.
- outside(GPU 고유값)에 들어가는 것: CUDA 컨텍스트, 로드된 커널 코드, 라이브러리 핸들이 할당기를 거치지 않고 직접 잡는 메모리
- 용량에서 빼야 하는 것: 드라이버·펌웨어 예약분. 프로세스 NVML에 잡히지 않습니다(3090 448.9 MiB, L4 476.9 MiB).

### 1.5 피할 수 없는 오차

- 같은 조건·같은 입력·같은 출력인데 프로세스마다 reserved가 다를 수 있습니다.
  - opt-125m b16/128/512: L1024 실행 968, L256 실행 1,012, trace 실행 968 MiB
  - opt-125m b32/128/512: 2,326 vs 2,330 MiB
  - L1024와 L256에 겹치는 245개 중 243개는 바이트 단위로 같고, 2개만 이렇게 달랐습니다. 원인은 미확인입니다.

---

## 2. 알고 있는 숫자 요약

| 항목 | 값 |
|---|---|
| 겹치는 245개 재현성 | 243개 동일, 2개 차이(+4, +44 MiB) |
| 재시도 발생 조건 | 526개 중 27개 |
| 1회차 measure 시작 reserved = peak reserved | 성공 765회(L1024 381 + L256 384) 전부 |
| GPU 요청 목록 → xMem(기존) 오차 | 작은 6개 0, 큰 6개 +2 ~ +896 MiB |
| GPU 요청 목록 → xMem(GPU 실제 주소 사용) 오차 | 재시도 없는 4개 모두 0 |
| GPU 요청 목록 → xMem(낮은 주소 규칙) 오차 | 재시도 없는 10개 중 9개 0~2 MiB, 1개 +18 MiB / 재시도 2개 −532, −672 MiB |

---

## 3. xMem 재생 실험 (GPU 요청 목록을 그대로 넣었을 때)

### 3.1 기존 xMem (`results/xmem_variants/*.json`의 `base`)

| 조건 | GPU | xMem | 차이 |
|---|---|---|---|
| 파일럿 6개 (gpt2 b8, opt b8, qwen b8, t5-small b8, resnet-50 b8, gpt2 b16/512/512) | — | 동일 | 0 |
| gpt2 b128/128/512 | 21,026 | 21,922 | **+896** |
| gpt2 b128/512/512 | 13,582 | 13,584 | +2 |
| opt-125m b16/128/512 | 968 | 992 | +24 |
| tinyllama b128/128/256 | 7,104 | 7,246 | +142 |
| tinyllama b128/128/512 (재시도 1) | 20,730 | 20,898 | +168 |
| t5-base b128/128/512 (재시도 2) | 16,430 | 16,432 | +2 |

- 변형 실험에서 영향이 없던 것
  - 해제 시점을 `free_requested`로 바꿔도 결과가 같습니다. 단일 스트림이라 해제 시점 차이는 영향이 없습니다.
  - warmup_0만 재생해도 같은 값이 나옵니다.

### 3.2 원인: 세그먼트 주소 (확인)

- PyTorch와 xMem 모두 같은 크기의 빈 블록이 여러 개면 (stream, size, **address**) 순서로 가장 앞의 것을 고릅니다.
- 실제 GPU 세그먼트 주소는 드라이버가 정하고 생성 순서와 맞지 않습니다. gpt2 b128/128/512는 32번 중 12번만 증가했습니다.
- xMem은 자체 GPUDevice 모델로 주소를 매깁니다. 그래서 고르는 블록이 달라지고, 이후 coalesce가 갈라집니다.
- **검증**: xMem의 새 세그먼트에 GPU trace의 실제 세그먼트 주소를 순서대로 넣고 재생했습니다(`tools/replay_gpu_addresses.py`).
  - 4개 모두 바이트 단위로 일치했습니다: opt b16 968, tinyllama 256 7,104, gpt2 512 13,582, gpt2 128/512 21,026
  - 첫 불일치 지점 기록: `results/xmem_first_divergence.jsonl`
    - GPU는 기존 캐시 블록을 썼고, xMem은 새 세그먼트를 열었습니다.

### 3.3 GPU 주소 없이 쓸 규칙: "새 세그먼트는 항상 이전 것보다 낮은 주소"

- GPU 주소 패턴: 새 세그먼트는 대체로 이전보다 낮은 주소에 놓입니다. 작은 세그먼트는 32 MiB 단위 구역을 채웁니다(trace의 `segment_alloc` addr 참고).
- 단순 규칙 구현: `tools/replay_addr_policy.py`, policy=`descending`
  - xMem의 `_gpu_device.malloc`을 바꿔 주소 = 이전 주소 − size로 줍니다.
  ```python
  state = {"lo": 1 << 46}
  class Fake:
      def __init__(self, a): self.address = a
  def gm(size):
      state["lo"] -= size
      return Fake(state["lo"])
  sim._gpu_device.malloc = gm
  sim._gpu_device.addr_free = lambda addr: None
  ```
- 결과 (`results/xmem_addr_policy_results.jsonl`)

  | 조건 | GPU | 낮은 주소 규칙 | 차이 |
  |---|---|---|---|
  | 파일럿 6개 | — | — | 0, 0, 0, +2(qwen), 0, +2(t5-small) |
  | gpt2 b128/128/512 | 21,026 | 21,026 | 0 |
  | gpt2 b128/512/512 | 13,582 | 13,582 | 0 |
  | tinyllama b128/128/256 | 7,104 | 7,104 | 0 |
  | opt-125m b16/128/512 | 968 | 986 | +18 (GPU 자체 변동 968/1012 범위) |
  | t5-base b128/128/512 (재시도) | 16,430 | 15,758 | −672 |
  | tinyllama b128/128/512 (재시도) | 20,730 | 20,198 | −532 |

- 재시도가 일어나면 드라이버가 세그먼트를 반납했다가 그 주소를 다시 씁니다. 단순 규칙은 이 동작을 흉내 내지 못합니다.
  - 그래서 재시도 조건은 정확한 값 대신 §4의 R_inf/R_min 판정으로 다룹니다.
- 미확인
  - 32 MiB 구역 채우기 같은 더 정교한 주소 모델이 opt b16의 +18이나 재시도 조건을 개선하는지
  - 다른 GPU·드라이버에서도 같은 패턴인지

### 3.4 용량별 재생: R_inf / R_min (`tools/capacity_sweep.py`, `results/capacity_sweep/`)

| 조건 | GPU reserved | GPU 최대 allocated | R_inf (용량 1 TiB) | 3090 용량으로 재생 | R_min |
|---|---|---|---|---|---|
| tinyllama b128/128/512 | 20,730 | 4,541 | 40,142 | 20,898 | 6,555 |
| gpt2 b128/512/512 | 13,582 | 10,099 | 13,584 | 13,584 | 13,584 |

- 위 표는 기존 xMem 주소로 계산한 값입니다. 낮은 주소 규칙으로 다시 계산하는 것은 INUOS 작업에 포함됩니다.
- TinyLlama: 넉넉하면 40 GB까지 커지지만, 6.6 GB 이상이면 재시도하면서 돌아갑니다.
- gpt2 512: prefill의 큰 임시 텐서가 동시에 살아 있어 R_min = R_inf입니다. 이보다 작은 GPU에서는 OOM입니다.
- t5-base는 계산이 출력 없이 종료되어 미완료입니다. 메모리 부족으로 추정합니다.

---

## 4. 예측 절차 (제안)

1. **CPU 요청 목록 생성**: 측정 프로토콜과 같은 순서로 만듭니다(로드 → warmup 3 → measure 1).
   - 크기는 FP16 기준으로 환산합니다. 환산이 빠지는 텐서가 없게 합니다. T5에서 2배 차이 요청이 이미 보고됐습니다.
   - attention 경로는 GPU와 같은 요청을 내야 합니다.
     - GPU는 sdpa = FlashAttention이라 L×L 행렬이 없습니다.
     - TinyLlama·Qwen 같은 GQA 모델은 `repeat_kv` 사본이 생깁니다.
   - GPU 전용 작업공간 요청을 GPU와 같은 시점에 넣습니다: cuBLAS 8,519,680 B(영구), cuBLASLt 1 MiB, CNN은 cuDNN 작업공간.
2. **xMem 재생**: 낮은 주소 규칙을 적용하고, 할당 정책은 PyTorch 2.6 기본값을 씁니다(xMem 기본값과 같음).
3. **출력**
   - `R_inf`: 용량 무제한 재생의 measure_0 시작 reserved
   - `R_min`: OOM 없이 끝나는 최소 용량(2 MiB 해상도 이진탐색)
   - 참고값: 최대 allocated
4. **후보 GPU 판정**: 추천 단계에서 처음 GPU 스펙을 씁니다.
   - 사용 가능 용량 C = 총용량 − 드라이버 예약분 − 해당 GPU outside
   - C ≥ R_inf + 마진: 실행 가능, 예측 reserved = R_inf
   - R_min ≤ C < R_inf: 실행 가능하지만 캐시 재시도가 일어나고, reserved는 C 근처까지 찹니다. 응답시간 영향은 미측정입니다.
   - C < R_min: OOM이므로 제외합니다.
5. **정확한 reserved가 필요하면** 후보 GPU의 용량으로 한 번 더 재생합니다. 재시도가 없는 범위에서만 정확합니다.

---

## 5. INUOS 작업 순서와 확인 방법

1. **요청 목록 비교 도구 준비**
   - GPU 쪽은 `traces/<cid>/trace.snapshot.pickle.gz`의 `device_traces[0]`입니다.
     - 이벤트 종류: alloc / free_requested / free_completed / segment_alloc / segment_free
   - 구간 경계는 `trace.json`의 `history.boundaries`에 있습니다. 각 경계에 alloc·free 누적 카운터가 기록돼 있습니다.
     - 경계 이름: model_load, warmup_0~2, measure_0의 begin/end
   - 비교 단위: 구간별 alloc 크기 다중집합 → 순서 → 각 블록의 수명(alloc~free 사이 이벤트 수)
2. **우선 조건 (어려운 순)**
   - `gpt2__b128__in512__new512`: prefill 큰 임시 텐서
   - `gpt2__b128__in128__new512`: KV 증가 단편화
   - `tinyllama-1.1b-chat__b128__in128__new256`: GQA repeat_kv
   - `opt-125m__b16__in128__new512`
   - 재시도 조건: `tinyllama-1.1b-chat__b128__in128__new512`, `t5-base__b128__in128__new512`
   - 파일럿 6개(작은 조건, 이전 handoff)
3. **일치 확인**: CPU 요청 목록을 `tools/replay_addr_policy.py`와 같은 방식으로 재생한 값이 위 §3.3 표의 GPU 값과 같은지 봅니다.
4. **전체 검증**: L256(`results/main395_L256/gpu_rtx3090_main395_L256_per_condition.csv`)과 L1024(이전 handoff `results/rtx3090_main395_L1024/`)
   - 기준값 열: `m1_peak_reserved` (byte)
   - 재시도 27개는 따로 집계합니다.
5. **보고**: `handoff_codex/reserved_prediction_report.md`와 조건별 CSV
   - 조건별 열: 예측 R_inf, R_min, GPU 값, 오차, 원인 분류(요청 차이 / 주소 / 재시도 / GPU 자체 변동)

---

## 6. 파일 (`/home/inuos/cgf-fp16-inuos-handoff-20261116/`)

| 경로 | 내용 |
|---|---|
| `RESERVED_PREDICTION_GUIDE_KO.md` | 이 문서 |
| `traces/<cid>/trace.json`, `trace.snapshot.pickle.gz` | 2026-11-16 수집한 GPU allocator trace 6개. 프로토콜은 load, warmup 3, measure 1이고 worker는 `d06f0f15`, 잘림 없음 |
| `run_trace_slack.sh`, `trace_slack.txt` | trace 수집 스크립트와 로그 |
| `tools/reserved_anatomy.py` | trace → 구간·원인별 세그먼트 증가 분해 (torch 불필요) |
| `tools/xmem_replay.py` | 기본 재생 (구간별 GPU 대비) |
| `tools/xmem_replay_variants.py` | 변형 재생 (작업공간 제거, free_requested, warmup0만, prefill 최대 요청 제거) |
| `tools/first_divergence.py` | xMem이 GPU와 처음 갈라지는 이벤트 찾기 |
| `tools/replay_gpu_addresses.py` | GPU 실제 세그먼트 주소로 재생 (원인 검증용) |
| `tools/replay_addr_policy.py` | 합성 주소 규칙으로 재생 (`descending` 권장) |
| `tools/capacity_sweep.py` | R_inf / R_min 계산 |
| `tools/export_for_inuos.py` | 조건별 CSV 생성기 |
| `results/anatomy/` | 12개 trace 분해 결과 (새 6 + 파일럿 6) |
| `results/xmem_variants/` | 12개 변형 재생 결과 |
| `results/xmem_first_divergence.jsonl` | 첫 불일치 지점 4건 |
| `results/xmem_addr_policy_results.jsonl` | 낮은 주소 규칙 재생 12건 |
| `results/capacity_sweep/` | R_inf / R_min 2건 |
| `results/alloc_retry_conditions.csv` | 캐시 재시도가 일어난 조건 (L1024 + L256) |
| `results/main395_L256/` | L256 본실험 결과 (조건별 CSV, summary, 원본 395×2) |
| `SHA256SUMS` | 체크섬 |

- xMem은 INUGPU 기준 commit `1ee4bd0`입니다. allocator 파일은 수정하지 않았습니다. 주소 규칙은 replay 스크립트에서 monkeypatch로만 적용했습니다.
- 파일럿 trace 6개는 `/home/inuos/cgf-fp16-inuos-handoff-20261112/reference/traces_attn_default/`에 있습니다.

# 추가 결과 (2026-11-16 오후) — reserved 예측 가이드 보충

- 기준 문서: `~/cgf-fp16-inuos-handoff-20261116/RESERVED_PREDICTION_GUIDE_KO.md`
- 이 문서는 그 가이드의 §1.5, §2, §3.3, §5에 보충 내용을 더합니다.
- 내용
  - 모델별 trace 13개 추가 (trace가 없던 모델 위주)
  - 같은 조건 반복 측정 30회
- 측정 환경과 프로토콜은 main395_L256과 같습니다.
  - RTX 3090, FP16, default attention
  - 조건마다 새 컨테이너, warmup 3 + measure 5
  - 조건 파일: `conditions_main395_l256.json`

---

## 0. INUOS 에이전트에게 (작업 프롬프트 보충)

```
기존 RESERVED_PREDICTION_GUIDE_KO.md §0 작업에 다음을 더해 주세요.
1. 검증 대상 trace를 19개로 늘립니다(기존 6개 + 이 폴더 traces/ 13개).
   CPU 요청 목록 → 낮은 주소 규칙(descending) 재생 값이 아래 §2 표의 GPU 값과 같은지 확인합니다.
   (GPU 요청 목록 자체를 재생하면 재시도 없는 10개 모두 0~8 MiB로 맞는다는 것을 INUGPU에서 확인했습니다.
    따라서 차이가 나면 원인은 CPU 요청 목록 쪽입니다.)
2. 새 모델군(GPT-2 medium/xl, T5-3b/small, Qwen b32/512, CNN 3종, ViT-large)의 요청 목록을 먼저 GPU와 맞춥니다.
   CNN은 cuDNN 작업공간 요청(아래 §3)이 GPU와 같은 크기·시점에 들어가야 합니다.
3. 오차 판정 시 GPU 자체 변동 범위(§4)를 허용 오차로 씁니다. 같은 요청 목록이라도 GPU 값이 최대 약 4.5% 갈립니다.
4. 결과는 기존 handoff_codex/reserved_prediction_* 보고서에 19개 trace 기준으로 갱신해 주세요.
```

---

## 1. 추가 trace 13개 (`traces/<cid>/trace.json`, `trace.snapshot.pickle.gz`)

- 생성 스크립트: `run_trace_models_and_repeat.sh`의 A 부분 (label `rtx3090__trace_models`)
- 모두 status=ok, 로그는 `logs/trace_models_and_repeat.txt`에 있습니다.
- reserved 분해: `results/anatomy_trace_models/<cid>.json|.txt` (도구: 기존 `tools/reserved_anatomy.py`)

| 조건 | peak reserved | peak allocated | slack (allocated 대비) | 재시도 | slack을 만든 주요 요청 |
|---|---|---|---|---|---|
| gpt2-medium b128/128/512 | 18,578 | 8,599 | 9,979 (116%) | 7 | KV 증가 (DynamicCache cat) |
| gpt2-xl b32/128/512 | 15,458 | 9,095 | 6,363 (70%) | 3 | KV 증가 |
| opt-125m b128/512/512 | 10,448 | 8,936 | 1,512 (17%) | 0 | prefill 전체 logits B×L×V×2, MLP 활성값 |
| qwen2.5-1.5b b128/128/512 | 6,868 | 5,821 | 1,047 (18%) | 0 | GQA repeat_kv 사본, MLP 활성값 |
| qwen2.5-1.5b b32/512/512 | 4,896 | 4,441 | 455 (10%) | 0 | MLP 활성값 |
| tinyllama b32/512/512 | 3,664 | 3,248 | 416 (13%) | 0 | repeat_kv, MLP 활성값 |
| t5-3b b16/128/512 | 10,268 | 9,360 | 908 (10%) | 2 | 디코더 self-attn KV 증가 |
| t5-3b b8/128/128 | 7,220 | 6,229 | 991 (16%) | 0 | (가중치 위주, 증가 작음) |
| t5-small b128/128/512 | 11,214 | 1,198 | 10,016 (836%) | 0 | 디코더 self-attn KV 증가 |
| resnet-18 b128/r512 | 2,548 | 2,528 | 20 (1%) | 0 | conv 활성값 |
| resnet-50 b128/r512 | 6,168 | 4,091 | 2,077 (51%) | 0 | conv 활성값 + cuDNN 작업공간 |
| mobilenet-v2 b128/r512 | 6,580 | 3,931 | 2,649 (67%) | 0 | 확장 conv 활성값 (B×96×256×256×2) |
| vit-large b128/r224 | 1,594 | 1,272 | 322 (25%) | 0 | 어텐션·MLP 활성값 |

- 단위는 MiB입니다. peak reserved는 measure_0 기준이고, 모든 조건에서 measure_0 시작 reserved와 같습니다.
- 확인된 점
  - KV를 매 스텝 torch.cat으로 늘리는 디코더는 batch가 크면 slack이 allocated보다 커집니다(gpt2-medium 116%, t5-small 836%).
    - 이런 조건은 3090에서도 용량 한계에 닿아 재시도가 일어납니다(gpt2-medium, gpt2-xl, t5-3b b16).
  - CNN도 slack이 클 수 있습니다(mobilenet 67%, resnet-50 51%).
    - warmup_0에서 앞쪽 큰 활성값 블록이 캐시로 남고, 뒤의 작은 요청이 이를 재사용하지 못하는 구조입니다.

---

## 2. GPU 요청 목록 → xMem 재생 (`results/xmem_addr_policy_trace_models.jsonl`)

- 도구: 기존 `tools/replay_addr_policy.py` (policy `descending` = 낮은 주소 규칙, `xmem` = 기존 xMem)
- 실행 스크립트: `run_trace_models_desc.sh` (xmem-gpu 컨테이너, CPU 전용)

| 조건 | 재시도 | GPU (MiB) | 낮은 주소 규칙 | 차이 | 기존 xMem | 차이 |
|---|---|---|---|---|---|---|
| mobilenet-v2-1.0-224__b128__r512 | 0 | 6,580 | 6,580 | +0 | 6,580 | +0 |
| opt-125m__b128__in512__new512 | 0 | 10,448 | 10,448 | +0 | 10,448 | +0 |
| qwen2.5-1.5b-instruct__b128__in128__new512 | 0 | 6,868 | 6,876 | +8 | 6,944 | +76 |
| qwen2.5-1.5b-instruct__b32__in512__new512 | 0 | 4,896 | 4,896 | +0 | 4,896 | +0 |
| resnet-18__b128__r512 | 0 | 2,548 | 2,548 | +0 | 2,548 | +0 |
| resnet-50__b128__r512 | 0 | 6,168 | 6,168 | +0 | 6,168 | +0 |
| t5-3b__b8__in128__new128 | 0 | 7,220 | 7,222 | +2 | 7,220 | +0 |
| t5-small__b128__in128__new512 | 0 | 11,214 | 11,214 | +0 | 11,336 | +122 |
| tinyllama-1.1b-chat__b32__in512__new512 | 0 | 3,664 | 3,664 | +0 | 3,616 | -48 |
| vit-large-patch16-224__b128__r224 | 0 | 1,594 | 1,594 | +0 | 1,594 | +0 |
| gpt2-medium__b128__in128__new512 | 7 | 18,578 | 17,020 | -1,558 | 17,336 | -1,242 |
| gpt2-xl__b32__in128__new512 | 3 | 15,458 | 14,538 | -920 | 14,912 | -546 |
| t5-3b__b16__in128__new512 | 2 | 10,268 | 9,710 | -558 | 9,710 | -558 |

- 결론
  - 재시도 없는 10개: 낮은 주소 규칙이 0~8 MiB 이내로 맞습니다. 기존 trace까지 합치면 재시도 없는 20개 중 19개가 0~8 MiB이고, 1개(opt b16)는 +18 MiB입니다. 이 +18은 GPU 자체 변동 범위 안입니다(§4).
  - 기존 xMem은 재시도 없는 10개 중 3개가 −48~+122 MiB 어긋납니다(qwen b128, t5-small, tinyllama b32). 낮은 주소 규칙이 이것을 모두 바로잡습니다.
  - 재시도 3개: 낮은 주소 규칙이 GPU보다 558~1,558 MiB 작게 나옵니다.
    - 재시도 시 세그먼트를 반납하고 다시 받는 순서를 단순 규칙이 재현하지 못하기 때문입니다.
    - 이런 조건은 가이드 §4의 R_inf / R_min 판정으로 다룹니다(3090에서는 C < R_inf인 경우).
  - 즉 **CPU 요청 목록이 GPU와 같으면, 재시도 없는 조건의 reserved는 낮은 주소 규칙 재생으로 거의 정확히 나옵니다.** 남은 일은 CPU 요청 목록을 GPU와 맞추는 것입니다.
- 미실행
  - 이 13개의 R_inf / R_min 용량 탐색
  - variant 재생(`xmem_replay_variants.py`). 큰 조건은 조건당 30분 이상 걸려 생략했습니다.

---

## 3. CNN·ViT의 GPU 전용 요청 (CPU 요청 목록에 넣어야 하는 것)

- 크기가 4 KiB 배수가 아닌 1 MiB 이상 요청은 cuDNN 작업공간이나 보조 버퍼로 추정합니다. 크기는 cuDNN이 고른 알고리즘에 따라 달라집니다.

| 조건 | 대표 크기 (byte, 횟수) |
|---|---|
| resnet-18 b128/r512 | 536,944,656 ×16, 268,730,384 ×12, 135,397,392 ×12, 71,827,472 ×12, 1,342,202,384 ×4, 402,800,656 ×4 |
| resnet-50 b128/r512 | 135,397,392 ×20, 536,944,656 ×12, 268,730,384 ×12, 71,827,472 ×8, 1,342,202,384 ×4, 671,383,568 ×4 |
| mobilenet-v2 b128/r512 | 202,113,792 ×4, 7,080,946, 5,125,120, 2,562,560 |
| vit-large b128/r224 | 91,488,272 ×4, 608,653,264 ×1 |

- 이 요청들은 GPU에서 cuDNN 알고리즘 선택이 끝난 뒤에야 크기가 정해집니다(benchmark=False, 휴리스틱 선택).
  - CPU에서 정확히 재현하려면 (a) 같은 cuDNN 버전의 휴리스틱 결과를 표로 만들어 쓰거나, (b) GPU trace에서 크기를 뽑아 조건별 상수로 넣는 방법이 있습니다.
  - (b)는 GPU별로 값이 달라질 수 있습니다. 확인되지 않았습니다.
- resnet-50의 1,342,202,384 B(약 1.25 GiB) 요청 하나가 새 세그먼트를 만들어 reserved를 크게 키웁니다(`results/anatomy_trace_models/resnet-50__b128__r512.json`의 segments).

---

## 4. 같은 조건 반복 측정 (GPU 자체 변동)

- 스크립트: `run_trace_models_and_repeat.sh`의 B 부분
  - opt-125m b16/128/512와 b32/128/512를 새 프로세스로 각각 15회 측정했습니다(label `rtx3090__repeat_r01..r15`).
- 원자료
  - `results/repeat_raw/rtx3090__repeat_rXX/<cid>/measure.json`
  - 정리표: `results/repeat_variability_opt125m.csv`

| 조건 | first_measure_peak_reserved (MiB) | 빈도 | peak allocated (MiB) | warmup_0 세그먼트 수 |
|---|---|---|---|---|
| opt-125m b16/128/512 | 968 | 7/15 | 628.75 | 35 |
|  | 1,012 | 8/15 | 627.91 | 37 → 38(warmup_1) |
| opt-125m b32/128/512 | 2,326 | 8/15 | 1,009.56 | 65 |
|  | 2,330 | 7/15 | 1,009.56 | 65 |

- 이전 실행: b16은 L1024 968, L256 1,012, trace 968이었고, b32는 L1024 2,326, L256 2,330이었습니다. 모두 위 두 값 중 하나입니다.
- 확인된 점
  - 모든 반복에서 요청 수가 같습니다(warmup마다 alloc 186,610 / 373,021 / 559,432). 입력과 생성 토큰 수(512)도 같습니다.
  - 그런데도 값이 두 가지로 갈리고, 비율은 거의 반반입니다.
  - b16에서는 peak allocated도 886,784 B 다릅니다. 같은 요청인데 어느 캐시 블록을 쪼개 쓰는지가 달라 블록 크기 합이 달라진 것입니다.
- 추정 원인: cudaMalloc이 돌려주는 세그먼트 주소 순서가 프로세스마다 다릅니다.
  - 가이드 §3.2에서 GPU 실제 주소를 xMem에 넣으면 정확히 일치했습니다.
  - 따라서 주소 순서 → 같은 크기 블록 중 선택 → 쪼개기 결과 → reserved로 이어진다고 봅니다.
  - 이번 반복에는 trace를 남기지 않아 주소를 직접 대조하지는 못했습니다.
- 예측에 주는 의미
  - 같은 요청 목록이어도 GPU 값은 몇 가지 중 하나가 됩니다. 이번 조건에서 차이는 최대 44 MiB(4.5%)였습니다.
  - 낮은 주소 규칙 값(opt b16: 986)은 이 범위(968~1,012) 안에 있습니다.
  - 따라서 오차 판정에 "GPU 자체 변동" 범주를 따로 두고, 마진(가이드의 고정 1 GB)이 이를 덮는지 확인하면 됩니다. 소형 조건에서는 덮습니다. 대형 조건의 변동 폭은 미측정입니다.

---

## 5. 파일 (`~/cgf-fp16-inuos-handoff-20261116b/`)

| 경로 | 내용 |
|---|---|
| `ADDENDUM_20261116b_KO.md` | 이 문서 |
| `traces/<cid>/` | 추가 trace 13개 (trace.json + snapshot) |
| `results/anatomy_trace_models/` | reserved 분해 결과 13개 (json, txt) |
| `results/xmem_addr_policy_trace_models.jsonl` | 낮은 주소 규칙 / 기존 xMem 재생 결과 |
| `results/repeat_variability_opt125m.csv` | 반복 측정 30회 정리 |
| `results/repeat_raw/` | 반복 측정 원자료 (measure.json 30개) |
| `run_trace_models_and_repeat.sh`, `logs/trace_models_and_repeat.txt` | GPU 실행 스크립트와 로그 |
| `run_trace_models_desc.sh` | xMem 재생 실행 스크립트 |
| `SHA256SUMS` | 무결성 확인 (`sha256sum -c SHA256SUMS`) |

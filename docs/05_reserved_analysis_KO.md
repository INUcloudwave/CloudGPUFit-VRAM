# 05. reserved 분석 — 구성 분해, 반복 변동, GPU R_min 실측 (INUGPU RTX 3090, 2026-11-16)

- 이 문서의 독자: 나중에 보고서를 정리할 사람 또는 AI
- 같은 내용의 노션 페이지: "졸업작품 / 실험 정리 / reserved 분석" (E1~E7 번호가 같음)
- 공통 환경
  - RTX 3090 24GB, FP16, transformers 기본 attention (decoder·ViT는 sdpa, T5·CNN은 eager)
  - 이미지 `cloudgpufit-bench:cu124-v1` (torch 2.6.0+cu124), worker `src/vram_worker.py` (sha `d06f0f15`)
  - 조건마다 새 컨테이너, warmup 3 + measure 5. 조건표 `config/conditions_main395_l256.json`
  - 비교 기준값: 첫 measure 구간의 peak reserved

---

## 0. 용어 (보고서에 그대로 써도 되는 정의)

| 용어 | 뜻 |
|---|---|
| allocated | 실제로 살아 있는 텐서 크기의 합 |
| reserved | PyTorch 할당기가 cudaMalloc으로 잡아 둔 메모리. 다 쓴 블록도 캐시로 쥐고 있어서 allocated보다 크다 |
| slack | reserved − allocated. 캐시로 쥐고 있는 여유분 |
| 요청 목록 (trace) | 할당기가 받은 alloc/free 요청의 크기·순서·수명 전체 |
| xMem | PyTorch 할당기를 CPU에서 흉내 내는 시뮬레이터. 요청 목록을 넣으면 reserved를 계산 |
| 재시도 (OOM-retry) | cudaMalloc이 실패하면 할당기가 캐시를 비우고 다시 시도하는 동작 |
| R_inf | GPU 용량이 무제한일 때의 reserved (재시도 없음) |
| R_min | 재시도를 하더라도 OOM 없이 실행되는 가장 작은 메모리 상한 |
| C | 후보 GPU에서 쓸 수 있는 용량 = 총용량 − 드라이버 예약분 − outside |

---

## 1. 결론 요약

1. reserved = 가중치 + 실행 중 최대 allocated + 라이브러리 작업공간 + slack. 예측이 어려운 것은 slack이다.
2. slack은 공식으로 맞힐 수 없다. GPU와 같은 요청 목록을 xMem에 재생해야 맞는다.
3. 요청이 같아도 기존 xMem은 최대 +896 MiB 틀렸다. 원인은 세그먼트 주소 순서다.
   - "새 세그먼트는 항상 더 낮은 주소" 규칙을 넣으면, 재시도 없는 20개 조건 중 19개가 0~8 MiB, 1개가 +18 MiB 안에서 맞았다.
4. **같은 GPU·같은 이미지·같은 입력이어도 reserved가 두 값으로 갈린다 (최대 44 MiB, 4.5%).** (E6, 반복 변동 실험)
   - 따라서 예측 정확도에는 바닥이 있다. 예측은 "한 값"이 아니라 "값 + 마진"으로 내야 한다.
5. **용량이 부족한 GPU에서의 판정 기준 R_min을 xMem이 GPU 실측과 거의 같게 예측한다** (LLM·T5 0 ~ +3.5%, 과소 예측 없음). CNN은 +500 MiB로 보수적이다. (E7, GPU R_min 실측)
6. 재시도 구간의 reserved는 용량 C 근처가 아니고 경로에 따라 달라서 예측할 수 없다. 판정(실행 가능/불가)만 낸다.

---

## 2. reserved 구성 (E1)

| 구성 요소 | 크기 결정 요인 | 비고 |
|---|---|---|
| 가중치 세그먼트 | 파라미터 수 × 2 B | 모델별 상수 |
| 최대 allocated | KV 캐시, prefill logits·MLP 중간값, conv 활성값 | 디코더는 KV + logits로 0.8~0.95배 설명 |
| 라이브러리 작업공간 | cuBLAS 8,519,680 B, cuBLASLt 1 MiB, cuDNN(CNN) | reserved 안에 들어감 (outside 아님) |
| slack | 요청 크기·순서·수명, 세그먼트 주소, GPU 용량 | 모델별 중앙값 8~39%, 극단적으로 t5-small b128 836% |

- slack을 만드는 메커니즘
  - DynamicCache의 `torch.cat` KV 증가 (매 스텝 더 큰 텐서를 새로 만들어 이전 블록이 재사용되지 못함)
  - GQA `repeat_kv` 사본 (TinyLlama, Qwen)
  - prefill 큰 임시 텐서: 입력이 길면 이 블록이 이후 KV 증가를 흡수함. 그래서 출력 길이와 무관하게 같은 reserved가 나옴
    - 예: Qwen2.5-1.5B b128 in256의 출력 128/256이 둘 다 6,728 MiB
  - CNN 앞단 큰 활성값, cuDNN 작업공간 (resnet-50 약 1.25 GiB)
  - 용량 한계 재시도
- peak reserved는 warmup_0(첫 실행)에서 정해진다.
- 방법
  - trace 모드로 할당기 이벤트를 기록했다. 대상은 파일럿 6개, slack이 큰 조건 6개, 모델별 대표 13개다.
  - `tools/reserved_anatomy.py`가 세그먼트를 만든 요청을 config 크기 공식으로 분류했다.
- xMem 쪽 (E2~E5)
  - E2: GPU 요청을 그대로 재생 → 작은 조건은 완전히 일치했으나 큰 조건은 최대 +896 MiB 차이
  - E3: 한 가지만 바꾼 변형 재생 → prefill 최대 요청 하나만 빼도 21,922 → 9,610 MiB로 바뀜
  - E4: GPU 실제 주소를 넣고 재생 → 바이트 단위로 일치. 원인은 주소 순서
  - E5: 낮은 주소 규칙 → 재시도 없는 조건은 0~8 MiB 이내. 재시도 조건 5개는 −532 ~ −1,558 MiB로 맞지 않음

---

## 3. 반복 변동 실험 (E6)

- **목적**: 같은 조건을 같은 환경에서 다시 돌리면 reserved가 항상 같은지 확인한다.
  - xMem과 GPU의 몇십 MiB 차이가 예측 방법의 문제인지, GPU 자체의 흔들림인지 가르기 위해서다.
- **방법**
  - opt-125m b16/128/512와 b32/128/512를 각각 15회 측정했다. 매번 새 프로세스·새 컨테이너로, 본실험과 같은 방식이다.
  - 실행 스크립트: `measurement/local-gpu-vm/05-fp16-vram/run_trace_models_and_repeat.sh` (B 부분)
  - 보조 근거: L1024와 L256 본실험에 겹치는 245개 조건을 비교했다.
- **결과** (`measurement/local-gpu-vm/05-fp16-vram/results/repeat_variability_opt125m.csv`, MiB)

| 조건 | peak reserved | peak allocated | warmup_0 세그먼트 수 | 요청 수 |
|---|---|---|---|---|
| b16/128/512 | **1,012 (8회) / 968 (7회)** | 628~629 | 37 / 35 | 186,610 (모두 같음) |
| b32/128/512 | **2,330 (7회) / 2,326 (8회)** | 1,010 (모두 같음) | 65 (모두 같음) | 186,610 (모두 같음) |

  - 겹치는 245개 조건 중 243개는 바이트 단위로 같았다.
- **해석**
  - 요청 목록은 매번 같다(결정적). 그런데 할당기가 세그먼트를 나누는 방식이 실행마다 달라질 수 있다.
  - cudaMalloc이 돌려주는 주소 순서가 프로세스마다 달라서 생기는 것으로 추정한다(E4와 같은 메커니즘). 직접 증명하지는 않았다.
- **보고서에서의 의미**
  - "동일 조건·동일 환경에서도 실측값이 변할 수 있다"는 직접 증거다.
  - 예측 오차를 평가할 때 이 폭(최대 4.5%)보다 작은 차이는 의미 있는 오차로 볼 수 없다.
  - 2026-11-17 cu128 이미지 비교에서 opt b16이 968/1,012로 갈린 것도 이 현상이다.

---

## 4. GPU R_min 실측 (E7)

- **왜 필요했나 (쉬운 설명)**
  - 큰 GPU에서는 할당기가 캐시를 넉넉히 쥐고 있어 reserved가 R_inf까지 커진다.
  - 작은 GPU에서는 용량에 닿으면 캐시를 비우고 재시도해서, 더 적은 메모리로도 끝까지 실행된다.
  - 그래서 "이 작업이 이 GPU에서 돌아가나?"의 기준은 R_inf가 아니라 R_min이다.
    - 예: t5-small b128은 R_inf가 11.2 GB인데 R_min은 1.4 GB다. R_inf만 보면 4 GB GPU를 탈락시키지만 실제로는 돌아간다.
  - 확인할 것은 두 가지였다.
    1. xMem이 계산한 R_min이 실제 GPU의 OOM 경계와 맞는가
    2. 재시도 구간의 reserved가 C 근처까지 차는가 (당시 예측 가이드의 가정)
- **방법**
  - GPU 측정 코드는 그대로 두고, 래퍼 `src_cap/cap_worker.py`가 `torch.cuda.set_per_process_memory_fraction(cap/total)`로 메모리 상한을 건다.
    - 상한을 넘으면 실제 cudaMalloc 실패와 같은 경로(캐시 비움 → 재시도 → OOM)를 탄다.
  - `tools/gpu_rmin_search.py`가 상한을 바꿔 가며 ok/OOM 경계를 32 MiB 해상도로 이진 탐색했다. 시도마다 새 프로세스를 쓰고, 7개 조건에서 총 55회 실행했다.
    - 러너: `tools/run_conditions_cap.py` (기존 러너에 `--mem-cap-mib` 추가)
  - xMem 쪽: `measurement/local-gpu-vm/03-xmem-probes/reserved-anatomy-20261116/capacity_sweep_desc.py` (낮은 주소 규칙, 2 MiB 해상도)
- **결과** (`measurement/local-gpu-vm/05-fp16-vram/results/rmin_gpu_vs_xmem_desc.csv`, MiB)

| 조건 | 상한 없는 GPU | GPU R_min (OOM, ok] | xMem R_min | 차이 |
|---|---|---|---|---|
| opt-125m b16/128/512 | 968 | (713, 734] | 737 | +3 ~ +24 |
| qwen2.5-1.5b b128/128/512 | 6,868 | (6,853, 6,870] | 6,876 | +6 ~ +23 |
| tinyllama b128/128/256 | 7,104 | (4,212, 4,240] | 4,222 | 범위 안 |
| gpt2 b128/512/512 | 13,582 | (13,556, 13,584] | 13,580 | 범위 안 |
| t5-small b128/128/512 | 11,214 | (1,354, 1,373] | 1,375 | +2 ~ +21 |
| tinyllama b128/128/512 | 20,730 | (5,552, 5,583] | 5,751 | +168 ~ +199 |
| resnet-50 b128/r512 | 6,168 | (4,610, 4,642] | 5,142 | +500 ~ +532 |

- 재시도 구간의 실제 reserved (tinyllama b128/128/512, `results/gpu_rmin_search.jsonl`)

| 상한 C | 20,732 | 12,636 | 8,588 | 6,564 | 5,583 | 5,552 |
|---|---|---|---|---|---|---|
| measure reserved | 8,028 | 8,202 | 7,920 | 5,652 | 5,530 | OOM |
| warmup당 재시도 수 | 2 | 4 | 8 | 18 | 42 | — |

- **해석**
  - LLM·T5: xMem R_min이 실측 구간 안이거나 최대 +3.5% 크다. 과소 예측(OOM 위험 방향)은 없었다.
  - CNN: GPU는 메모리가 부족하면 cuDNN이 작업공간이 작은 알고리즘으로 바꾼다(peak allocated 4,091 → 3,835 MiB). xMem은 이를 재현하지 못해 +500 MiB 크게 잡는다. 보수적 방향이라 판정은 안전하다.
  - 재시도 구간의 reserved는 C보다 훨씬 작고 C를 따라가지도 않는다 → "reserved ≈ C" 가정은 틀렸고 정정했다.
  - 상한이 낮을수록 재시도가 많아진다. 응답시간 영향은 미측정이다.
- **결과로 정한 판정 규칙**

| 조건 | 판정 | 예측 reserved |
|---|---|---|
| C ≥ R_inf | 실행 가능, 재시도 없음 | R_inf |
| R_min ≤ C < R_inf | 실행 가능, 재시도 발생 | 예측하지 않음 (경로 의존) |
| C < R_min | OOM, 후보 제외 | — |

---

## 5. 한계

- 모든 결과는 RTX 3090 한 대, cu124 이미지 기준이다. 다른 GPU에서의 주소 패턴과 R_min은 미확인이다.
- 반복 실험은 opt-125m 2개 조건뿐이다. 큰 조건의 흔들림 폭은 미측정이다.
- GPU R_min 탐색은 조건마다 1회만 했다. R_min 자체의 실행 간 변동은 미측정이다.
- 상한은 `set_per_process_memory_fraction`으로 흉내 낸 것이다. 실제 작은 GPU에서는 드라이버 예약분과 outside를 따로 빼야 한다.
- 재시도 조건 5개는 낮은 주소 규칙으로도 맞지 않는다(R_inf 기준).

---

## 6. 파일 위치 (레포 기준)

| 내용 | 경로 |
|---|---|
| 구성 분해 결과 (조건별 json/txt) | `measurement/local-gpu-vm/05-fp16-vram/results/reserved_anatomy/` (`trace_models/` = 모델별 13개) |
| 분해 도구 | `measurement/local-gpu-vm/05-fp16-vram/tools/reserved_anatomy.py`, `tools/xmem_replay_variants.py` |
| GPU trace 원본 | `measurement/local-gpu-vm/05-fp16-vram/runs/rtx3090__trace_models/`, `runs/rtx3090__trace_slack/` (스크립트 `run_trace_models_and_repeat.sh`, `run_trace_slack.sh`) |
| xMem 재생·주소 규칙·R_inf/R_min | `measurement/local-gpu-vm/03-xmem-probes/reserved-anatomy-20261116/` (스크립트 + `out/`) |
| 반복 변동 | `measurement/local-gpu-vm/05-fp16-vram/results/repeat_variability_opt125m.csv`, 원본 `runs/rtx3090__repeat_r01~r15/` |
| GPU R_min | `measurement/local-gpu-vm/05-fp16-vram/results/gpu_rmin_search.jsonl`(시도별), `gpu_rmin_search_summary.jsonl`(조건별), `rmin_gpu_vs_xmem_desc.csv`(비교), 원본 `runs/rtx3090__capsearch/` |
| GPU 상한 실행 코드 | `measurement/local-gpu-vm/05-fp16-vram/src_cap/cap_worker.py`, `tools/run_conditions_cap.py`, `tools/gpu_rmin_search.py`, `run_gpu_rmin_search.sh` |
| 실행 로그 | `measurement/local-gpu-vm/05-fp16-vram/logs/trace_models_and_repeat.txt`, `trace_slack.txt`, `gpu_rmin_search.txt` |
| INUOS 전달 문서 | `measurement/local-gpu-vm/05-fp16-vram/handoff_inuos/` (예측 가이드, 보충, R_min 검증·정정) |

# GPU 실측 ↔ CPU 예측 정렬표 (2026-11-15, INUOS)

- 기준 GPU 코드: `cgf-fp16-inuos-handoff-20261115/src/vram_worker.py` (sha256 d06f0f15…), `run_conditions.py` (77e3f4a9…)
  - handoff `SHA256SUMS` 전체 검증 통과
- CPU 코드: `src/cgf680/cpu_worker2.py`, `protocol2.py`, `load_emu.py`, `shadow.py`, `tools/run395.py` (커밋 42d96a9)
- 대조 데이터
  - CPU: `runs/main395/cases/*/result.json` (이전 1024 조건표로 실행 중인 결과)
  - GPU: `results/rtx3090_main395_L1024/gpu_rtx3090_main395_L1024_per_condition.csv`
  - 둘 다 ok인 374조건 비교
  - 스크립트·출력: `handoff_codex/alignment_20261115/`
- 판정: 일치 / 불일치(CPU에서 맞출 수 있음) / 구조적 불가

## 1. 항목별 판정

| # | 항목 | GPU | CPU (코드 위치) | 판정 | 근거 |
|---|---|---|---|---|---|
| 1 | 조건표 | `conditions_main395_l256.json` d4a9b09c… | `manifest/conditions_main395.json` 2634bda7… (`cases395.py`) | **불일치** | 겹침 245, 새 조건 150은 CPU 미실행 (§3) |
| 2 | 모델·revision | `models.json` 7b11e3e3… | 같은 파일 (sha 동일) | 일치 | sha256 동일 |
| 3 | 입력 | seed 20261006, randint(3,V), special→3, mask=1 | `protocol2.host_inputs` | 일치 | 입력 sha256 375/375 동일 |
| 4 | dtype | 실제 FP16 실행 | FP32 실행 + shadow dtype으로 FP16 요청 크기 환산 (`shadow.py`) | 구조적 불가 | CPU(Cascade Lake)는 FP16 연산이 매우 느림. LLM·T5 allocated 피크 중앙값 오차 −0.1~−1.5% |
| 5 | attention | 기본값(인자 생략). OPT·TinyLlama·Qwen·GPT-2·ViT=sdpa, T5·CNN=eager | 기본값(인자 생략) | 이름 일치 / 커널 구조적 차이 | `_attn_implementation` 374/374 동일. 실제 커널은 GPU flash/mem-eff, CPU는 CPU용 sdpa 커널. allocated 일치로 보아 L×L 임시 텐서 과대 예측은 없음 |
| 6 | 생성 설정 | from_model_config + greedy, min=max=N, use_cache | `protocol2.generation_config` (같은 코드) | 일치 | 코드 대조 |
| 7 | 측정 순서 | 로드 → warmup 3 → measure 5, 캐시 유지, measure 직전 peak만 리셋 | 로드 합성 → 호출 trace 1개를 warmup 3 + measure 1로 4회 재생, 캐시 유지 (`cpu_worker2.replay_protocol`) | 일치 (비교 기준은 measure_0) | 4회 요청 동일성 36/36 검증 기록. GPU measure 1~4회차는 CPU에 없음 |
| 8 | 텐서 수명 | 회차마다 입력 `.to(cuda)`, 출력과 함께 기록 후 del | `invoke`에서 clone, 마커 M4(기록)·M5(del) | 일치 | 코드 대조 |
| 9 | 기준값 | `summary.first_measure_peak_reserved` 등 | 같은 키 | 일치 | after_load reserved 374/374 오차 0% |
| 10 | 용량/OOM | 25,306,529,792 B, 용량 근처에서 캐시 반환 후 재시도, 실패 시 gpu_oom | xMem 용량 **1 TiB** (`CAP = 1 << 40`) | **불일치** | §2.1. LLM·T5 큰 오차 15건 전부 이 원인 |
| 11 | GPU 전용 할당 | cuBLAS workspace 8,519,680 B, cuBLASLt 1 MiB, cuDNN workspace | 없음 (cuBLAS만 별도 variant 값으로 계산) | 구조적 불가 | CPU trace에 나타나지 않음. 정량화는 GPU trace 필요 |
| 12 | 프로세스 격리 | 조건마다 새 컨테이너 | 조건마다 새 컨테이너 (`run395.py`) | 일치 | 코드 대조 |
| 13 | 단위 | byte | byte | 일치 | |
| 14 | torch 스레드 | 4 | 20 (B단계) / 5 (A단계) | 불일치(영향 미확인) | CPU 커널의 스레드별 임시 버퍼가 CPU trace에 섞일 수 있음. vision 오차와의 관련 미확인 |
| 15 | GPU OOM 조건 | gpu_oom으로 기록 | CPU RAM(42g) 부족으로 예측 자체 실패 | 구조적 제약 | 겹치는 GPU OOM 6건이 CPU에서도 OOM 5 + 미실행 1. FP32 실행이라 메모리가 2배 이상 |

## 2. 주요 불일치 상세

### 2.1 용량 (항목 10)

- reserved 오차 20% 초과인 LLM·T5 15건은 모두 CPU 예측 reserved가 GPU 용량보다 큽니다.
  - 예: t5-base b128/128/1024는 CPU 233,892 MiB, GPU 21,302 MiB입니다.
  - 같은 조건의 allocated는 CPU 5,863 MiB, GPU 5,872 MiB로 일치합니다.
- GPU 측에서는 이 15건 모두 `num_alloc_retries > 0`입니다. allocator가 용량에 닿아 캐시된 segment를 반환하고 재시도했습니다.
- CPU는 1 TiB 용량으로 재생하므로 반환이 일어나지 않고, 캐시가 계속 쌓입니다.
- allocator 밖 GPU 사용량(NVML − reserved)은 312~316 MiB입니다. 따라서 allocator가 실제로 쓸 수 있는 용량은 약 25,306,529,792 − 312 MiB = 24,979,374,080 B로 추정합니다.
- 이 용량으로 다시 재생한 결과: §4 참고 (분석용 재생, 기존 결과는 수정하지 않음)

### 2.2 vision (항목 4·11·14와 관련, 원인 미확정)

- allocated 피크 중앙값 오차
  - resnet-18: −16%
  - resnet-50: −9%
  - mobilenet_v2: +28%
  - vit-large: −1%
- CNN만 크게 어긋납니다. CPU(mkldnn)와 GPU(cuDNN)의 conv 경로 차이로 추정하지만 미확인입니다.
- 확인하려면 GPU trace 모드로 CNN 1~2조건의 요청 기록이 필요합니다.

## 3. 조건표 전환 (항목 1)

- 겹치는 245개
  - CPU ok 239개
  - CPU OOM 5개: gpt2-xl b128 3개, t5-3b b128 2개. GPU에서도 모두 gpu_oom입니다.
  - CPU 미실행 1개: t5-3b b128 512/512. GPU에서도 gpu_oom입니다.
  - 측정 코드와 설정이 같으므로 CPU 결과를 그대로 재사용할 수 있습니다.
- 새 조건 150개(256 포함): CPU는 아직 실행하지 않았습니다.
- 빠진 150개(1024 포함): CPU 결과는 이전 GPU L1024 실측과 대조하는 용도로 보존합니다.
- 현재 CPU 실행(`cgf395-main`)은 이전 표를 쓰고 있습니다.
  - 남은 7개는 모두 1024 조건이거나 GPU OOM 조건입니다.
  - 이어지는 C단계 재시도 6개도 모두 1024 조건입니다.

## 4. 용량 제한 재생 결과

- 대상: LLM·T5에서 reserved 오차가 20%를 넘은 15건. 모두 GPU에서 캐시 반환(`num_alloc_retries > 0`)이 일어난 조건입니다.
- 재생 용량: 24,979,374,080 B
- 결과 파일: `alignment_20261115/replay_capacity_24979374080.jsonl`

| GPU 실측 대비 | 건수 | 조건 |
|---|---|---|
| ±5% 이내 | 9 | qwen b128 128/1024 (−0.0%), gpt2-medium b128 128/512, gpt2-xl b32 128/512, tinyllama b128 128/512, t5-base b128 128/512, t5-3b 4건 |
| 5~10% | 3 | opt b128 128/512 (−6.7%), t5-small b128 128/1024 (+7.4%), t5-base b128 512/1024 (−6.8%) |
| 크게 어긋남 | 3 | opt b128 128/1024 (+38%), tinyllama b128 128/1024 (+48%), t5-base b128 128/1024 (−70%) |

- 1 TiB 재생에서는 GPU 대비 +40%~+1000%였습니다.
- 용량 근처의 reserved는 캐시를 반환하는 시점에 민감합니다. 그래서 이 영역에서는 OOM 판정을 1차 목표로 둡니다.
- 크게 어긋난 3건의 원인 후보는 GPU 전용 할당(cuBLAS 등)과 요청 순서의 작은 차이입니다(미확인).

## 5. 수정 계획 (사용자 확인 전, 아직 적용하지 않음)

1. 조건표 교체
   - `manifest/conditions_main395_l256.json`를 추가하고 `cases395_l256.jsonl`을 생성합니다.
   - 겹치는 245개는 기존 결과를 재사용하고, 새 150개만 실행합니다.
2. 용량
   - xMem 재생 용량을 GPU 유효 용량으로 바꿉니다.
   - 예측 peak가 용량을 넘거나 재생이 실패하면 "OOM 예측"으로 따로 판정합니다.
   - 기존 1 TiB 값도 함께 기록합니다.
3. 스레드
   - GPU와 같은 4로 맞출지 결정이 필요합니다. 속도가 느려집니다.
   - 먼저 소수 조건에서 스레드 수에 따라 요청이 같은지 확인하는 방안을 제안합니다.
4. vision
   - GPU trace 모드 요청 대상을 지정합니다(예: resnet-18 b32 r224, mobilenet_v2 b32 r224).

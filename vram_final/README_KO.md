# 최종 VRAM 예측기 (`vram_final/`)

앱(오케스트레이터)이 부르는 VRAM 예측기다. `experiments/inugpu-inuos` 브랜치의 최종 실험 방식(2026-11-23 실험 종료, 11-26 세대별 cuDNN 표)을 단일 명령으로 묶었다.

## 방법

1. **GPU 적재 단계 요청 합성**: `from_pretrained(fp16, device_map=cuda)` + `.to(fp16)`이 GPU 할당자에 보내는 요청을 체크포인트와 모델 구조로 만든다 (`cgf680/load_emu.py`).
2. **CPU에서 모델 1회 실행**: 입력 복사 + generate(또는 forward) 1회를 PyTorch profiler `[memory]` 이벤트로 기록한다.
   - LLM·T5: CPU에 FP16 명령(avx512_fp16/amx_fp16)이 있으면 실제 FP16 실행(`fp16_real`), 없으면 FP32 실행 + FP16 크기 환산(`fp32_shadow`).
   - 이미지 분류: 항상 `fp32_shadow` + 요청별 층 구분(`cgf680/layer_shadow.py`). 합성곱 결과를 검증한 요청 목록이 이것이다.
   - 환경변수 `CGF_VRAM_CPU_MODE=auto|fp16_real|fp32_shadow`로 바꿀 수 있다.
3. **할당자 재생**: 적재 요청 → 실행 요청 4회(warmup 3 + 측정 1)를 xMem `CachingAllocator`(commit 1ee4bd08, 원본 무수정, 실행 때마다 sha256 검사)에 재생한다.
   - 새 세그먼트 내림차순 주소, 첫 GEMM 출력 직후 cuBLAS 작업 공간, GPU별 실효 용량(재시도·OOM 재현) — `cgf680/replay3.py`
   - 합성곱 모델: CPU 합성곱 라이브러리 버퍼 제거 + GPU 세대별 cuDNN 측정 표 + cuBLASLt 1 MiB + max_pool2d 인덱스 — `final_eval_pergpu.py`와 같은 규칙
   - 합성곱 모델이 GPU 용량을 넘을 때: cuDNN 대체 계획(작업 공간이 작은 다음 후보로 바꿈)까지 재현 — `cgf680/replay5.py`, `results/reserved_v3/fallback/` 규칙. 용량이 충분하면 결과는 위와 같다(108/108).
4. **VRAM(GPU) = 예측 reserved + outside(GPU 종류별 실측 상수)**

| AWS | GPU | outside | cuDNN 표 |
|---|---|---|---|
| g4dn | T4 | 130 MiB | Turing (g4dn 측정) |
| g5 | A10G | 308 MiB | Ampere (RTX 3090 측정) |
| g6 | L4 | 232 MiB | Ada (g6 측정) |
| g6e | L40S | 510 MiB | Ada (g6 측정) |
| g7 | RTX PRO 4500 | 320 MiB | Blackwell (g7e 측정) |
| g7e | RTX PRO 6000 | 686 MiB | Blackwell (g7e 측정) |

## 파일

| 경로 | 내용 | 출처 (experiments/inugpu-inuos) |
|---|---|---|
| `predict.py` | 진입점. `--json -`로 요청을 받아 GPU별 결과 JSON 출력 | 새로 작성 (아래 모듈을 묶음) |
| `cgf680/*.py` | 프로토콜, 요청 기록, FP16 환산, 층 구분, 적재 합성, 재생, GPU 규칙 | `cpu-vm/03-fp16-cpu-prediction/src/cgf680/` 그대로 복사 |
| `cgf_xmem_check.py` | xMem 원본 검사 | `cpu-vm/02-reserved-diag/src/cgf_diag/replay.py`의 `verify_xmem` |
| `data/gpu_profiles.json` | GPU별 용량·outside·cuBLAS 규칙 입력 | `manifest/gpu_profiles.json` (5c39458) |
| `data/cudnn_candidates/*.json` | 세대별 cuDNN 후보 계획 전체의 작업 공간(휴리스틱 순서)과 측정 끝 상태. 대체 계획 재생용 | 위와 같은 probe 파일의 `candidates`·`end` |
| `data/cudnn/*.json` | 세대별 cuDNN 1순위 작업 공간 (형태 4,206개) | `cudnn_version/inugpu_20261123/probe_rtx3090_cudnn9.1.0.jsonl`, `cudnn_fit/pergpu/probes/probe_{aws-g6.xlarge,aws-g4dn.xlarge,g7e}.jsonl` |
| `data/xmem-allocator-lock.json` | xMem 파일 sha256 | `provenance/xmem-allocator-lock.json` |

## 통합 확인 (2026-11-26, Sapphire Rapids CPU, torch 2.6.0, transformers 4.52.4)

같은 조건의 AWS GPU 실측(`aws/gpu_measurements/*/measure.json`의 `first_measure_peak_reserved`)과 비교했다. 6개 GPU 모두 바이트 단위로 같았다.

| 조건 | CPU 방식 | 결과 |
|---|---|---|
| OPT-125M b8 입력 512 생성 128 | fp16_real | 6종 모두 918,552,576 B = 실측. g6 VRAM 1,161,822,208 B = PPT 예시 NVML 실측 |
| T5-small b8 256/128 | fp16_real | 6종 모두 306,184,192 B = 실측 |
| ResNet-18 b8 r224 | fp32_shadow + 층 구분 | g4dn·g6·g6e 88,080,384 / g5 83,886,080 = 실측 |
| ResNet-18 b128 r512 (cuDNN 영향 큼) | fp32_shadow + 층 구분 | T4·A10G 2,671,771,648 / L4·L40S 3,474,980,864 / g7·g7e 3,678,404,608 = 실측 |

- 오프라인 확인이라 가중치는 실제 config로 만든 임의 가중치를 썼다. 요청 크기와 순서는 가중치 값과 무관하다(생성 길이 고정).
- 측정한 cuDNN 표에 없는 합성곱 형태가 나오면 작업 공간을 0으로 두고 경고를 낸다(과소예측 가능).
- 큰 모델은 CPU RAM이 많이 필요하다. 시연은 작은 모델·배치로 하는 것이 빠르다.

## 대체 계획 재생 확인 (2026-11-27)

`results/reserved_v3/fallback/cap_predictions_preregistered.jsonl`(RTX 3090 상한 실측과 15/15 일치한 사전 등록 예측) 중 ResNet-50 b128 r512 7개 지점을 이 예측기의 스트림 생성(`conv_model_stream_fallback`, Ampere 후보 목록)으로 다시 계산했다. 7개 모두 reserved·allocated가 같았다(6,170 MiB: 6,168 / 4,091.1, 5,130~4,642 MiB: 4,630 / 3,835.1, 4,610 MiB: OOM). 대체 계획이 없으면 5,130~4,642 MiB 5개 지점이 OOM으로 판정된다. regnet 8개 지점은 층 구분 파일이 저장소에 없어 다시 계산하지 않았다.

# INUGPU 확인용 트레이스 5개 분석 (받음 2026-11-20, 분석 11-22)

- 트레이스: INUGPU `runs/rtx3090__trace_confirm` (읽기 전용 사본 `/home/inuos/cgf-fp16-inugpu-fetch-20261120/`)
- 요청 이유: `handoff_codex/INUGPU_TRACE_REQUEST_20261116_KO.md`

## 1. GPU 요청 목록 재생 (`verify_guide.py` → `verify.jsonl`)

| 조건 | GPU 실측 | GPU 요청 목록 재생 | CPU 요청 목록 재생 (INUOS) |
|---|---|---|---|
| qwen b128/256/128 | 6728 MiB | 6728 | 6168 (−560) |
| qwen b32/1024/128 | 6728 | 6728 | 6168 (−560) |
| qwen b128/128/128 (대조군) | 4896 | 4898 | 4896 |
| ResNet-50 b128 r224 | 1252 | 1252 | 904 (cuDNN 작업 공간 없음) |
| MobileNet b128 r224 | 1288 | 1288 | 1690 (CPU oneDNN 버퍼) |

- 할당자 모델은 5/5 맞는다. 차이는 요청 목록에서 난다.
- CNN 두 조건은 기존 결론과 같다. 1단계(CPU 버퍼 제거)와 세대별 표로 다룬다(`../../cudnn_fit/`).

## 2. qwen −560 MiB 원인

- 가설 1 (기각): GPU에만 있는 층별 1.5 MiB 요청(배치 × 12 헤드 × 길이 × 4 B, attention 보조 출력으로 추정)
  - GPU 요청 목록에서 이 요청만 빼고 재생해도 6728 그대로였다(`remove_test.py` → `remove_test.jsonl`).
  - cuBLASLt 1 MiB 요청을 빼도 그대로였다.
- 발산 위치 (`diverge.py` → `diverge.json`)
  - 두 목록은 큰 요청(≥ 1 MiB, 한쪽에만 있는 크기 제외) 수가 같다(244,377).
  - 재생 중 reserved가 처음 갈라지는 곳은 warmup_0 첫 층, MLP 중간 텐서(560 MiB) 할당 직전이다.
  - 바로 앞에서 192 MiB 블록(= 배치 128 × 256 × 1536 × 4 B, RMSNorm의 FP32 은닉 상태)의 **해제 순서**가 다르다.
    - GPU: 96 할당 → 96 할당 → 96 해제 → 192 해제
    - CPU: 96 할당 → 192 해제 → 96 할당 → 96 해제
  - 그 결과 GPU에서는 560 MiB 텐서가 기존 빈 공간에 못 들어가 세그먼트가 하나 더 생긴다.
- 원인 (확정)
  - INUOS의 FP32 실행 + FP16 크기 환산이 만든 차이다.
  - FP32로 실행하면 Qwen2RMSNorm의 `hidden_states.to(torch.float32)`는 새 텐서를 만들지 않는다(no-op).
  - shadow 환산은 FP16 실행이었다면 생겼을 할당 크기는 재현하지만, 그 텐서가 해제되는 시점까지는 재현하지 못한다.
  - 근거: 실제 FP16으로 실행한 AWS r7i 요청 목록은 같은 조건을 정확히 맞춘다.
    - b128/256/128: 오차 0
    - RTX 3090 qwen 45개 조건에서 100 MB를 넘는 조건: INUOS 4개, AWS FP16 실행 0개
- 의미: qwen 오차는 GPU나 할당자 문제가 아니라 CPU 실행 방식 문제다. CPU에서 실제 FP16으로 실행하면 사라진다.

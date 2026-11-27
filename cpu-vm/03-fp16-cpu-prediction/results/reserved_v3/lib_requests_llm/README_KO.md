# GPU에만 있는 짧은 요청: LLM·T5·ViT에서도 보이는가 (2026-11-23)

- 방법 (`gpu_only_transients.py`)
  - 3090 트레이스의 measure_0 구간과, 같은 조건의 CPU 요청 목록을 비교한다.
  - "짧은 요청" = 다음 할당 전에 해제되는 할당. 크기별 개수를 비교해 GPU에만 있는 것을 센다.
  - 트레이스 15개: 11-17 받은 것과 새 모델 4개
- 0.5 MiB 이상만 보면 (`gpu_only_transients.json`)

| 모델 | GPU에만 있는 짧은 요청 |
|---|---|
| gpt2-medium, gpt2-xl, opt-125m, qwen 1.5B·0.5B, pythia | 1 MiB (cuBLASLt) 수만 번 |
| T5 (small, 3b) | 없음 (T5 linear에는 bias가 없어 cuBLASLt 경로를 타지 않는 것으로 보임) |
| ViT-large | 1 MiB 145, 87.25 MiB 1 |
| convnext | 1 MiB 37 (블록 안의 Linear), 큰 것 2개 (cuDNN 작업 공간) |
| 층마다 하나씩 크기가 다른 것 | opt 3 MiB×12, pythia 4×24, qwen0.5b 3.5×24, smollm2 3.75×32, tinyllama 2×22 (CPU에는 다른 크기로 있음. 어텐션 임시 버퍼로 추정) |

- 해석
  - cuBLASLt 1 MiB는 CNN만이 아니라 bias가 있는 linear 전부에서 생긴다.
  - 1 MiB 이하라 small pool(2 MiB 세그먼트)에 들어간다. 그래서 reserved 영향은 최대 2 MiB 정도다(LLM 잔여 ±2 MiB와 맞는 크기).
  - LLM에 규칙으로 넣으려면 LLM 층 구분(layer_shadow)이 필요하다. 영향이 작아 보류한다.
  - 층마다 크기가 다른 어텐션 임시 버퍼는 지금 예측(대부분 바이트 일치)에는 영향이 없었다. smollm2 +58 MiB와의 관계는 미확인이다.

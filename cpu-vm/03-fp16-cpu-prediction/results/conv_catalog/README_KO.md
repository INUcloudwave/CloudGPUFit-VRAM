# HF 이미지 분류 모델의 합성곱 층 구성 목록 (2026-11-23)

- 1) `tools/hf_vision_catalog.py`: HF API의 image-classification 공개 메타데이터 → `hf_image_cls_list.jsonl` 30,667개
  - transformers 19,978, timm 2,812, 기타 7,877
- 2) `tools/hf_vision_configs.py`: transformers 모델의 config.json → `hf_configs.jsonl` (19,683 ok, 401/404 289개)
  - 가중치는 받지 않았다. 라벨 목록은 뺐다.
- 3) 구조가 같은 config 묶기 → `unique_configs.json`: **1,021개** (179 model_type)
  - 다운로드 기준 상위 10개가 93.4%, 50개가 98.6%, 100개가 99.4%를 차지한다.
- 4) `tools/conv_catalog_meta.py`: config로 모델을 만들고 meta 장치에서 배치 1, 기본 해상도로 한 번 실행한다 → `meta_convs.jsonl`
  - meta 장치는 크기만 계산하고 데이터와 메모리는 쓰지 않는다. 1,021개에 149초가 걸렸다.
  - 623개 ok (다운로드 기준 96.6%)
  - 실패·제외 398개: model_type 없음 63, timm_wrapper 45 (timm 미설치), 원격 코드(auto_map) 등
- 결과 (`layer_configs.json`)
  - 합성곱 호출 16,079개
  - **서로 다른 층 구성 1,354개** (C_in, C_out, 커널, stride, padding, dilation, groups. 배치와 해상도에는 무관하다)
  - 종류 23개. 많은 순: 1×1 8,691, 7×7 dw 1,548, 3×3 dw 1,318, 3×3 1,131, 5×5 dw 663, 3×3 grouped 445, 2×2 s2 425, …, 패치 임베딩(16×16 s16 등)
- 용도: deviceless가 안 될 때의 대안(층 구성 × 배치·해상도 격자 측정)의 규모를 정한다.
  - 예: 1,354 × 72점 ≈ 9.7만 형태. 3090 측정 속도(초당 약 36개) 기준 약 45분/GPU (추정)
- 큰 파일 `hf_*.jsonl`은 git에 넣지 않는다(위 스크립트로 다시 만들 수 있다).

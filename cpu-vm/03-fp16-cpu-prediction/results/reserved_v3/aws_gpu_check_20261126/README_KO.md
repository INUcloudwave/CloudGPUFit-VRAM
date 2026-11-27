# AWS GPU 확인 다시 실행: g7e(RTX PRO 6000) 추가 (2026-11-26)

- 스크립트: `../aws_gpu_check/aws_check.py` (그대로). 입력에 g7e.2xlarge__l256__cu128 실측을 더했다. 다른 GPU 결과는 11-18과 같다.
- g7e (RTX PRO 6000 Blackwell, 96 GB, cu128 이미지, torch 2.7.1, cuDNN 9.7.1, driver 595.112.07)
  - 395/395 ok (OOM 없음)
  - outside: 684 ~ 692 MiB, 중앙값 686 MiB → `manifest/gpu_profiles.json`의 rtxpro6000 실측 칸을 채움
  - cuBLAS 작업 공간: 디코더 조건 192개에서 3090과 같은 8,519,680 B
  - 주소 규칙 판별용 조건: 3090에서 내림차순과 맞은 47개 중 43개가 g7e에서도 같음 (다른 GPU 48/52와 같은 수준)
  - v3 정확도 (재시도 없는 조건, 100 MB 이내): LLM·T5·ViT 311/321 (96.9%)
  - 합성곱 60개는 세대별 cuDNN 표로 60/60 (`../cudnn_fit/pergpu/`)

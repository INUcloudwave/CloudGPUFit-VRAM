# 소프트웨어·이미지 민감도 사전 설계 (RTX 3090, 2026-11-08)

목적: 같은 GPU/driver에서 컨테이너 이미지·torch·CUDA wheel 버전이 바뀌면 첫 forward 이후 outside가 얼마나 달라지는지 확인. 판정은 사용자 실용 기준 ≤100MiB(practical-outside-criteria.json), 원래 16MiB 기준도 병기.

변형 이미지(images/Dockerfile.variant, images/variants.json): 모두 python:3.11.11-slim-bookworm + pip, 나머지 패키지 S0와 동일 고정(numpy 2.2.2, transformers 4.52.4, accelerate 1.4.0, nvidia-ml-py 12.570.86, Pillow 11.1.0, safetensors 0.8.0, tokenizers 0.21.4).
- S0 대조: cloudgpufit-bench:cu124-v1 (pytorch 공식 conda 이미지 기반, torch 2.6.0+cu124)
- S1: torch 2.6.0 cu124 — base 이미지만 다름 (S0 대비)
- S2: torch 2.6.0 cu126 — CUDA wheel 묶음 (S1 대비)
- S3: torch 2.5.1 cu124 — torch 버전 (S1 대비)
- S4: torch 2.8.0 cu128 — 최신 스택 전체 (S1 대비)

설계: 디스크 제약(여유 약 15GB)으로 변형 이미지는 하나씩 빌드. 각 변형마다 3블록 × {S0, Sx} × {GPT-2, Qwen7B} = 12 fresh worker, 블록 내 순서 셔플. S0를 같은 세션 대조군으로 반복. S1~S4 간 비교는 각 suite의 S0 대조가 일치할 때만 간접 비교.

고정: BF16, B1/T256/N64, eager attention, CUDA_MODULE_LOADING=LAZY, direct, 4 threads, probe.py sha 7df9a02c…, driver 570.211.01, GPU 동일.

주요 결과: 블록 내 (Sx − S0) outside 차이(after_first_forward~final_idle), 모델별. 보조: 초기화/로딩 단계, reserved, device gap, 실제 runtime/cuBLAS/cuDNN 버전(manifest).

품질 기준·outside 정의는 기존과 동일. 실패 시 중단·재시도 없음. 해석 한계: 버전 변경은 종속 라이브러리 묶음 효과이며 단일 요소 인과가 아님. 단일 GPU 환경.

# cuDNN 작업 공간 측정 (GPU 종류별 1회, 3단계)

- 목적: GPU 세대별 cuDNN 작업 공간 표 만들기. 사용자 모델 없이 합성곱 형태만 GPU에서 실행한다.
- 측정하는 것
  - 1순위 작업 공간
  - 메모리가 부족할 때 넘어가는 후보들의 작업 공간 순서 (메모리 상한을 걸어 측정)
  - 같은 조건을 다시 실행했을 때 같은 값인지
- 설정: 측정 작업자와 같다(cudnn.benchmark=False, TF32 off, LAZY, FP16, NCHW).
- 파일
  - `cudnn_ws_probe.py`: 측정 스크립트 (torch만 필요)
  - `conv_signatures.json`: 3,253개 형태
    - 실제 비전 65개 조건의 합성곱 1,180개 (기존 GPU 실측과 대조 가능)
    - 일반 격자 2,073개 (표에 없는 모델용)
  - `make_signatures.py`: 위 목록 생성 (INUOS)
  - `run_probe.sh`: 도커 실행 (측정과 같은 이미지)
- 실행 (GPU에서)
  ```bash
  GPU_LABEL=rtx3090 IMAGE=cloudgpufit-bench:cu124-v1 bash run_probe.sh --src model --limit 50   # 시험
  GPU_LABEL=rtx3090 IMAGE=cloudgpufit-bench:cu124-v1 bash run_probe.sh                          # 전체
  # Blackwell(g7, g7e)는 IMAGE=cloudgpufit-bench:cu128-v1
  ```
  - 출력: `probe_<GPU_LABEL>.jsonl`
  - 중간에 끊겨도 같은 명령으로 이어서 돈다.
- 3090에서 먼저 확인할 것
  - 트레이스에서 얻은 값과 같은지 (`results/reserved_v3/cudnn_fit/ws_3090_traces*.jsonl`. 예: ResNet 7×7 첫 층 b128 r512 = 1280.02 MiB)
  - 후보 순서 측정이 정상적으로 끝나는지
  - 전체 걸린 시간
- 왜 형태 그대로 재야 하나 (2026-11-20 확인)
  - INUGPU 새 트레이스(ResNet-50·MobileNet b128 r224)를 보면, 큰 작업 공간(첫 층, 3×3)은 r512에서 뽑은 배수와 맞는다.
  - 공간 크기가 작은 1×1 층은 r512에서는 0이었는데 r224에서는 9~32 MiB였다. 해상도에 따라 다른 알고리즘을 고른다.

## 버전 기록
- v1 (11-20): 첫 버전. INUGPU 시험에서 후보 순서 문제 발견
- v2 (11-20, `4160f36`): x, w 할당 전 캐시 비우기, 1 MiB 이하에서 멈춤, 멈춘 이유(`end`) 기록
  - 3090 전체 결과 `results/rtx3090/`: 1순위는 정상, 후보 순서는 724개가 `no_change`
- v3 (11-22, INUGPU 제안을 INUOS가 검토·반영)
  - 상한을 실제로 새로 생긴 세그먼트로 계산한다(memory_snapshot 전후 비교). 작업 공간 세그먼트 − 2 MiB, 출력 세그먼트는 실제 크기
  - 기존 세그먼트에 들어간 작업 공간은 `shared_segment_limit`으로 정상 종료
  - 더 큰 작업 공간으로 넘어가면 `larger_fallback`
  - INUGPU 시험(3090 전체): no_change 724 → 3, 1순위 값 3,253개 모두 v2와 같음, 94초
  - 결과에 `ws_seg`(작업 공간 세그먼트 크기)와 `out_seg`가 추가된다. reserved에 영향을 주는 것은 세그먼트 크기다.

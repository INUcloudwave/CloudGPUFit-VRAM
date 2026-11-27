# cuDNN 버전 영향과 deviceless 확인 (RTX 3090, 2026-11-23)

- 측정: INUGPU, 같은 이미지(torch 2.6.0+cu124)에 PyPI cuDNN을 LD_PRELOAD로 바꿔 끼웠다. 형태 4,206개, 버전마다 4,206 ok
  - 각 결과 첫 줄의 environment.cudnn으로 실제 로드된 버전을 확인했다.
  - 원본: `inugpu_20261123/` (스크립트 `gpu_probe/deviceless/run_version_test.sh`)
- 재현성: 9.1.0을 다시 측정한 결과가 이전 측정(3,253 + 953)과 4,206/4,206 같았다.

## 1. 버전 영향 (`version_compare.py`, `version_impact.py`)

| 버전 | 1순위 작업 공간이 9.1과 같은 형태 | 9.1 표로 예측한 CNN 108개 조건 (해당 버전 표 기준 차이) |
|---|---|---|
| 9.5.1 (torch 2.7 기본) = 9.7.1 (cu128 이미지) | 3,671/4,206 | 108/108이 100 MB 이내, 93개 같음, −36 ~ +10 MiB |
| 9.10.2 (torch 2.8~2.10 기본) = 9.12.0 | 3,649 | 105/108, 102개 같음, 최대 −1,538 MiB (resnet-50 b128 r512) |
| 9.28.0 | 3,591 | 106/108, 85개 같음, −94 ~ +190 MiB |

- 버전은 묶음으로 바뀐다: 9.1 | 9.5 = 9.7 | 9.10 = 9.12 | 9.28
- 85% 이상의 형태는 버전과 무관하지만, 큰 조건에서는 GB 단위로 차이가 날 수 있다. 표는 "cuDNN 묶음 × GPU 세대"마다 필요하다.

## 2. deviceless (cuDNN 9.28 + frontend 1.32.2, `gpu_probe/deviceless/deviceless_query.py`)

- INUGPU 진단·수정 제안(수용): deviceless 실행에 cuBLAS·cudart 경로가 빠져 강제 종료됐다. 고친 실행 결과는 `inugpu_diag/deviceless_DIAG2_cublas_cudart.jsonl` (4,206 오류 없음, 281 s)
- GPU 쪽(같은 9.28) 실측 1순위와 비교 (`deviceless_match.py`)
  - deviceless 1순위가 실측과 같다: 1,513/4,206 (36%)
    - 작업 공간이 그대로 같은 것 391개
    - 1,122개는 "작업 공간 + 출력 크기" (PyTorch가 출력 크기만큼 한 번 더 할당한다. 주로 1×1)
  - 실측값이 후보 목록 어딘가에 있다: 3,425 (81%)
  - PyTorch(2.6, frontend 내장본)는 상위 후보 중 eng22/47/51/45/57 등을 자주 건너뛰고 eng0·eng28을 고른다. 선택 규칙은 아직 모델링하지 못했다.
- INUOS CPU 전용(GPU·드라이버 없음, libcuda stub)
  - `check_support`가 cudaGetDevice를 불러 실패한다 → `--no-check-support` 옵션을 추가했다.
  - 결과 `gpu_probe/deviceless/deviceless_rtx3090_cudnn9.28.0.42_inuos_cpu.jsonl`: 4,206 오류 없음, 3,177 s (4코어)
  - GPU 쪽 deviceless와 비교
    - CPU 목록은 GPU 쪽 목록의 부분집합이다. 순서와 작업 공간은 모두 같다.
    - eng3, eng4, eng6, eng58이 CPU에서는 빠진다. 장치가 있어야 만들어지는 엔진으로 추정한다.
    - 1순위가 같은 형태: 4,095/4,206
- 결론
  - deviceless로 CPU에서 후보와 작업 공간을 얻을 수 있다(일부 엔진 제외).
  - 그러나 PyTorch가 실제로 고르는 계획과 맞추는 규칙이 없다. 그래서 현재 예측 방법은 측정 표를 유지한다.
  - 또 PyPI 기본 torch(2.8~2.10)는 cuDNN 9.10.2라서, deviceless 정식 버전(9.12 이상)과 기본 조합이 없다.

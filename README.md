# CloudGPUFit-VRAM

GPU 없이 CPU 실행만으로 딥러닝 추론 워크로드의 GPU VRAM 사용량을 예측하는, CloudGPUFit의 **VRAM 예측 파트** 저장소입니다.
GPU 실측 데이터, CPU 기반 예측 코드와 결과, 그리고 예측 방식을 다듬어 온 실험 기록을 담고 있습니다.

- 통합 시뮬레이터: https://github.com/INUcloudwave/CloudGPUFit
- 졸업논문: 「CloudGPUFit: 클라우드 GPU 비용-성능 추천 시뮬레이터」(2026)
- VRAM 파트 담당: 김연준

## 예측 방식

```
예측 VRAM = 예측 reserved + outside
```

- **reserved**: PyTorch CUDA caching allocator가 확보하는 메모리입니다.
  1. 모델을 CPU에서 한 번 실행하면서 메모리 할당 요청 목록을 기록합니다. FP16 명령을 지원하는 CPU에서는 실제 FP16으로 실행합니다.
  2. GPU 적재 단계의 요청을 체크포인트와 모델 구조로 합성합니다.
  3. 두 요청 목록을 [xMem](https://github.com/Stone-ResearchLife/xMem)의 할당자 시뮬레이터에 재생합니다. 이때 GPU와 다르게 동작하는 부분(새 세그먼트 주소 순서, cuBLAS 작업 공간, GPU별 실효 용량, 합성곱의 cuDNN 작업 공간)을 보정합니다.
- **outside**: 할당자 밖에서 쓰이는 메모리(CUDA 컨텍스트 등)입니다. GPU 종류별로 실측한 상수를 씁니다.

| AWS 인스턴스 | GPU | outside |
|---|---|---|
| g4dn | T4 | 130 MiB |
| g5 | A10G | 308 MiB |
| g6 | L4 | 232 MiB |
| g6e | L40S | 510 MiB |
| g7 | RTX PRO 4500 | 320 MiB |
| g7e | RTX PRO 6000 | 686 MiB |

자세한 설명은 [`prediction/final/README_KO.md`](prediction/final/README_KO.md)를 참고하세요.

## 주요 결과

- 검증 조건: 13개 모델 × 395개 조건 (배치 1·8·16·32·128, 입력·생성 길이와 이미지 해상도 조합)
- 정답값: 로컬 RTX 3090과 AWS GPU 인스턴스 6종(g4dn, g5, g6, g6e, g7, g7e)의 실측 peak reserved
- 정확도 기준: |예측 − 실측| ≤ 100 MB(10⁸ B)인 조건의 비율
- 아래는 모두 최종 방식(CPU에서 실제 FP16 실행) 기준이며, 수치는 졸업논문과 같습니다.

**로컬 RTX 3090** (GPU에서 성공한 384개 조건, 논문 <표 14>)

| 조건 그룹 | 조건 수 | 100 MB 이내 |
|---|---|---|
| 재시도 없는 트랜스포머 조건 | 312 | 300 (96.2%) |
| 할당기 재시도 조건 | 12 | 6 (50.0%) |
| 합성곱 모델 (CNN) | 60 | 60 (100%) |
| **전체** | **384** | **366 (95.3%)** |

- 바이트 단위 일치: 266개
- GPU에서 OOM이 난 11개 조건은 모두 OOM으로 판정했습니다.
- 재시도 구간의 reserved 값은 오차가 커서, 시스템에서는 실행 가능 여부만 판정합니다.

**AWS GPU 6종** (재시도 없는 LLM·T5·ViT 조건, 논문 <그림 6>)

| 인스턴스 | GPU | 100 MB 이내 |
|---|---|---|
| g4dn | T4 | 90.7% (274/302) |
| g5 | A10G | 95.8% (296/309) |
| g6 | L4 | 96.4% (298/309) |
| g6e | L40S | 95.0% (303/319) |
| g7 | RTX PRO 4500 | 95.9% (301/314) |
| g7e | RTX PRO 6000 | 94.7% (304/321) |

- 실행 가능 여부(OOM) 판정: AWS GPU 6종의 395개 조건 모두에서 실측과 일치했습니다 (논문 <표 31>).
- 예측 규칙을 만들 때 쓰지 않은 LLM 3개의 48개 조건도 모두 100 MB 이내였습니다 (논문 <표 19>).

## 저장소 구조

```
CloudGPUFit-VRAM/
├── measurement/                 # GPU 실측 (예측의 정답값)
│   ├── local-gpu-vm/            #   로컬 GPU VM (RTX 3090): 측정 코드, 실측 결과, 오차 원인 분석
│   └── aws/                     #   AWS GPU 인스턴스 실측 (README_KO.md에 전체 설명)
├── prediction/                  # CPU 기반 예측
│   ├── local-cpu-vm/            #   로컬 CPU VM: 예측기 개발과 예측 결과
│   ├── aws-r7i-cpu/             #   AWS r7i(FP16 지원 CPU)에서 실제 FP16으로 실행한 예측
│   ├── final/                   #   최종 예측 코드 (통합 시뮬레이터가 호출하는 예측기)
│   └── xMem/                    #   xMem 수정 패치와 xMem 라이선스
├── docs/                        # 실험 전체 지도와 분석 문서
│   └── ai-context/              #   AI 작업용으로 쓴 통합 맥락 문서
└── LICENSE
```

- 실험 전체를 한눈에 보려면 [`docs/00_REPORT_MAP_KO.md`](docs/00_REPORT_MAP_KO.md)부터 읽으세요.
- 문서 안의 `INUGPU`는 로컬 GPU VM, `INUOS`는 로컬 CPU VM을 가리킵니다.

## 실험 흐름

각 폴더 앞의 번호가 진행 순서입니다.

| 시기 (2026) | 내용 | 폴더 |
|---|---|---|
| 10월 | GPU 정답값 1차 측정 (342개 조건) | `measurement/local-gpu-vm/01-cloudgpufit-bench` |
| 10월 ~ 11월 초 | CPU 예측기 개발, 요청 목록 기록 + xMem 재생 방식 확정 | `prediction/local-cpu-vm/00-early-eval-recovery`, `01-cpu-predictor-342` |
| 11월 초 | outside 정의와 GPU별 측정 | `measurement/local-gpu-vm/02-outside-analysis` |
| 11월 중순 | reserved 오차 원인 분석, 측정 방식 재설계 | `measurement/local-gpu-vm/03-xmem-probes`, `04-reserved-diag`, `prediction/local-cpu-vm/02-reserved-diag` |
| 11월 중순 | FP16 실측과 예측 (395개 조건) | `measurement/local-gpu-vm/05-fp16-vram`, `prediction/local-cpu-vm/03-fp16-cpu-prediction` |
| 11월 중순 ~ 하순 | AWS GPU 6종 실측, r7i CPU 실제 FP16 예측 | `measurement/aws`, `prediction/aws-r7i-cpu` |
| 11월 하순 | 최종 예측기 통합 | `prediction/final` |

날짜별 상세 이력은 `docs/00_REPORT_MAP_KO.md` 2절에 있습니다.

## 재현

- 예측: [`prediction/final/README_KO.md`](prediction/final/README_KO.md)
  - xMem은 commit `1ee4bd08`을 받아 원본 그대로 사용합니다. 할당자 파일은 실행할 때마다 sha256으로 검사합니다.
  - 통합 확인 환경: torch 2.6.0, transformers 4.52.4
- AWS r7i CPU 예측: `prediction/local-cpu-vm/03-fp16-cpu-prediction/aws/README_AWS_KO.md`
- GPU 실측: `measurement/local-gpu-vm/05-fp16-vram/aws/`의 측정 패키지

## 자료에 관한 참고 사항

- **저장소에 없는 파일**: 50MB를 넘는 파일 일부는 GitHub 업로드 제약 때문에 넣지 않았습니다. 실험에 쓴 VM이 삭제되어 **원본은 더 이상 남아 있지 않습니다.** 목록·크기·sha256은 [`measurement/local-gpu-vm/EXCLUDED_LARGE_FILES.md`](measurement/local-gpu-vm/EXCLUDED_LARGE_FILES.md), [`prediction/local-cpu-vm/EXCLUDED_LARGE_FILES.md`](prediction/local-cpu-vm/EXCLUDED_LARGE_FILES.md), `measurement/aws/README_KO.md` 4절에 기록으로만 남깁니다. 분석 결과와 요약은 저장소의 결과 파일과 문서로 확인할 수 있습니다.
- **재사용 결과의 복사본**: `prediction/local-cpu-vm/03-fp16-cpu-prediction/runs/main395_l256/cases/`의 244개 조건은 원래 `../../main395/cases/<조건>`을 가리키는 심볼릭 링크였습니다(길이 1024 실험과 겹치는 조건의 결과 재사용, 목록은 `runs/main395_l256/reuse_manifest.json`). 업로드 과정에서 같은 내용의 실제 폴더로 복사되었습니다. 문서의 "심볼릭 링크" 설명은 이 복사본을 가리킵니다.
- **gr6.4xlarge**: 측정은 했지만 논문 대상에서 제외했습니다(`measurement/aws/gr6.4xlarge-l256/NOTE_KO.md`).
- **경로 표기**: 저장소를 정리하면서 폴더 위치를 바꿨습니다. 문서(.md) 안의 경로는 새 위치로 고쳤지만, 결과 데이터 파일(json, csv, log) 안의 경로는 실험 당시 기록 그대로입니다. 옛 경로는 다음과 같이 대응됩니다.

  | 실험 당시 | 이 저장소 |
  |---|---|
  | `inugpu/` | `measurement/local-gpu-vm/` |
  | `inuos/` | `prediction/local-cpu-vm/` |
  | `aws/gpu_measurements/` | `measurement/aws/` |
  | `aws/cpu_fp16_real_r7i/` | `prediction/aws-r7i-cpu/` |
  | `vram_final/` | `prediction/final/` |

## Acknowledgement / 참고 및 인용

이 저장소의 reserved 예측은 xMem의 CPU 기반 GPU 메모리 추정 방식과 할당자 시뮬레이터를 바탕으로 합니다.
xMem은 원래 **학습(training)** 워크로드를 대상으로 하며, 이 프로젝트는 이를 **추론(inference)** 워크로드에 적용하면서 생기는 차이를 보정했습니다.

- 원본 저장소: https://github.com/Stone-ResearchLife/xMem (MIT License, Copyright (c) 2025 Jiabo Shi)
- 이 저장소에는 xMem 원본 코드가 들어 있지 않습니다. 실행할 때 commit `1ee4bd08`을 받아 사용합니다.
- 추론 실행을 위해 xMem 파일 3개를 고치는 패치(`prediction/xMem/patches/core_inference.patch`)가 있으며, 할당자(`perf_estimator/allocator`)는 원본을 수정 없이 사용합니다.
- 이 프로젝트는 xMem 저자들과 관계가 없으며, 저자들의 검증이나 보증을 받은 것이 아닙니다.

xMem을 참고하신다면 원 논문을 인용해 주세요.

```bibtex
@inproceedings{jiaboXMemCPUBasedApproach2025,
  title     = {xMem: A CPU-Based Approach for Accurate Estimation of GPU Memory in Deep Learning Training Workloads},
  author    = {Shi, Jiabo and Pezaros, Dimitrios and Elkhatib, Yehia},
  booktitle = {Proceedings of the 26th International Middleware Conference},
  year      = {2025},
  month     = dec,
  publisher = {ACM},
  address   = {Nashville, TN, USA},
  doi       = {10.1145/3721462.3770773},
  isbn      = {979-8-4007-1554-9}
}
```

## License

MIT License (Copyright (c) 2026 INUcloudwave). 단, `prediction/xMem/` 아래 파일은 xMem의 MIT License(Copyright (c) 2025 Jiabo Shi)를 따릅니다.

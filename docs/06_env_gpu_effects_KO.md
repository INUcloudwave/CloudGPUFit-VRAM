# 06. 실행 환경이 VRAM 측정값에 주는 영향 — 드라이버 · PyTorch 배포판 · CUDA/cuDNN · GPU 종류, outside (2026-11-16 ~ 11-18)

- 이 문서의 독자: 나중에 보고서를 쓰는 사람이나 AI. 실험의 동기, 방법, 결과, 결론, 반영한 내용, 진행 중 생긴 문제를 모두 적었다.
- 앞 문서
  - 측정 방식: `docs/04_fp16_vram.md`
  - reserved 구성과 예측: `docs/05_reserved_analysis_KO.md`
  - 코드·이미지 계보: `docs/07_code_env_lineage_KO.md`
- 용어
  - outside = 같은 시점의 `NVML 프로세스 사용량 − torch.cuda.memory_reserved()`. CUDA 컨텍스트, 커널 코드, 라이브러리 핸들 등 할당기 밖의 메모리다.
  - 비교 기준값 = 첫 측정 회차의 peak reserved / peak allocated (`summary.first_measure_peak_*`)
  - 모든 원값은 byte, MiB = 2^20 B

---

## 0. 한눈에 보는 결론

| 질문 | 답 | 근거 |
|---|---|---|
| 드라이버 버전이 바뀌면 VRAM이 바뀌나 | **아니다** (T4, R595.112.07 → R580.196.07) | 39조건 allocated·reserved 바이트 단위 동일, outside 최대 2 MiB(NVML 분해능) |
| PyTorch 배포판(2.6.0 → 2.5.1, 같은 CUDA 12.4·cuDNN 9.1)이 바뀌면 | **메모리는 아니다**. 단 호환성 문제가 생긴다 | 실행된 33조건 동일. opt·vit 6조건은 transformers 4.52.4가 torch < 2.6에서 `.bin` 가중치 로딩을 거부 |
| CUDA 12.4 → 12.8, torch 2.6 → 2.7, cuDNN 9.1 → 9.7 (RTX 3090) | **대부분 같다** | 384조건 중 355(92%) reserved 동일. CNN·ViT만 수십 MiB(최대 −36 MiB, −12.9%) 차이 → cuDNN 영향으로 추정 |
| GPU 종류(T4 ↔ 3090, 같은 이미지) | **allocated는 거의 같고, reserved는 용량에 따라 달라진다** | allocated는 378조건 중 291 동일, 나머지 ±2.5 MiB. reserved 동일은 221/378. 큰 차이는 T4 용량(16 GB)에서 재시도가 생긴 조건 |
| outside는 무엇이 정하나 | **GPU 종류**. 모델·배치·드라이버·PyTorch·CUDA와 무관 | T4 130/132 MiB, 3090 312/314/316 MiB (395조건 전수). 이전 연구의 L4 232, A10G 308, L40S 510과 같은 계열 |
| 같은 조건을 다른 날·다른 인스턴스·다른 물리 GPU에서 재면 | **T4에서는 바이트 단위로 같다** | g4dn 11-16 실측 vs 11-17 A: 39/39 동일. 재시작 후 물리 GPU가 바뀐 확인 재실행도 동일 |

- 예측기에 반영할 것 (§7)
  - outside는 GPU 종류별 상수로 둔다.
  - 드라이버 버전은 입력에서 뺀다.
  - PyTorch 버전은 호환성 조건으로 다룬다.
  - CNN 보정은 GPU × cuDNN 버전 단위로 한다.
  - reserved는 GPU 용량을 넣어 재시도를 흉내 내야 한다.

---

## 1. 실험 목록과 동기

| # | 실험 | 기간 | 왜 했나 | 어디서 |
|---|---|---|---|---|
| E8 | g4dn.xlarge(T4) L256 395조건 실측 | 11-16 | 3090과 같은 패키지로 다른 GPU 정답값 확보 (AWS 실행 패키지 `cgf-fp16-aws-full-20261115.tar`) | AWS, 사용자 실행 |
| E9 | g7.2xlarge(Blackwell) 실측 시도 → cu128 이미지 제작 | 11-17 | 신형 GPU 정답값. 기존 이미지가 Blackwell을 지원하지 않아 실패 → 이미지 추가 | AWS + INUGPU |
| E10 | RTX 3090에서 cu124 ↔ cu128 이미지 비교 (395조건) | 11-17 | g7은 cu128로만 잴 수 있다. 그래서 g7과 3090의 차이를 "버전 차이"와 "GPU 차이"로 나누려면 같은 3090에서 버전만 바꾼 기준이 필요 | INUGPU |
| E11 | g4dn 환경 비교 A → C → B (39조건 × 3) | 11-17 ~ 11-18 | 팀 요청: 드라이버와 PyTorch 배포 환경이 VRAM·outside에 각각 어떤 영향을 주는지 분리 | AWS, 사용자 실행 (INUGPU 패키지) |
| — | T4 ↔ 3090 비교 (E8 + 기존 3090 L256) | 11-18 분석 | GPU 종류 효과. 예측 전제("요청 목록은 GPU와 무관")를 점검 | INUGPU 분석 |

---

## 2. E8 — g4dn.xlarge(T4) L256 실측

- 방법: 3090 L256과 같은 번들, 같은 이미지(`cu124-v1`, `457da255`), 같은 워커(`d06f0f15`), 같은 조건표(`d4a9b09c`)
  - 순서: pilot 26개 PASS → measure 395 → validate 395 (`aws/RUN_ON_AWS.sh`)
- 환경 (395개 모두 같음)
  - Tesla T4 15,360 MiB, 드라이버 595.112.07, 커널 7.0.0-1017-aws
  - torch 스레드 4, `CUDA_MODULE_LOADING=LAZY`, `PYTORCH_CUDA_ALLOC_CONF` 미설정
- 결과: measure **378 ok / 17 gpu_oom**, validate 370 ok / 25 gpu_oom
  - OOM은 3090보다 6개 많다(gpt2-medium b128 in512 2개, t5-3b 4개). T4 용량이 작기 때문이다.
- attention 구현: sdpa 200 / eager 195 (3090과 같은 분포)
- 파일
  - `gpu-vm/05-fp16-vram/runs/aws-g4dn.xlarge__l256/` (원본)
  - `results/gpu_g4dn.xlarge_T4_main395_L256_per_condition.csv` (조건별 1행, byte)
  - 로그: `logs/aws-g4dn.xlarge__l256/` (내부 IP는 `ip-REDACTED`로 가림)

## 3. E9 — g7(Blackwell)과 cu128 이미지

- 문제
  - g7.2xlarge(NVIDIA RTX PRO 4500 Blackwell Server Edition, compute capability 12.0, 드라이버 595.112.07)에서 pilot 26개가 모두 3.3초 만에 `compat_or_runtime_error`로 실패했다.
- 원인 (확인됨)
  - `cu124-v1` 안의 torch 2.6.0+cu124는 sm_50 ~ sm_90 커널만 들어 있다 (`torch._C._cuda_getArchFlags()`).
  - Blackwell(sm_120)에는 CUDA 12.8 이상, torch 2.7 이상 빌드가 필요하다.
- 조치: `cloudgpufit-bench:cu128-v1` 제작 (`gpu-vm/05-fp16-vram/aws/Dockerfile.cu128`, `aws/IMAGE_INFO_cu128.json`)
  - base: `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`, digest로 고정 (`sha256:c16f4c74…`)
  - pip로 추가한 9개 패키지(transformers 4.52.4, accelerate 1.4.0, huggingface_hub 0.36.2, tokenizers 0.21.4, safetensors 0.8.0 등)는 cu124-v1과 같은 버전으로 고정
  - 같이 바뀐 것: python 3.11.11 → 3.11.13, cuDNN 9.1.0 → 9.7.1, cuBLAS 12.4 → 12.8, numpy 2.2.2 → 2.2.6
  - 지원 GPU 구조: sm_75 ~ sm_120 (T4·3090·Blackwell 모두 가능)
- AWS 스크립트에 이미지 선택 기능을 추가했다.
  - `CGF_IMAGE_VARIANT=cu124|cu128` 또는 번들 안의 `IMAGE_VARIANT` 파일로 고른다.
  - cu128로 돌리면 라벨 끝에 `__cu128`이 붙어 기존 결과와 섞이지 않는다.
  - 원본 스크립트는 `*.v20261115`로 보존했다.
- 상태
  - 번들 `cgf-fp16-aws-cu128-20261117.tar`(sha256 `5030113d…`)로 사용자가 g7 측정을 마쳤다(라벨 `aws-g7.2xlarge__l256__cu128`).
  - **결과 tar는 아직 INUGPU에 오지 않아 분석하지 못했다.** E10과 묶어 분석할 예정이다.

## 4. E10 — RTX 3090에서 cu124 ↔ cu128 (버전 효과)

- 방법: 같은 3090, 드라이버 570.211.01, 같은 워커·러너·조건표. 이미지만 cu128-v1로 바꿈 (라벨 `rtx3090__main395_l256__cu128`, `run_cu128_rtx3090.sh`)
  - pilot 26 PASS 후 measure 395 + validate 395 (11-17 13:59 ~ 21:30)
- 결과 (`results/version_gpu_compare/rtx3090_cu124_vs_cu128_summary.md`, 조건별 `…_per_condition.csv`)
  - 상태: measure (ok, ok) 384 / (oom, oom) 11, validate 377 / 18. **상태가 바뀐 조건 0개**
  - peak reserved 바이트 단위 동일 **355/384 (92%)**

| 모델 | 동일 | reserved 차이 (MiB) | 해석 |
|---|---|---|---|
| qwen, t5-small, t5-3b, gpt2-medium, gpt2-xl, mobilenet | 전부 | 0 | — |
| gpt2 / t5-base | 19/20, 44/45 | +20 / −12 | 반복 변동 범위 |
| opt-125m | 41/45 | −44, −18, −4, +18 | −44는 같은 이미지에서도 968/1,012로 갈리던 조건 (`docs/05` E6) |
| tinyllama | 39/45 | −8 ~ +26, +204 1개 | +204(b128/128/256) 원인 미확인 |
| resnet-18 / resnet-50 | 13/20 | −36 ~ +10 / −18 ~ +10 | cuDNN 9.1 → 9.7로 알고리즘·작업공간이 바뀐 것으로 추정 (작은 조건 위주, 최대 −12.9%) |
| vit-large | 2/5 | −38 ~ +6 | 같은 계열로 추정 |

  - peak allocated: 46조건에서 1 MiB 미만 차이, resnet-50 1조건 −2 MiB. 라이브러리 작업공간 크기 변화로 추정한다.
  - outside(NVML 피크 − peak reserved): cu124 {312: 319, 314: 63, 316: 2}, cu128 {312: 319, 314: 65} → **변화 없음**
- 한계: torch, CUDA, cuDNN이 한꺼번에 바뀌어서 각각의 효과는 분리할 수 없다.

## 5. E11 — g4dn 환경 비교 A → C → B (드라이버 · PyTorch 분리)

### 5.1 설계

| 환경 | 드라이버 | 이미지 | 라벨 |
|---|---|---|---|
| A 기준 | 595.112.07 (AMI 기본, 데이터센터 runfile 배포본, open 커널 모듈) | `cu124-v1` (torch 2.6.0, E8과 같은 이미지 ID) | `aws-g4dn.xlarge__envcmp_A` |
| C PyTorch | A와 같음 | `pt251-cu124-v1` (`457da255` 위에 pip 레이어 1개: torch 2.5.1, torchvision 0.20.1, torchaudio 2.5.1, triton 3.1.0. cuDNN 9.1.0.70·cuBLAS 12.4.5.8·python·transformers는 동일) | `…_envcmp_C` |
| B 드라이버 | **580.196.07** (R580 데이터센터 최신, runfile, open 모듈, A와 같은 커널) | A와 같은 이미지 ID | `…_envcmp_B` |

- 조건 39개 = 13모델 × (가벼움·중간·무거움). E8 g4dn 결과에서 규칙으로 자동 선택했다 (`tools/select_envcmp_conditions.py`).
  - 가벼움: batch 1 중 peak reserved가 가장 작은 조건
  - 무거움: 할당기 재시도 0회, peak reserved ≤ T4 메모리의 85%, 실행 ≤ 600초 중 가장 큰 조건
  - 중간: 같은 필터에서 두 값의 기하평균에 가장 가까운 조건
  - 피크가 같으면 입력이 작은 조건, 그다음 실행 시간이 짧은 조건을 고른다.
  - 39개 모두 기존 측정 5회의 시간 편차 ≤ 7%, peak 5회 동일
  - 표: `gpu-vm/05-fp16-vram/config/envcmp39_selection.md`, 조건표 `config/conditions_envcmp39.json` (원 조건 객체를 그대로 복사)
- 추가 측정: 래퍼 `src_env/outside_worker.py`가 세 시점에 GPU를 동기화한 뒤 torch allocated/reserved, NVML 프로세스 사용량, outside를 byte로 기록한다(`measure.outside.json`). 워커는 바꾸지 않았다.
  - 세 시점: CUDA 초기화 직후, 모델 로딩 직후, 워밍업 3회 직후(측정 1회차 시작, peak 리셋 전)
  - 래퍼가 측정값을 바꾸지 않는 것은 3090 스모크로 확인했다. 3조건 모두 기존 L256과 바이트 단위로 같았다.
- attention 실제 경로: 환경마다 측정 후 별도 프로세스로 profiler 커널 이름을 수집했다(`runs/<라벨>/_diag/attention_kernels.json`).
- 실행 순서와 보호장치 (`envcmp/run_env.sh`): 다음 경우에는 실행하지 않는다.
  - 커널이 A와 다를 때
  - persistence mode가 A와 다를 때
  - C인데 드라이버가 A와 다를 때, 또는 B인데 드라이버가 580이 아닐 때
  - 조건 시작 시 다른 GPU 프로세스가 있을 때
- 가이드: `gpu-vm/05-fp16-vram/envcmp/GUIDE_ENVCMP_KO.md` (실제로 보낸 판은 `GUIDE_ENVCMP_KO_as_shipped.md`)

### 5.2 결과 (`results/envcmp_g4dn_ACB_REF/envcmp_summary.md`, 조건별 `envcmp_wide.csv`)

| 환경 | ok | 기타 |
|---|---|---|
| A | 39 | — |
| C | 33 | 6 (`compat_or_runtime_error`, opt 3 · vit 3, 모델 로드 단계) |
| B | 39 | — |
| REF (E8 원본 39개) | 39 | — |

| 지표 | REF − A | B − A (드라이버) | C − A (PyTorch) |
|---|---|---|---|
| peak allocated | 39/39 동일 | 39/39 동일 | 33/33 동일 |
| peak reserved | 39/39 동일 | 39/39 동일 | 33/33 동일 |
| NVML 피크 | 39/39 동일 | 36/39 동일, 3개 −2 MiB | 32/33 동일, 1개 −2 MiB |
| CUDA 초기화 직후 outside | (기록 없음) | 39/39 동일 | 39/39 동일 |
| 로딩 직후 outside | (기록 없음) | 39/39 동일 | 33/33 동일 |
| 워밍업 직후 outside | (기록 없음) | 36/39 동일, 3개 −2 MiB (t5-small 2, t5-base 1) | 32/33 동일, 1개 −2 MiB (t5-base b1) |
| 할당기 재시도 | 0 | 0 | 0 |

- −2 MiB는 NVML 보고 단위이고, 원래 130과 132 사이를 오가던 t5 조건에서만 나왔다. 환경 효과로 보지 않는다.
- C의 오류 6개 (확인됨)
  - 원인: transformers 4.52.4의 `check_torch_load_is_safe()`가 torch < 2.6에서 `torch.load`를 거부한다(CVE-2025-32434 대응).
  - 13개 모델 중 `.bin` 가중치는 opt-125m과 vit-large뿐이고, 나머지는 safetensors라 영향이 없다.
  - INUGPU에서 같은 검사를 두 이미지로 실행해 재현했고, 재실행(`envcmp_C__rerun1`, 드라이버 595)에서도 같은 오류가 났다.
  - 원칙("오류는 기록하고 다른 조건으로 대체하지 않음")대로 결과에 남겼다. 보안 검사를 우회한 C′ 실행은 하지 않았다.
- attention 커널 (로그로 확인)
  - T4에서는 A·B 모두 모든 sdpa 모델이 메모리 효율 커널(`fmha_cutlassF_f16_aligned_*_sm75`, PyTorchMemEffAttention)을 썼다.
  - 3090은 FlashAttention이다(11-11 진단). T4(sm75)는 Flash를 지원하지 않는다.
  - 드라이버가 바뀌어도 커널 선택은 같았다.
  - C 진단은 첫 조건(opt) 로드 오류로 실패했다. 드라이버를 바꾼 뒤라 다시 할 수 없어 미실시로 남겼다.
- 확인 재실행
  - B 측정 중 두 번째 `run_env.sh B` 실행의 GPU 확인 컨테이너가 tinyllama b32/256/512 측정과 겹쳤다.
  - 그 조건을 다른 날, 인스턴스 재시작 후(물리 GPU UUID가 b8b78207 → d39f750e로 바뀜) 다시 쟀다(`envcmp_B__rerunconfirm`).
  - allocated 2,974,057,472 B, reserved 3,965,714,432 B, outside 136,314,880 B로 **모두 바이트 단위 동일**했다.

### 5.3 진행 중 생긴 문제와 처리 (보고서의 "실험 과정"용)

| 시점 | 문제 | 원인 | 처리 |
|---|---|---|---|
| C 실행 | opt·vit 6개 오류 | transformers 보안 검사 (위) | 결과로 기록. 3090 스모크에서도 재현 |
| B 설치 1차 | runfile 설치 중단 (`Answer: Abort installation`) | 기존 nvidia 모듈이 로드된 상태에서 `--silent`가 기본 답 "중단"을 선택 | persistence 끄기 → 모듈 내리기 후 재설치 |
| B 설치 2차 | `nvidia` 모듈을 내릴 수 없음 | DLAMI가 올려 둔 `efa_nv_peermem`, `gdrdrv`가 nvidia를 붙잡음. `nvidia_drm`은 없음 | 두 모듈을 먼저 rmmod → 성공. 재부팅 후 580, open 모듈, 같은 커널, persistence Enabled 확인 |
| B 실행 | 두 번째 실행이 `other GPU process present`로 중단 | 첫 실행이 다른 tmux 창에서 진행 중이었음 | 안전장치가 작동한 것. 겹친 1조건은 확인 재실행으로 영향 없음을 확인 |
| 점검 스크립트 | 기록 파일까지 날짜로 읽다가 오류 | 스크립트 버그 | `envcmp/check_envcmp.py` 수정 |

- 설치 기록: `envcmp/records_aws_g4dn/` (1차 실패 로그, 설치 로그, runfile sha256, 재부팅 후 모듈 목록, 환경별 호스트 기록)
  - instance-id와 내부 IP는 가렸다.

## 6. T4 ↔ RTX 3090 (같은 이미지 cu124-v1, GPU 효과)

- 자료: E8(T4)과 3090 L256 (`results/version_gpu_compare/rtx3090_vs_t4_cu124_*`)
- 상태: (ok, ok) 378, (ok, OOM) 6 — T4 용량이 작아서 생긴 OOM
- **peak allocated: 291/378 바이트 단위 동일, 나머지도 ±2.5 MiB 이내**
  - GPU가 달라도 요청 총량은 거의 같다는 뜻이다.
  - 2.5 MiB 이하 차이는 라이브러리 작업공간이나 attention 커널(메모리 효율 vs Flash) 차이로 추정한다.
- peak reserved: 221/378 동일

| 차이 유형 | 대표 예 (3090 → T4, MiB) | 해석 |
|---|---|---|
| T4 용량에서 재시도가 생김 | gpt2 b128/128/512 21,026 → 9,752 (재시도 0 → 1), t5-base b128/256/512 17,848 → 5,488, tinyllama b128/128/512 20,730 → 9,544 | 용량에 닿으면 reserved가 경로에 따라 정해진다 (`docs/05` E7과 같은 현상) |
| 양쪽 모두 재시도 구간 | opt b128/128/512 4,972 → 13,510 (둘 다 재시도 1) | 재시도 구간 값은 예측 대상이 아님 |
| 재시도 없이 T4가 더 큼 | qwen b128 in512 3개 +928, in128/in256 +152 / +184 (allocated 동일) | **원인 미확인**. 세그먼트 배치 또는 커널 차이 추정 |
| 소폭 | ±2 ~ ±100 MiB 다수 | 주소 순서·작업공간 차이 추정 |

- 결론
  - 예측의 전제인 "요청 목록(allocated)은 GPU와 거의 무관하다"는 T4와 3090 사이에서 성립했다.
  - reserved는 GPU 용량을 넣어 재시도를 재현해야 하고, 재시도가 없어도 GPU별로 수백 MiB 차이가 날 수 있다(qwen).
  - 다음 실험으로 T4 GPU trace 수집을 제안한다 (`docs/00` §6).

## 7. outside 정리와 반영

| GPU (이미지) | CUDA 초기화 직후 → 로딩 직후 → 워밍업 직후 (MiB) | NVML 피크 − peak reserved (MiB, 조건 수) | 출처 |
|---|---|---|---|
| T4 (cu124) | 102 → 118(opt만 102) → 130 / 132 | {130: 265, 132: 113} (L256 378) | E8, E11 |
| T4 (cu124, 드라이버 580) | A와 동일 (−2 MiB 3건) | — | E11 B |
| T4 (pt251-cu124) | A와 동일 (−2 MiB 1건) | — | E11 C |
| RTX 3090 (cu124) | 254 → 296 → 312 | {312: 319, 314: 63, 316: 2} | 3090 L256, 스모크, 11-03 ~ 11-10 outside 연구 |
| RTX 3090 (cu128) | — | {312: 319, 314: 65} | E10 |
| L4 / A10G / L40S | — | 232 / 308 / 510 (이전 연구, `docs/02`) | 11-03 ~ 11-10 |

- 해석
  - 측정 중에는 outside가 더 늘지 않는다. 워밍업 직후 값이 곧 측정 중 NVML 피크와 peak reserved의 차이다.
  - 모델·배치·길이와 무관하다.
  - `CUDA_MODULE_LOADING=EAGER`만 예외다(+728 ~ 754 MiB, 이전 연구). 측정은 모두 LAZY다.
- 반영
  1. 예측 VRAM = 예측 reserved + **GPU별 outside 상수**(관측 최대값: T4 132, 3090 316, L4 232, A10G 308, L40S 510 MiB). g7(Blackwell)은 결과를 받은 뒤 추가한다.
  2. 드라이버 버전은 예측 입력에서 뺀다 (T4 R595 ↔ R580에서 효과 0).
  3. PyTorch 버전은 메모리 값이 아니라 호환성 조건으로 다룬다 (torch < 2.6 + transformers 4.52.4 + `.bin` 모델 = 로드 불가).
  4. CNN 보정은 GPU 종류 × cuDNN 버전 단위로 둔다 (3090 cu128에서 resnet 수십 MiB 차이).
  5. 후보 GPU 판정은 R_inf / R_min과 용량 C로 한다(`docs/05` §4). T4 결과가 이를 지지한다: 용량에 닿은 조건에서 reserved가 크게 줄었다.

## 8. 한계

- 드라이버 비교는 T4 한 종류, R595와 R580 두 브랜치뿐이다.
- 조건마다 독립 실행은 1회이고, 확인 재실행은 2건이다.
- NVML 분해능이 2 MiB라서 그보다 작은 outside 차이는 볼 수 없다.
- E10은 torch·CUDA·cuDNN이 동시에 바뀌어 원인을 나눌 수 없다.
- 미확인: tinyllama +204 MiB (E10), qwen +928 MiB (T4), opt만 로딩 직후 outside가 늘지 않는 이유
- g7 결과는 아직 받지 못했다.

## 9. 파일 위치 (레포 `gpu-vm/05-fp16-vram/` 기준)

| 내용 | 경로 |
|---|---|
| T4 L256 원본 / 조건별 CSV / 로그 | `runs/aws-g4dn.xlarge__l256/`, `results/gpu_g4dn.xlarge_T4_main395_L256_per_condition.csv`, `logs/aws-g4dn.xlarge__l256/` |
| 3090 cu128 원본 / CSV / 로그 | `runs/rtx3090__main395_l256__cu128/`, `results/gpu_rtx3090_cu128_main395_L256_per_condition.csv`, `logs/cu128_rtx3090.txt`, `logs/main395_rtx3090__main395_l256__cu128_summary.md` |
| 버전·GPU 비교표 | `results/version_gpu_compare/` (도구 `tools/compare_gpu_runs.py`) |
| 환경 비교 원본 | `runs/aws-g4dn.xlarge__envcmp_{A,C,B}/`, `…_C__rerun1/`, `…_B__rerunconfirm/` |
| 환경 비교 집계 | `results/envcmp_g4dn_ACB_REF/` (INUGPU 재집계, REF 포함), `results/envcmp_aws_g4dn/` (AWS에서 만든 집계) |
| 호스트·드라이버 기록 | `envcmp/records_aws_g4dn/` |
| 환경 비교 도구 | `src_env/outside_worker.py`, `tools/run_conditions_env.py`, `tools/compare_envs.py`, `tools/select_envcmp_conditions.py`, `envcmp/*.sh`, `envcmp/check_envcmp.py`, `envcmp/verify_images.py`, `envcmp/make_bundle.sh` |
| 3090 스모크 (래퍼·C 이미지 검증) | `runs/rtx3090__envcmp_smoke_{A,C}/`, `results/envcmp_smoke_3090/`, `run_envcmp_smoke_3090.sh` |
| 이미지 정의 | `aws/Dockerfile.cu128`, `aws/IMAGE_INFO_cu128.json`, `aws/Dockerfile.pt251`, `aws/IMAGES_INFO_envcmp.json` |
| 번들 (레포 밖) | `EXCLUDED_LARGE_FILES.md` 참고 |

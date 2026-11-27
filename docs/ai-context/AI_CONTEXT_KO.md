# CloudGPUFit 실험 — 로컬 AI 분석용 전체 맥락 (2026-11-15 작성, 11-18 갱신: §4.6 AWS, §4.8~4.10 추가)

> **보고서를 쓰는 AI는 먼저 `docs/00_REPORT_MAP_KO.md`(INUGPU·INUOS·AWS 통합 지도)를 읽는다.** 이 문서는 INUGPU 측정 방식의 상세 정의(§5)와 실험별 기록이다.

> **2026-11-20 INUOS 추가**: 보고서 작성용 통합 문서는 `REPORT_CONTEXT_KO.md`다(INUGPU·INUOS·AWS 전체, 이후 실험 포함). 이 문서의 §4.6 "AWS 결과 미확인"은 지난 내용이다. AWS GPU 5종과 CPU 실제 FP16 결과는 `aws/`에 있다.

이 문서는 이 브랜치를 읽는 AI가 **지금 무슨 실험이 어떤 상태로 돌아가고 있는지** 판단할 수 있도록 쓴 것입니다.
먼저 이 문서를 끝까지 읽고, 근거가 필요할 때 아래에 적힌 파일을 여세요.

- 표기 규칙
  - "확인됨"은 코드·실행 기록·결과 파일로 확인한 사실입니다.
  - "미확인"은 아직 검증하지 않은 내용입니다.
  - "INUOS 쪽 정보(11-12 시점)"는 INUOS가 2026-11-12에 작성한 매니페스트에서 옮긴 내용입니다. 이후 바뀌었을 수 있습니다.

---

## 1. 프로젝트 한 줄 요약

- CloudGPUFit은 GPU를 빌리기 전에 **CPU만으로 추론 워크로드의 GPU VRAM 사용량을 예측**하고, 그 결과로 AWS GPU 인스턴스를 추천하는 졸업작품입니다.
- 최종 VRAM 추정 = **xMem 예측 reserved** + **outside**
  - xMem: PyTorch CUDA caching allocator를 CPU에서 흉내 내는 시뮬레이터
  - outside: allocator 밖 메모리. CUDA context 등이 여기에 해당합니다.

## 2. VM 두 대의 역할

| VM | 역할 | 이 브랜치의 디렉터리 |
|---|---|---|
| **INUGPU** | RTX 3090 24GB VM. **GPU 실측(정답값)**을 냅니다. 같은 패키지로 AWS GPU 실측도 실행합니다. | `measurement/local-gpu-vm/` |
| **INUOS** | CPU 전용 VM. 같은 조건을 CPU에서 실행해 allocator 요청을 기록하고 xMem으로 **GPU VRAM을 예측**합니다. | `prediction/local-cpu-vm/` |

- 두 VM의 Claude 에이전트는 서로 직접 대화하지 못합니다.
  - 파일 전달: INUGPU → INUOS 방향 scp만 됩니다. 반대 방향은 키가 등록되지 않아 막혀 있고, 사용자가 의도적으로 열지 않았습니다.
  - 지금까지 정보는 handoff 문서와 매니페스트를 서로 복사해서 주고받았습니다.
- 분석 AI가 할 일: 두 디렉터리를 대조해서 **GPU 실측과 CPU 예측이 같은 조건, 같은 절차, 같은 기준값으로 비교되고 있는지**, 그리고 어디서 값이 어긋나는지 판단하는 것입니다.

## 3. 브랜치 구조

```
AI_CONTEXT_KO.md                  ← 이 문서 (먼저 읽기)
EXCLUDED_LARGE_FILES.md           ← 50MB 초과로 뺀 파일 8개 목록·sha256 (VM 삭제로 원본 없음)
docs/                             ← 실험별 상세 문서 (설계·실행 명령·코드 파일별 설명·결과 수치·미확인 사항)
  01_early_and_bench.md           ← 초기 벤치(03-31), 14B OOM(09-14), cloudgpufit-bench v1~v3(10월)
  02_outside.md                   ← outside 분석 (11-03~11-10)
  03_xmem_probes_reserved_diag.md ← xMem 검증 프로브(08월), FP16 재생, reserved 오차 trace 수집(11-10)
  04_fp16_vram.md                 ← FP16 VRAM 실측: 파일럿·본실험 395·AWS 패키지 (11-11~)
  05_reserved_analysis_KO.md      ← reserved 구성 분해·반복 변동·GPU R_min 실측 (11-16, 보고서용 해설 포함)
  06_env_gpu_effects_KO.md        ← 드라이버·PyTorch·CUDA/cuDNN·GPU 종류 영향, outside, T4 L256, g7/cu128 (11-16~18)
  07_code_env_lineage_KO.md       ← 코드·이미지·번들·조건표 계보 (무엇이 언제 왜 바뀌었나)
  00_REPORT_MAP_KO.md             ← 보고서용 통합 지도 (먼저 읽기)
measurement/local-gpu-vm/
  00-early-scripts/               ← VM 홈의 초기 단독 스크립트·결과 (ResNet/YOLO 벤치, 14B OOM 테스트 등)
  01-cloudgpufit-bench/           ← 342조건 벤치 하네스 v1~v3, results/ (원본 JSON 포함)
  02-outside-analysis/            ← outside 분석 전체 (outside-lab 코드, results, AWS 번들·결과)
  03-xmem-probes/                 ← xMem 단독 검증 스크립트, cgf-fp16-replay-20261111/, xMem-local-changes/(로컬 수정 패치)
  04-reserved-diag/               ← reserved 오차 원인 분리 GPU trace 수집 (요청서·수집기·반환물, 대형 snapshot 제외)
  05-fp16-vram/                   ← 현재 주 실험 (FP16 VRAM 실측)
    src/vram_worker.py            ← GPU 측정 worker (정답 코드, sha256 d06f0f15…)
    src/run_conditions.py         ← runner: 조건마다 새 Docker 컨테이너 1회 (sha256 77e3f4a9…)
    config/                       ← 조건표(680, main395, main395_l256)·모델 revision·생성기
    tools/                        ← 요약·파일럿 판정·CSV 추출·xMem 재생·진단
    aws/                          ← AWS 실행 스크립트 (RUN_ON_AWS.sh 등)
    runs/                         ← 실행 원본 (스모크, 파일럿 v1/v2, pathcheck, main395 L1024·L256, trace_models·trace_slack, repeat_r01~15, capsearch)
    results/                      ← 파일럿 보고서, xMem 재생표, Qwen 진단, L1024·L256 조건별 CSV, reserved_anatomy/, 반복 변동, R_min
    src_cap/, tools/*cap*         ← GPU 메모리 상한 실행 래퍼 (R_min 실측용, 측정 코드는 그대로)
    handoff_inuos/                ← INUOS에 보낸 reserved 예측 가이드·보충·R_min 검증 문서
    src_env/, envcmp/             ← 환경 비교 래퍼(세 시점 outside)·AWS 실행 스크립트·가이드·AWS 호스트 기록(records_aws_g4dn/)
    runs/aws-g4dn.xlarge__*       ← AWS T4 결과 (L256 395, envcmp A/C/B). 내부 IP·instance-id는 가림
    handoff_codex/                ← 11-12 GPU 매니페스트(gpu_manifest.json)
prediction/local-cpu-vm/                            ← INUOS 코드와 문서가 들어갈 자리 (§9)
```

- 제외한 것: 모델 캐시(hf-cache), Docker 이미지 tar, xMem 원본 저장소(.git), 50MB 초과 파일 8개(`EXCLUDED_LARGE_FILES.md`), 진행 중인 L256 결과
- 코드와 결과 원본(JSON·CSV·로그·소형 trace)은 대부분 포함됨. 실험별 해설은 `docs/`를 먼저 보라

---

## 4. 시간순 실험 이력

### 4.0 그 이전 실험 (2026-03 ~ 10) — 완료, 상세는 `docs/01_early_and_bench.md`, `docs/03_xmem_probes_reserved_diag.md`

- 03-31: ResNet-50/YOLOv5s FP32 벤치마크 (`measurement/local-gpu-vm/00-early-scripts/`). 당시 목적은 CloudSim 입력값
- 08-16~23: xMem 단독 검증 프로브·GT 스크립트 (`measurement/local-gpu-vm/03-xmem-probes/`). 결과는 대부분 stdout만 남음
- 09-14: Qwen2.5-14B FP16 24GB 적재 테스트: strict는 LOAD_OOM, offload는 0.57 tok/s
- 10-06~27: cloudgpufit-bench (`measurement/local-gpu-vm/01-cloudgpufit-bench/`)
  - 모델 9종(gpt2×3, t5×3, mobilenet/resnet-50/vit-large) × 342조건 × 5모드(performance/memory/first_logits/tokens/components), eager attention, fp16/fp32/bf16
  - v1 17.8시간 → v3(모델 공유+hybrid 격리+반복 축소) 3.7시간. memory peak는 342 중 326 완전 일치, 지연 p90은 중앙값 1~2% 차이
  - 이 폴더에는 RTX 3090 결과만 있음. AWS(T4/A10G/L4/L40S) 342조건 결과는 이 VM에 없음(미확인)

### 4.1 outside 분석 (INUGPU 3090 + AWS, 2026-11-03 ~ 11-10) — 완료

- 위치: `measurement/local-gpu-vm/02-outside-analysis/`
  - 종합 문서: `NOTION_SUMMARY_20261108.md`
  - 실험별 보고서: `reports/*.REPORT_KO.md`
- outside 정의: 같은 조회 시점의 `NVML 프로세스 메모리 − torch.cuda.memory_reserved`
- 확인된 결론
  - 3090, driver 570, `CUDA_MODULE_LOADING=LAZY`에서 첫 추론 이후 outside는 **312MiB**(비전 314)로 고정입니다.
    - 모델 크기, 배치, 길이, dtype, attention, torch/CUDA 버전이 바뀌어도 차이는 0~2MiB였습니다.
    - 로컬 190개 모델·1375행 사후평가에서 최대 오차는 2MiB였습니다.
  - 단계별로는 254MiB(초기화) → 296MiB(로딩) → 312MiB(첫 forward)로 늘어납니다. 그래서 **첫 추론 이후 값**을 써야 합니다.
  - `CUDA_MODULE_LOADING=EAGER`이면 +728~754MiB가 됩니다(1040MiB). 고정이 아닌 유일한 요인입니다.
  - 드라이버 예약분(`디바이스 used − 프로세스 NVML`)은 448.9MiB로 일정합니다. outside와는 별개입니다.
  - AWS L4(g6.xlarge, driver 595): outside 232MiB, 소프트웨어 S0/S4 차이 0
  - 기존 AWS 자료: T4 130, A10G 308, L4 232, L40S 510MiB
- 미완료
  - g5(A10G) 결과는 회수만 됐습니다(`aws-results/g5-xlarge-01/*.tar.gz`, 브랜치에는 없음). 분석했는지는 미확인입니다.
  - GPU 특성으로 outside를 예측하는 가설(H5, H6)은 검증되지 않았습니다.

### 4.2 reserved 오차 원인 분리 trace 수집 (INUGPU, 2026-11-10) — 완료

- 위치: `measurement/local-gpu-vm/04-reserved-diag/`
  - 요청서: `handoff/GPU_COLLECTION_REQUEST.md` (INUOS 작성)
- 배경: CPU의 xMem 예측 reserved가 GPU 실측보다 크게 작았습니다. 예: L4 gpt2 FP16 b4/768/128에서 CPU 928MiB, GPU 1678MiB.
- 원인 후보
  - (a) 실행 이력(초기 캐시)
  - (b) xMem과 PyTorch allocator의 정책 차이
  - (c) GPU 전용 요청(cuBLAS workspace 등)
- 수집 내용: 3 case(gpt2 fp16 두 가지, gpt2-medium fp32) × 4 run(R0 AWS 재현, R1 통제, R2 cold single, D1 empty_cache 진단) = 12 프로세스
  - 각 프로세스에서 allocator 이벤트 전체를 기록했습니다.
- 결과 tar는 INUGPU에 있습니다. 브랜치에는 수집 코드와 요청서만 넣었습니다.

### 4.3 FP16 VRAM 파일럿 26개 (INUGPU 3090, 2026-11-11) — 완료

- 위치: `measurement/local-gpu-vm/05-fp16-vram/results/pilot/`
  - 보고서: `REPORT_rtx3090_pilot_eager_vs_default_KO.md`
- 비교: 26개 조건을 v1(eager 고정)과 v2(transformers 기본 attention)로 측정했습니다.
- 확인된 결론
  - 기본 attention은 OPT, TinyLlama, Qwen, GPT-2, ViT에서 sdpa이고, 실제 커널은 FlashAttention만 쓰였습니다(L×L 행렬 없음).
    - T5, ResNet, MobileNet은 기본값이 eager입니다.
  - sdpa 14조건 중 10개에서 peak reserved가 줄었습니다. 최대는 TinyLlama b16/512/512의 −1060MiB입니다.
  - Qwen2.5 FP16 eager는 출력이 NaN입니다. 원인은 QKᵀ가 FP16 최대값을 넘는 것이고, sdpa에서는 정상입니다.
  - 52개 measure 모두에서 **1회차 시작 reserved = 1회차 peak**였고, 5회 peak가 모두 같았습니다.
    - 즉 reserved peak는 warmup 3회가 만든 캐시 상태에서 이미 정해집니다.
  - **xMem 재생**: GPU trace 12개의 요청을 그대로 xMem에 넣으면 모든 경계에서 reserved, allocated, segment 수가 GPU와 완전히 같았습니다.
    - 결론: **allocator 정책은 문제가 아닙니다. CPU와 GPU가 다르면 요청 목록(입력 trace)의 차이를 먼저 봐야 합니다.**
- 이후 **v2(기본 attention)를 기준 방식으로 확정**했습니다.

### 4.4 본실험 395개 — 1024 포함 조건표 (INUGPU 3090, 2026-11-12) — 완료

- 위치: `measurement/local-gpu-vm/05-fp16-vram/results/main395_L1024/`
- 조건표 `config/conditions_main395.json` (sha256 `2634bda7…`)
  - 텍스트 6종(OPT, TinyLlama, Qwen, T5×3)은 입력·생성 길이 {128,512,1024}²
- 실행: 2026-11-12 09:51 ~ 23:13 (measure 11.3시간, validate 2.1시간)
- 결과
  - measure: 381 ok / 14 gpu_oom. OOM은 모두 `warmup_0`에서 났습니다.
    - gpt2-xl b128: 3개
    - t5-3b b128: 8개
    - t5-3b b32: 3개
  - validate: 371 ok / 24 gpu_oom
    - 그중 10개는 validate에서만 OOM이었고 measure 값은 유효합니다. opt-125m, gpt2, gpt2-medium의 b128 조건입니다.
- 조건별 상세: `gpu_rtx3090_main395_L1024_per_condition.csv` (byte 단위, 입력 sha256 전체, OOM 시점 상태, validate 결과)

### 4.5 본실험 395개 — 1024 → 256 조건표 (INUGPU 3090, 2026-11-15~16) — 완료

- 사용자 결정: 1024 조건이 너무 오래 걸려서 **1024를 256으로 바꿨습니다.**
- 조건표 `config/conditions_main395_l256.json` (sha256 `d4a9b09caff3064c00c04358bf2e462996bced23de93da98652bd1aba04b7289`)
- 시작: 2026-11-15 17:00 KST
  - label: `rtx3090__main395_l256`
  - 18:05 기준 83/395 measure, 모두 ok
- 예상 소요: measure 약 6.2시간 + validate 약 1.4시간
- 결과(INUGPU): `runs/rtx3090__main395_l256/`. 끝나면 `summary.csv`가 생기고, `tools/export_for_inuos.py`로 조건별 CSV를 만듭니다.
  - **완료(11-16 00:45)**: measure 384 ok / 11 gpu_oom (gpt2-xl b128 3개, t5-3b b128 8개, 모두 warmup_0), validate 377 ok / 18 gpu_oom
  - 조건별 결과: `measurement/local-gpu-vm/05-fp16-vram/results/gpu_rtx3090_main395_L256_per_condition.csv`, 원본 `runs/rtx3090__main395_l256/`

### 4.6 AWS 실측 (사용자 직접 실행)

- AWS 패키지: 단일 tar `cgf-fp16-aws-full-20261115.tar`
  - 이미지, 코드, 조건표가 들어 있고, 모델 캐시는 실행 시 받습니다.
  - sha256 `77f7a941e62cf7b159d94218a373f6b33ff23ad08b0b8c4ce19500430cad7087`
- 실행 순서: 체크섬 → 인스턴스 타입 확인 → 이미지 검증 → 모델 캐시 해시 검증 → 파일럿 26개 PASS → 본실험 395개
  - label: `aws-<타입>__l256`
  - 1024 조건표로 하던 이전 실행은 label `aws-<타입>`입니다.
- 사용자가 g6(L4)에서 이전 패키지로 실행한 적이 있습니다. 그 결과는 아직 어느 VM에서도 확인되지 않았습니다(미확인).
- **11-18 갱신**
  - g4dn.xlarge(T4) L256 395: 완료(378 ok / 17 OOM) → `runs/aws-g4dn.xlarge__l256/`, `results/gpu_g4dn.xlarge_T4_main395_L256_per_condition.csv`
  - g7.2xlarge(Blackwell): cu124 이미지 미지원으로 실패 → cu128 번들로 재측정 완료, **결과 미수신**
  - g4dn 환경 비교 A/C/B: 완료 → §4.10

---

### 4.7 reserved 분석 (INUGPU 3090, 2026-11-16) — 완료

- 상세·보고서용 해설: `docs/05_reserved_analysis_KO.md` (노션 "reserved 분석" 페이지와 E1~E7 번호 동일)
- 구성 분해 (E1~E5): reserved = 가중치 + 최대 allocated + 라이브러리 작업공간 + slack
  - GPU 요청 목록을 xMem에 재생하면 큰 조건에서 최대 +896 MiB 틀렸는데, 원인은 세그먼트 주소 순서였다.
  - "새 세그먼트는 더 낮은 주소" 규칙으로 재시도 없는 20개 중 19개가 0~8 MiB 안에서 맞았다.
- **반복 변동 (E6)**: 같은 조건·같은 환경으로 15회씩 측정했다.
  - opt b16은 968/1,012 MiB, b32는 2,326/2,330 MiB로 두 값 중 하나로 갈렸다. 요청 수는 매번 같았다.
  - 의미: 동일 조건에서도 실측값이 최대 4.5% 변할 수 있다. 이 폭보다 작은 예측 오차는 구분할 수 없다.
- **GPU R_min 실측 (E7)**: 메모리 상한을 걸어 OOM 경계를 찾았다(7개 조건, 55회).
  - xMem R_min은 LLM·T5에서 0 ~ +3.5%로 맞았고 과소 예측은 없었다. CNN은 +500 MiB(cuDNN이 알고리즘을 바꿈)였다.
  - 재시도 구간의 reserved는 용량 근처가 아니고 경로에 따라 달라서 예측하지 않는다. 판정 규칙을 정정했다.

### 4.8 g7(Blackwell)과 cu128 이미지 (2026-11-17)

- cu124-v1(torch 2.6.0+cu124)은 sm_50~sm_90만 지원 → g7(sm_120) pilot 26개 전부 `compat_or_runtime_error`
- `cloudgpufit-bench:cu128-v1` 제작(torch 2.7.1, CUDA 12.8, cuDNN 9.7.1, pip 패키지는 cu124-v1과 같은 버전). AWS 스크립트에 이미지 선택 추가(라벨 `__cu128`)
- 상세: `docs/06` §3, `docs/07` §5~6

### 4.9 RTX 3090 cu124 ↔ cu128 버전 비교 (2026-11-17) — 완료

- 라벨 `rtx3090__main395_l256__cu128`, 395 measure + validate. 상태 변화 0, reserved 바이트 단위 동일 355/384
- 차이는 CNN·ViT(cuDNN 추정) 수십 MiB, tinyllama 1조건 +204 MiB(미확인). outside 분포 동일
- 결과: `results/gpu_rtx3090_cu128_main395_L256_per_condition.csv`, `results/version_gpu_compare/`. 상세 `docs/06` §4

### 4.10 g4dn 환경 비교 A → C → B (AWS, 2026-11-17~18) — 완료

- A: 드라이버 595.112.07 + torch 2.6.0 / C: torch 2.5.1(드라이버 그대로) / B: 드라이버 580.196.07(이미지 그대로). 39조건(13모델 × 3)
- 드라이버·PyTorch 모두 allocated·reserved 바이트 단위 동일, outside 차이 ≤ 2 MiB(NVML 분해능). C에서 opt·vit 6개는 transformers의 torch<2.6 `.bin` 로딩 거부로 실패(기록)
- A는 11-16 g4dn 실측과 39/39 바이트 단위 동일. T4 outside는 CUDA 초기화 102 → 로딩 118 → 워밍업 130/132 MiB
- 상세: `docs/06` §5, 원본 `runs/aws-g4dn.xlarge__envcmp_*`, 집계 `results/envcmp_g4dn_ACB_REF/`

## 5. GPU 측정 방식 (v2, 현재 기준) — 정확한 정의

- 정답 코드는 `measurement/local-gpu-vm/05-fp16-vram/src/vram_worker.py`입니다. 아래 줄번호는 이 파일 기준입니다.
- 환경
  - Docker 이미지 `cloudgpufit-bench:cu124-v1` (ID `sha256:457da255…`)
  - python 3.11.11, torch 2.6.0+cu124, cuDNN 90100, transformers 4.52.4
  - 시드 20261006, TF32 off, `cudnn.benchmark=False`, `CUDA_MODULE_LOADING=LAZY`, `PYTORCH_CUDA_ALLOC_CONF` 미설정
- 조건 1개 = **새 컨테이너 1개**: `run_conditions.py:130`. 이전 조건의 캐시를 물려받지 않습니다.
- 절차
  1. FP16으로 모델 로드 (`:242`)
     - `attn_implementation` 인자를 생략해 기본값을 씁니다.
     - `low_cpu_mem_usage=True, device_map={"":"cuda:0"}` 후 `.to(cuda, fp16).eval()`
  2. 입력 생성 (`:271`, `:277`)
     - 토크나이저는 쓰지 않습니다. 텍스트는 `randint(3, vocab, (B,L))`이고, 특수 토큰은 3으로 바꿉니다.
     - 이미지는 `randn(B,3,R,R)`을 fp16으로 바꿉니다.
     - 호스트 CPU에 두고 매 회차 `.to(cuda)`로 복사합니다.
  3. 생성 설정 (`:292`)
     - greedy, beam 1, `use_cache=True`, `min_new_tokens=max_new_tokens=N`
     - hub의 generation_config는 쓰지 않습니다.
  4. **warmup 3회 → measure 5회** (`:323`, `:343`)
     - 캐시는 유지하고 `empty_cache`는 부르지 않습니다.
     - 각 measure 직전에 `reset_peak_memory_stats()`만 호출합니다 (`:348`).
  5. 기록: 회차별 start / peak / end_with_output / released, NVML(10ms 간격)
- **비교 기준값: `summary.first_measure_peak_reserved` (byte)**
  - 보조 기준: `first_measure_peak_allocated`, `after_load reserved`, `first_measure_start_reserved`
- validate는 별도 프로세스에서 1회 실행합니다.
  - forward hook으로 모든 스텝의 logits에 isfinite를 검사합니다.
  - validate의 메모리 값은 측정값으로 쓰지 않습니다.
- 실패 처리
  - 결과를 `status`/`failed_stage`로 기록하고 자동 재시도하지 않습니다.
  - **배치나 dtype을 바꿔 다시 돌리지 않습니다.**
  - 분류: `gpu_oom`, `invalid_input`, `generation_length_mismatch`, `nonfinite_output` 등
- OOM 기준 용량: 3090의 `cuda_total_memory` = 25,306,529,792 B

## 6. 조건표 (현재: `conditions_main395_l256.json`, 395개)

| 그룹 | 모델 | 축 | 후보 | 수 |
|---|---|---|---|---|
| 디코더 LLM | OPT-125M, TinyLlama-1.1B-Chat, Qwen2.5-1.5B-Instruct | 입력×생성 | {128,256,512}² | 135 |
| GPT-2 | GPT-2, Medium, XL | 입력×생성 | {128,512}² | 60 |
| 인코더-디코더 | T5-small, base, 3B | 입력×생성 | {128,256,512}² | 135 |
| CNN | ResNet-18/50, MobileNet V2 | 해상도 | 128/224/256/512 | 60 |
| ViT | ViT Large | 해상도 | 224 | 5 |

- 배치: 1/8/16/32/128
- 모델 revision은 `config/models.json`에 고정되어 있습니다.
- 이전 1024 조건표와 겹치는 id는 245개, 새로 생긴 id(256 포함)는 150개입니다.
  - 겹치는 245개는 측정 설정이 같으므로 L1024 실행과 L256 실행의 값이 같아야 합니다. **재현성 검증에 쓸 수 있습니다.**

---

## 7. INUOS(CPU 예측) 쪽 — 알려진 내용 (INUOS 쪽 정보, 11-12 시점)

- 레포: INUOS `/home/inuos/cgf-fp16-680-20261111`
  - 브랜치 `exp/fp16-680-cpu-20261111`, 커밋 `42d96a9`
- 핵심 코드
  - `src/cgf680/cpu_worker2.py` (CPU worker v2)
  - `tools/run395.py` (실행기)
  - `docs/run_pilot395.sh`, `docs/run_main395.sh`
- 예측기 의존 코드
  - `/home/inuos/CloudGPUFit-cpu-singlepass-20261108/` (`inference_trace_runtime.py` 등)
  - `/home/inuos/xmem-work/xMem`
  - `/home/inuos/cgf-reserved-diag-20261110/src`
- 조건 목록: `manifest/cases395.jsonl`, `manifest/conditions_main395.json`
  - **1024가 들어간 이전 조건표 기준입니다.** 256 조건표로 바꿨는지는 미확인입니다.
- 실행 방식
  - 컨테이너 `cloudgpufit-cpu-validation:20261108`, 20스레드, 메모리 제한 42g, network none
  - phase A/B/C로 나눠 실행합니다.
  - 정밀도: `fp32_execution_fp16_request_sizes`. CPU에서는 FP32로 실행하고, allocator 요청 크기만 FP16 기준으로 환산합니다.
  - 기준값 매핑: GPU `summary.first_measure_*`를 "fourth replay, k=3, measure_0 after warmup_0/1/2"로 대응시킵니다. xMem 재생으로 GPU 절차를 흉내 내는 방식입니다.
  - xMem 용량은 1 TiB라서 GPU OOM은 재현되지 않습니다.
- 기타 실행 이력: `runs/emu-validate`(fp16 요청 크기 환산 검증), `runs/fp32-calib`(fp32 보정)
- 결과 파일
  - `results/main395/status_summary.json`
  - `results/main395/gpu_trace_request_diff.jsonl`: GPU trace와 CPU 요청의 차이
- INUOS가 남긴 미해결 질문과 INUGPU의 답: `measurement/local-gpu-vm/05-fp16-vram/INUOS_GPU_EXPERIMENT_SETUP_KO.md` §8
  - cuBLAS workspace(8,519,680 B)와 cuBLASLt(1 MiB) 요청이 6개 trace 밖에서 어떻게 나오는지는 미확인입니다.
  - T5에서 GPU 8KB vs CPU 16KB, GPU 1MiB vs CPU 2MiB처럼 정확히 2배 차이 나는 요청이 있습니다. 원인은 미확인입니다.
    - 후보: FP32 실행 텐서에 FP16 환산이 빠진 것, 또는 T5LayerNorm의 float32 연산

## 8. 지금 알려진 불일치 후보 (분석 AI가 검증할 것)

| # | 항목 | GPU | CPU | 상태 |
|---|---|---|---|---|
| 1 | 조건표 | L256 (d4a9b09c…) | L1024 기준일 가능성 | 미확인 |
| 2 | 실행 정밀도 | 실제 FP16 | FP32 실행 + FP16 요청 크기 환산 | 구조적 차이. 환산이 빠진 텐서가 있는지 확인 |
| 3 | attention 경로 | sdpa = FlashAttention (L×L 없음) | CPU sdpa 경로(flash CPU인지 math인지) 미기록 | math면 과대 예측 |
| 4 | GPU 전용 요청 | cuBLAS/cuBLASLt/cuDNN workspace | 없음 → 주입 방식 사용 | 빈도와 수명 미확인 |
| 5 | 측정 순서 | 로드 → warmup 3 → measure 5, 캐시 유지 | 재생 k=3 후 measure_0 | 경계는 대응됨, 값 일치는 미확인 |
| 6 | OOM | 25.3GB 초과 시 gpu_oom | xMem 1TiB | 별도 판정 필요 |
| 7 | 단위 | byte (CSV), MiB (`summary.csv`) | — | 혼동 주의 |

- 분석 우선순위 제안
  1. 같은 condition_id끼리 GPU `first_measure_peak_reserved`와 CPU 예측을 대조합니다. 모델별·그룹별 오차 패턴을 봅니다.
  2. 오차가 큰 그룹은 요청 목록 차이(`gpu_trace_request_diff`)를 봅니다. xMem 정책은 재생 실험에서 검증이 끝났습니다.
  3. L1024와 L256에 겹치는 245개의 재현성을 확인합니다.

## 9. `prediction/local-cpu-vm/` 디렉터리에 넣을 것 (INUOS에서 직접 올려야 함)

- 이 문서와 `measurement/local-gpu-vm/`는 INUGPU에서 만들었습니다. INUOS 코드는 INUOS에서 올려야 정확합니다.
- 권장 구성
  - `prediction/local-cpu-vm/cgf-fp16-680/`: `src/`, `tools/`, `docs/`, `manifest/`, `results/main395/`(요약 파일만), `README.md`
  - `prediction/local-cpu-vm/cpu-singlepass/`: `CloudGPUFit-cpu-singlepass-20261108`의 예측기 코드
  - `prediction/local-cpu-vm/handoff_codex/cpu_manifest.json`: 11-12 매니페스트. 4.3MB이므로 원하면 포함합니다.
  - `prediction/local-cpu-vm/INUOS_CONTEXT_KO.md`: INUOS 에이전트가 작성. §7을 최신 상태로 보완하는 문서입니다.
- 넣지 말 것: 모델 캐시, `npz`/`pickle` trace, SSH 키, 토큰

## 10. 운영 제약 (실험 정책, 사용자 지시)

- 생산 예측 로직과 과거 결과는 보존합니다.
- 실패 조건을 몰래 낮은 배치나 다른 dtype으로 바꾸지 않습니다.
- 모델 ID나 후보를 자동으로 바꾸지 않습니다. 제약이 발견되면 원인과 해당 케이스를 기록합니다.
- 공유 blob이나 활성 컨테이너의 파일을 직접 삭제하지 않습니다.
- AWS를 실행할 때는 배정된 인스턴스와 실행 범위를 확인합니다.
- 키, 토큰, 개인 정보는 기록하지 않습니다.

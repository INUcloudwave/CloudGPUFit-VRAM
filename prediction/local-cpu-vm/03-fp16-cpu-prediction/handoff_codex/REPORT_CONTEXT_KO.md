# CloudGPUFit 실험 전체 맥락 — 보고서 작성용 (INUGPU · INUOS · AWS 통합, 최종 2026-11-23)

이 문서는 실험 과정을 직접 보지 못한 사람이나 AI가 **보고서를 쓸 수 있도록** 만든 안내서다.
각 실험을 왜 했고, 어떻게 했고, 결과가 어땠고, 그 결과로 무엇을 바꿨는지를 한곳에 모았다.
코드가 원래 무엇이었고 어떻게 바뀌었는지도 정리했다. 숫자와 세부는 각 항목에 적힌 파일에서 확인한다.

- **실험은 2026-11-23에 종료했다(사용자 결정). 논문은 §0의 최종 결론부터 읽고, 근거는 §7·§8에서 찾는다.**
- 읽는 순서
  1. §0 최종 결론, 그다음 이 문서 전체
  2. 필요한 부분의 상세 문서
     - `docs/ai-context/AI_CONTEXT_KO.md`: INUGPU 쪽 GPU 측정, 11-17까지
     - `prediction/local-cpu-vm/INUOS_CONTEXT_KO.md`: INUOS 쪽 CPU 예측
     - `measurement/aws/README_KO.md`: AWS 실행과 결과
- 표기
  - **확인됨**: 코드, 로그, 결과 파일로 확인한 사실
  - **추정**: 근거는 있으나 직접 검증하지 않은 해석
  - **미확인**: 검증하지 않은 내용
- 단위: 결과 파일 원값은 byte다. MiB = 2^20 B, MB = 10^6 B. 허용 오차 "100 MB"는 10^8 B다.
- 날짜는 모두 2026년이다. 시각은 KST다(AWS 로그 원문은 UTC).

---

## 0. 최종 결론 (논문용 요약, 2026-11-23)

### 0.1 방법 (한 문장)
CPU에서 모델을 실제로 실행해 PyTorch 할당 요청 목록을 기록하고, GPU에서만 생기는 요청(cuBLAS·cuBLASLt·cuDNN 작업 공간 등)을 규칙과 GPU 세대별 측정 표로 넣은 뒤, PyTorch CUDA caching allocator 시뮬레이터(xMem + 내림차순 세그먼트 주소)로 재생해 reserved를 예측하고, GPU별 상수 outside를 더해 VRAM을 예측한다.

### 0.2 예측 파이프라인의 구성 요소와 근거

| 구성 요소 | 내용 | 근거 (절) |
|---|---|---|
| CPU 실행 + 요청 기록 | FP32로 실행하고 요청 크기를 FP16으로 환산(INUOS) / 실제 FP16 실행(AWS r7i) | §5, §7.2, §7.6 |
| 세그먼트 주소 규칙 | 새 세그먼트는 내림차순 주소. 3090 직접 확인, A10G·L4·L40S·Blackwell 간접 확인 | §7.3, §7.7 |
| cuBLAS 작업 공간 | 8,519,680 B, 첫 GEMM 직후. GPU 세대·PyTorch 버전으로 계산, 5종 확인 | §7.5 |
| CPU 라이브러리 버퍼 제거 | oneDNN 등 CPU에만 있는 버퍼를 합성곱 호출 안에서 뺀다 | §7.4, §7.8 |
| cuDNN 작업 공간 | 수식이 없어 GPU 세대·cuDNN 버전별로 합성곱 형태를 미리 측정한 표를 쓴다 | §7.8 |
| cuBLASLt 1 MiB, max_pool 인덱스 | GPU에만 있는 짧은 요청 2종 | §7.8 |
| cuDNN 대체 계획 | 메모리가 빠듯하면 작업 공간이 작은 계획으로 바뀌는 동작을 시뮬레이터에 반영(replay5) | §7.8 |
| outside | GPU 종류별 상수(T4 130, A10G 308, L4 232, L40S 510, RTX PRO 4500 320, RTX PRO 6000 686, 3090 312 MiB) | §7.7, §8.2 |

### 0.3 핵심 수치 (100 MB 이내 = |예측 − 실측| ≤ 10^8 B)

| 실험 | 결과 | 절 |
|---|---|---|
| RTX 3090, 기존 모델 519개 조건 (v3) | 91.9% (일반 조건 96.8%) | §8.1 |
| AWS GPU 6종 (LLM·T5·ViT, 재시도 없는 조건) | 94~98% (T4 283/302 ~ L4 303/309, g7e 311/321) | §8.2 |
| 합성곱 모델 60개 조건, 최종 방식 (세대별 cuDNN 표) | **GPU 7종 모두 60/60** (L4 반복 측정 포함 8개 실측 세트). Ampere·Ada·Turing 58개, Blackwell 53개 바이트 일치 (11-26) | §8.5 |
| **처음 보는 모델 6개 96개 조건 (3090)** | **CNN 48/48 바이트 일치, LLM 46/48** (AWS 실제 FP16 실행으로는 48/48) | §8.6 |
| 메모리 상한 실측 (cuDNN 대체 계획) | 15/15 지점 바이트 일치 | §8.7 |
| PPT 예시 (L4, OPT-125M b8 512/128) | 오차 0~2 MiB | §8.3 |

### 0.35 지원 인스턴스 (2026-11-26 기준 모두 확인)

| AWS 인스턴스 | GPU (세대) | 실측 | outside | cuDNN 표 | 합성곱 60개 |
|---|---|---|---|---|---|
| g4dn | T4 (Turing) | 395 | 130 MiB | g4dn 측정 | 60/60 |
| g5 | A10G (Ampere) | 395 | 308 | 3090 표 (같은 세대) | 60/60 |
| g6 | L4 (Ada) | 395 (+ gr6 반복) | 232 | g6 측정 | 60/60 |
| g6e | L40S (Ada) | 395 | 510 | g6 표 (같은 세대) | 60/60 |
| g7 | RTX PRO 4500 (Blackwell) | 395 | 320 | g7e 표 (같은 세대) | 60/60 |
| g7e | RTX PRO 6000 (Blackwell) | 395 | 686 | g7e 측정 | 60/60 |

- GPU별 상수(용량, outside, cuBLAS 작업 공간)는 `manifest/gpu_profiles.json`, 세대별 cuDNN 표는 `results/reserved_v3/cudnn_fit/pergpu/probes/`

### 0.4 결정 사항 (사용자)
- 11-23: qwen의 FP32 그림자 오차(RMSNorm 해제 시점)는 **보정하지 않는다**. 해당 조건은 qwen b128 긴 입력 일부뿐이고(새 모델 48개 중 2개), 실제 FP16 실행으로 원인이 확인됐다.
- 11-23: 실험을 마무리하고 논문 작업으로 넘어간다. §11의 남은 항목은 향후 과제로 쓴다.

### 0.5 논문에 반드시 쓸 한계 (§10)
- cuDNN 표는 GPU 세대 × cuDNN 버전 묶음마다 측정해야 한다(GPU당 몇 분). Ampere·Ada·Turing·Blackwell 4개 세대 표를 만들었다(11-26).
- CPU만으로 cuDNN 선택을 알아내는 NVIDIA deviceless 기능은 동작은 하지만 PyTorch의 선택과 36%만 맞아 쓰지 못했다.
- 재시도 조건(용량 근처)의 reserved, smollm2 +58 MiB 1개 조건, T4 SDPA 모델의 원인은 미확인이다.
- 반복 측정 변동이 최대 4.5%라 그보다 작은 오차는 의미가 없다.

---

## 1. 프로젝트와 최종 식

- **CloudGPUFit**: GPU를 빌리기 전에 CPU만으로 추론 워크로드(모델 + 배치 + 입력 길이 + 생성 길이 / 해상도, FP16)의 GPU VRAM 사용량을 예측하고, 맞는 AWS GPU 인스턴스를 추천하는 졸업작품이다.
- 지원 워크로드는 3종이다.
  - 디코더 LLM: OPT, TinyLlama, Qwen2.5, GPT-2
  - 텍스트 인코더-디코더: T5
  - 이미지 분류: ResNet, MobileNet, ViT
- **최종 VRAM 예측 = 예측 reserved + outside(GPU 종류별 상수)**
  - **reserved**: PyTorch CUDA caching allocator가 GPU에서 잡아 둔 메모리(`torch.cuda.memory_reserved` 최대값). CPU 실행으로 얻은 요청 목록을 할당자 시뮬레이터 xMem에 재생해서 예측한다.
  - **outside**: allocator 밖 메모리(CUDA context, 라이브러리 커널 등) = NVML 프로세스 사용량 − reserved. GPU 종류와 드라이버마다 한 번 재 둔 상수다.
- 정답값: GPU 실측 `summary.first_measure_peak_reserved`(측정 1회차의 reserved 최대값). VRAM 전체로 비교할 때는 `max_nvml_process_peak`를 쓴다.

## 2. 실험 환경 (세 곳)

| 이름 | 하드웨어 | 역할 | 소프트웨어 | 브랜치 위치 |
|---|---|---|---|---|
| **INUGPU** | RTX 3090 24 GB VM, driver 570.211.01 | GPU 실측(정답값). AWS GPU 실행 패키지 제작 | 이미지 `cloudgpufit-bench:cu124-v1` (torch 2.6.0+cu124, cuDNN 9.1.0, transformers 4.52.4). Blackwell용 `cu128-v1` (torch 2.7.1+cu128, cuDNN 9.7.1) | `measurement/local-gpu-vm/` |
| **INUOS** | CPU 전용 VM, Xeon Cascade Lake 24 vCPU, RAM 47 GiB, FP16 명령 없음 | CPU 실행, 요청 기록, xMem 재생, 예측, 분석 | 이미지 `cloudgpufit-cpu-validation:20261108` (ID `sha256:875290b5…`, torch 2.6.0+cu124 CPU 실행, transformers 4.52.4) | `prediction/local-cpu-vm/` |
| **AWS GPU** | g4dn T4, g5 A10G, g6 L4, gr6 L4(반복), g6e L40S, g7 RTX PRO 4500 Blackwell. driver 모두 595.112.07 | INUGPU와 같은 측정을 다른 GPU에서 실행(사용자가 직접 실행) | g4dn~g6e는 cu124-v1, g7은 cu128-v1 (cu124에 sm_120 커널이 없어 파일럿 26개 전부 실패) | `measurement/aws/` |
| **AWS CPU** | r7i.8xlarge, Sapphire Rapids 32 vCPU, 247 GiB (FP16 명령 있음). 새 LLM 48개는 r7i 16 vCPU·124 GiB (11-23) | INUOS와 같은 CPU 예측을 **실제 FP16 실행**으로 다시 함 | INUOS와 같은 이미지(ID 동일) | `prediction/aws-r7i-cpu/` |

- VM 간 연결
  - INUOS → INUGPU: 읽기 전용 ssh. 보고 가져오기만 하고, GPU 작업은 사용자 허락 없이 돌리지 않는다.
  - INUGPU → INUOS: scp 가능.
  - 두 VM의 AI는 직접 대화하지 못한다. handoff 문서로 주고받았다.
- AWS 쪽은 사용자가 직접 실행했다. 결과는 INUOS `~/aws-result`로 받아 INUOS에서 분석했다.

## 3. GPU 측정 방식 (정답값의 정의, 측정 프로토콜 v2)

- 코드: `measurement/local-gpu-vm/05-fp16-vram/src/vram_worker.py` (sha256 `d06f0f15…`). 상세는 `docs/ai-context/AI_CONTEXT_KO.md` §5.
- 조건 1개 = 새 Docker 컨테이너 1개다(캐시 이력 없음).
- 절차
  1. FP16 모델 로드. attention은 transformers 기본값이다(OPT·Llama·Qwen·GPT-2·ViT는 sdpa, T5·CNN은 eager).
  2. 입력 생성: 랜덤 토큰, 시드 20261006
  3. greedy 생성(`min_new_tokens = max_new_tokens`)
  4. **warmup 3회 → measure 5회**. 캐시는 유지하고, 각 measure 직전에 `reset_peak_memory_stats()`만 호출한다.
- 설정: TF32 off, `cudnn.benchmark=False`, `CUDA_MODULE_LOADING=LAZY`, `PYTORCH_CUDA_ALLOC_CONF` 미설정
- 실패 조건은 `gpu_oom` 등으로 기록하고, 배치나 dtype을 바꿔 다시 돌리지 않는다.
- 측정 프로토콜 v1(eager 고정)과 v2(기본 attention)를 파일럿 26개로 비교한 뒤 v2로 확정했다(11-11).
- 확인된 성질
  - 1회차 시작 reserved = 1회차 최대값이고, 5회 최대값이 모두 같다. 즉 reserved 최대값은 warmup이 만든 캐시 상태에서 정해진다.
  - 같은 조건을 15회 반복해도 최대 4.5% 흔들리는 조건이 있다(opt b16: 968 / 1,012 MiB). 이보다 작은 예측 오차는 구분할 수 없다.

## 4. 조건표

- 현재 조건표: `main395_l256` 395개 (sha256 `d4a9b09c…`)

| 그룹 | 모델 | 축 | 수 |
|---|---|---|---|
| 디코더 LLM | OPT-125M, TinyLlama-1.1B-Chat, Qwen2.5-1.5B-Instruct | 배치 {1,8,16,32,128} × 입력 {128,256,512} × 생성 {128,256,512} | 135 |
| GPT-2 | GPT-2, Medium, XL | 배치 × 입력 {128,512} × 생성 {128,512} | 60 |
| T5 | small, base, 3B | 입력·생성 {128,256,512} | 135 |
| CNN | ResNet-18, ResNet-50, MobileNet V2 | 해상도 128/224/256/512 | 60 |
| ViT | ViT-Large | 224 | 5 |

- 이전 조건표 `main395`(L1024)에서 1024를 256으로 바꿨다. 1024 조건이 너무 오래 걸려서 사용자가 11-15에 결정했다. 겹치는 245개는 두 표 모두에 있다.
- 모델 revision은 고정이다. 모델이나 후보를 자동으로 바꾸지 않는다.

## 5. 예측 파이프라인 (최종 형태)

```
[CPU] 조건마다 새 컨테이너
  1. GPU 로드 단계 요청 합성 (load_emu: from_pretrained(device_map=cuda) 순서 재현, 13/13 모델 RTX 3090과 동일)
  2. 모델 실행 1회(입력 복사 + generate)를 torch.profiler [memory]로 기록 → 할당/해제 요청 목록
     - INUOS: FP32로 실행하고 요청 크기를 FP16으로 환산 (shadow 태그, 환산 규칙 v1)
     - AWS r7i: 실제 FP16으로 실행 (fp16_real, 환산 없음)
  3. 같은 요청 목록을 4번(warmup 3 + measure 1) 이어 붙임 (4회 요청 목록이 같음을 36/36 조건에서 확인)
[재생] xMem CachingAllocator (commit 1ee4bd08, 원본 무수정)
  4. reserved 예측 v3: 내림차순 세그먼트 주소 + cuBLAS 작업 공간 + 실효 용량
     → R_inf (용량 무제한), R_at (해당 GPU 실효 용량, 재시도/OOM 재현), R_min (OOM 없이 도는 최소 용량)
[최종] VRAM = reserved 예측 + outside(GPU별 상수)
```

- 시뮬레이션 시점(GPU 없이)에 쓰는 정보만 쓴다.
  - 조건별 정보: CPU 실행 결과 전부
  - GPU 종류별 고정값: GPU 이름 → 세대 → cuBLAS 작업 공간 크기(계산), cuda_total_memory, outside
    - 실효 용량 = cuda_total − outside. GPU 종류마다 한 번 재 두는 표다.
  - 조건별 GPU 실측은 예측에 쓰지 않는다. 출처: `prediction/local-cpu-vm/03-fp16-cpu-prediction/handoff_codex/reserved_prediction_inputs_KO.md`

## 6. 이름이 헷갈리는 버전들 (반드시 구분)

| 축 | 이름 | 뜻 |
|---|---|---|
| 옛 INUOS 예측기 (342 조건표, FP16 이전) | 휴리스틱 v2 / native v3 / singlepass v4 | 10-26~11-10의 이전 세대 예측기다. 지금 방식과 무관하다 |
| GPU 측정 프로토콜 | v1 / v2 | v1 eager 고정, v2 기본 attention (현재) |
| CPU 작업자 | v1 `cpu_worker.py` / v2 `cpu_worker2.py` | v2가 GPU 측정 프로토콜 v2와 같은 절차다 (현재) |
| FP16 환산 규칙 (shadow rule) | v1 `shadow.py` / v2 `rule2_shadow.py` | v2는 T5 내부 임시 텐서 수정. reserved 영향 ±2 MiB로 사실상 없어서 생산값은 v1 유지 |
| CPU 실행 dtype | `fp32_shadow` / `fp16_real` | INUOS는 FP32 실행 + 환산, AWS r7i는 실제 FP16 |
| **reserved 예측 방법** | **v2 / v3** | v2 = 원본 xMem 재생(주소 원본, 작업 공간 없음, 용량 1 TiB). `result.json`의 `summary` 값. v3 = `replay3.py`(내림차순 주소 + cuBLAS 작업 공간 + 실효 용량, R_inf/R_at/R_min) |
| 합성곱 모델 방침 | (버전 없음) | CPU 라이브러리 버퍼 제거(replay4) + 세대별 cuDNN 측정 표 + cuBLASLt·max_pool 규칙. 평가 스크립트(`cudnn_fit/final_eval.py`, `newmodels_eval.py`)에 구현했다. 대체 계획은 replay5 (§7.8) |

## 7. 시간순 실험 이력 (왜 → 어떻게 → 결과 → 반영)

### 7.1 이전 단계 (03~11-10) — 요약만. 상세는 `docs/ai-context/AI_CONTEXT_KO.md` §4.0~4.2, `prediction/local-cpu-vm/INUOS_CONTEXT_KO.md` §2~3

| 시기 | 곳 | 실험 | 결과와 반영 |
|---|---|---|---|
| 03-31 | INUGPU | ResNet-50/YOLOv5s FP32 벤치 | CloudSim 입력값 용도. 이후 방향 전환 |
| 08 | INUGPU | xMem 단독 검증 프로브 | xMem을 할당자 시뮬레이터로 채택 |
| 09-14 | INUGPU | Qwen2.5-14B FP16 24 GB 적재 | strict는 LOAD_OOM |
| 10-06~27 | INUGPU | cloudgpufit-bench 342 조건 × 5 모드 | 하네스 v1 17.8 h → v3 3.7 h |
| 10-26~11-10 | INUOS | 옛 예측기 휴리스틱 v2 → native v3 → singlepass v4 (342 조건, 325 ok) | CPU에서 요청을 기록해 xMem으로 재생하는 방식이 이때 자리 잡음 |
| 11-03~10 | INUGPU + AWS | **outside 분석** | 3090 driver 570, LAZY 로딩에서 첫 추론 이후 outside는 312 MiB로 고정(모델·배치·dtype 무관, 차이 0~2 MiB). EAGER면 +728~754 MiB. AWS 기존 자료: T4 130, A10G 308, L4 232, L40S 510 MiB. → **outside를 GPU별 상수로 쓰는 근거** |
| 11-10~11 | INUOS + INUGPU | **reserved 오차 원인 분리** (342 예측 vs AWS L4) | 큰 오차의 원인은 실행 이력 차이였다. GPU는 한 프로세스에서 여러 모드를 캐시 유지로 이어 잼. 같은 GPU 요청을 xMem에 넣으면 0~2 MiB. → INUGPU가 측정을 재설계함(VRAM 전용, 로드 → warmup 3 → measure 5) |

### 7.2 FP16 VRAM 실측과 CPU 예측 기반 (11-11 ~ 11-16)

1. **INUGPU 파일럿 26개 (11-11)**
   - 왜: eager 고정과 기본 attention 중 무엇을 기준으로 할지 정하려고
   - 결과
     - 기본 attention(sdpa = FlashAttention)이 14개 중 10개에서 reserved를 줄였다(최대 −1060 MiB).
     - Qwen FP16 eager는 출력이 NaN이었다.
     - GPU 요청 목록을 xMem에 넣으면 12/12 trace가 모든 경계에서 GPU와 같았다.
   - 반영
     - 측정 프로토콜 v2 확정
     - "할당자 정책은 문제가 아니다. CPU와 GPU가 다르면 요청 목록 차이를 본다"가 이후 분석의 기본 원칙이 됨
2. **INUOS CPU 작업자 v1 → v2 (11-11~12)**
   - 왜: GPU 측정 절차와 똑같이 맞추려고
   - 어떻게
     - `protocol2.py`: GPU 작업자와 같은 입력, 생성 설정, 호출
     - `load_emu.py`: GPU 로드 단계 요청 합성. 13/13 모델이 3090과 같음
     - `shadow.py`: FP32 실행에서 FP16 요청 크기 계산. INUOS CPU는 FP16이 매우 느려서 이 방식을 씀
     - 실행 1회를 4번 이어 붙여 재생
   - 코드: 커밋 `4b5465d`, `7332115`, `0d39be7`
3. **본실험 395 (L1024) (11-12)**
   - INUGPU: 381 ok / 14 gpu_oom
   - INUOS CPU: 사용자 요청으로 중단(388 결과)
4. **L256 전환 (11-15)**
   - INUGPU L256: 384 ok / 11 gpu_oom
   - INUOS L256: 386 ok / 9 CPU 메모리 부족(gpt2-xl b128 3개, t5-3b b128 6개)
   - 겹치는 244개는 결과를 재사용했다.
5. **스레드 확인 (11-15)**
   - CPU 요청 목록은 torch 스레드 수에 따라 달라진다(스레드별 CPU 작업 버퍼). 예측 요약값은 12/12 같았다.
   - 배정 코어 수와는 무관하다(뒤의 AWS 실행 설계에 사용).
6. **v2 비교 결과 (3090, 11-15~16)**
   - LLM·T5 allocated 오차는 −0.1~−1.5%였다.
   - reserved 큰 오차는 용량 1 TiB(캐시 반환 미재현), 주소 순서, cuBLAS 작업 공간 누락에서 나왔다.
   - CNN은 allocated부터 틀렸다.

### 7.3 reserved 예측 v3 (11-16, INUOS) — 상세 `prediction/local-cpu-vm/03-fp16-cpu-prediction/results/reserved_v3/PROCESS_KO.md`

- 왜: GPU `first_measure_peak_reserved`를 CPU 정보만으로 맞히려고. 출발점은 INUGPU의 reserved 분석(E1~E7)과 가이드 `RESERVED_PREDICTION_GUIDE_KO.md`다.
- 어떻게
  1. **가이드 검증**: GPU trace 12개를 xMem에 재생했다.
     - 원본 xMem 주소로는 큰 조건에서 최대 +896 MiB 틀렸다.
     - "새 세그먼트는 기존보다 낮은 주소"(내림차순)로 바꾸면 12/12가 맞았다.
       - 이 규칙은 INUGPU 가이드가 3090 trace에서 관찰해 제안했고, INUOS가 재현했다.
       - CUDA 드라이버 동작이고 문서화된 규칙은 아니다.
     - "같은 크기면 낮은 주소 블록을 고른다"는 PyTorch 코드에 정해진 규칙이다.
  2. **cuBLAS 작업 공간**: GPU 전용 요청 8,519,680 B(해제 안 함)를 첫 GEMM 출력 직후에 넣는다.
     - 위치는 CPU 프로브(`gemm_probe --exact`)로 찾는다. 522/522 조건에서 CPU 저장 요청 목록 앞부분과 같았다.
  3. **용량**: 실효 용량 = cuda_total − outside(3090: 25,306,529,792 − 327,155,712 = 24,979,374,080 B)
     - 용량이 빠듯할 때의 캐시 반환·재시도는 xMem이 재현한다.
  4. 출력은 R_inf / R_at / R_min이다.
- 결과 (RTX 3090, 비교 519개, 허용 오차 100 MB — 사용자가 정한 기준)

  | 그룹 | 개수 | v3 | v2 |
  |---|---|---|---|
  | 일반(LLM·T5·ViT, 재시도 없음) | 436 | 96.8% | 86.9% |
  | 재시도 조건 | 23 | 60.9% | 0% |
  | CNN | 60 | 68.3% | 65.0% |
  | 전체 | 519 | 91.9% | 80.5% |

  - 재시도 판정은 23/23, GPU OOM 판정은 3/3 맞았다.
- 반영: `replay3.py`, `tools/reserved_predict.py`. 생산값(v2, `result.json`)은 보존하고 새 파일로 추가했다.
- 원인 확인 실험 (`results/reserved_v3/confirm/`)
  - **T5 24/48 MiB**
    - `isinf` 내부 임시 텐서 `abs(x)`가 bool 출력의 태그를 물려받아 FP32 크기로 남았다.
    - 환산 규칙 v2로 고쳤지만 reserved 영향은 ±2 MiB(4개 조건)로 사실상 없었다(`rule2_t5/`).
  - **qwen −560 / −2240 MiB**
    - allocated는 같은데 GPU에만 T×8960×2 세그먼트가 하나 더 있다.
    - GPU trace가 필요하다. **미해결**
  - **재시도 조건 남은 오차**
    - 첫 캐시 반환은 GPU와 같지만 두 번째 반환에서 갈라진다.
    - 재사용 정책 변경으로는 해결되지 않았다. **미해결**

### 7.4 합성곱 모델 오차 원인 — 층 분해 (11-17, INUOS) — 상세 `results/reserved_v3/confirm/hybrid/README_KO.md`, 방침 `handoff_codex/reserved_conv_models_policy_KO.md`

- 왜: v3에서 MobileNet은 +2.17 GB 과대, ResNet-50은 −1.89 GB 과소였다(3090, b128 r512).
- 어떻게
  - 요청 목록을 두 층으로 나눴다.
    - **모델 층**: 연산 출력. 장치와 무관
    - **라이브러리 층**: CPU는 oneDNN(mkldnn) 버퍼, GPU는 cuDNN·cuBLAS·cuBLASLt 작업 공간
  - CPU 층은 연산 종료 시점 표식(크기 999,983 B인 표식 요청)으로 구분했다. 처음에는 CPU malloc 주소 재사용 때문에 오분류가 있었는데 이 방식으로 고쳤다.
- 결과 (3090 GPU trace 3개)
  - CPU 모델 층과 GPU 모델 층이 해제 순서까지 같았다(3/3).
  - "CPU 모델 층 + GPU 라이브러리 층"으로 재생하면 GPU 실측과 정확히 같았다(3/3): ResNet-50 6168, ResNet-18 2548, MobileNet 6580 MiB.
  - GPU 정보 없이(CPU 모델 층 + cuBLAS만) 재생한 결과
    - MobileNet 6580: 정확
    - ResNet-18 2288: −260
    - ResNet-50 4370: −1798. 남은 차이는 전부 cuDNN 작업 공간이다.
- 결론
  - MobileNet 과대예측 = CPU 라이브러리 버퍼가 남아서 생긴 것. 제거하면 해결된다.
  - ResNet 과소예측 = cuDNN 작업 공간
    - `benchmark=False`면 같은 상황에서는 같은 크기지만, GPU가 있어야 알 수 있다.
    - 사용자 모델·입력 조합을 다 재 두는 표는 현실적이지 않다.
- 반영(방침, **구현 전**)
  - CPU 라이브러리 버퍼 제거
  - cuDNN 작업 공간은 [0, max(k×출력, c)] 범위로 보고하고 신뢰도 낮음으로 표시한다.
  - 인스턴스 추천은 범위의 상한 기준이다.
  - k, c는 GPU trace를 더 받아 정한다.

### 7.5 cuBLAS 작업 공간 크기 계산 (11-17, INUOS)

- 왜: 사용자가 상수로 넣지 말고 GPU 모델명으로 세대를 정해 계산하라고 지시했다.
- 어떻게: `gpu_rules.py`
  - GPU 이름 → compute capability(공개 사양)
  - PyTorch `CublasHandlePool.cpp`의 규칙: cc 9.0(Hopper)은 32 MiB, 나머지는 4096 KiB×2 + 16 KiB×8 = 8,519,680 B. `CUBLAS_WORKSPACE_CONFIG`가 있으면 그 값
- 결과: AWS 실측에서 T4·A10G·L4·L40S(torch 2.6)와 Blackwell(torch 2.7.1) 모두 8,519,680 B로 확인됐다(§7.7).

### 7.6 AWS CPU 실제 FP16 실행 (11-17 16:08 ~ 11-18 02:21, r7i.8xlarge)

- 왜: INUOS의 FP32 실행 + 환산 단계를 없애고, 실제 FP16 CPU 실행으로도 같은 예측이 나오는지 보려고. INUOS CPU에는 FP16 명령이 없다.
- 어떻게 (`tools/run_aws.py`, `aws/` 스크립트, 커밋 `bfa58c4`)
  - INUOS와 같은 이미지, 작업자, 조건표, 조건별 torch 스레드 수를 썼다.
  - 작업자 옵션 `--dtype-mode fp16_real`: 환산 없이 실행하고, 같은 실행 안에서 첫 GEMM 위치를 기록한다.
  - 병렬 설계
    - 요청 목록은 스레드 수에만 의존하고 코어 수와는 무관하다(§7.2-5).
    - 그래서 스레드 수는 유지하고 배정 코어만 줄여 여러 조건을 동시에 돌렸다.
    - 오래 걸리는 조건부터 넣었다(LPT).
    - 컨테이너마다 메모리 상한을 걸었다.
  - 실패 조건은 마지막 단계 S에서 96 GiB로 다시 돌렸다.
  - 원래 r7i.16xlarge를 계획했지만 vCPU 한도 32 때문에 8xlarge로 실행했다.
- 결과
  - **395/395 ok**: INUOS에서 메모리 부족이던 9개도 성공했다. P 단계에서 메모리 부족이던 t5-3b 3개는 S 단계에서 성공했다.
  - 10.2시간 걸렸다.
  - v3 재생까지 끝났다(`prediction/aws-r7i-cpu/results/reserved_v3/aws_fp16_l256/predictions.jsonl`).
- INUOS 방식과 비교 (§8.4)
  - LLM은 거의 같다.
  - T5는 AWS가 약간 나쁘다(원인 미확인).
  - CNN은 CPU FP16 oneDNN 버퍼가 커서 더 크게 과대예측한다.

### 7.7 AWS GPU 5종 실측과 간접 검증 (측정 11-16~17 사용자, 분석 11-18 INUOS) — 상세 `prediction/local-cpu-vm/03-fp16-cpu-prediction/results/reserved_v3/aws_gpu_check/README_KO.md`

- 왜: 3090에서만 확인한 규칙(내림차순 주소, cuBLAS 크기, outside)이 다른 GPU에서도 성립하는지 보려고
- 어떻게
  - AWS 결과에는 할당기 trace가 없다.
  - 그래서 측정 시작 할당량과 최대 할당량이 3090과 바이트까지 같은 조건(= 같은 요청 목록)만 골라, reserved가 3090과 같은지 봤다.
  - 판별용 조건: 원본 주소와 내림차순 주소의 예측이 다르고, 3090 실측이 v3와 정확히 같았던 조건
- 결과

  | GPU | 측정 결과 | 3090과 2 MiB 이내 | 판별용 조건에서 v3와 같음 | outside | 실효 용량 (B) |
  |---|---|---|---|---|---|
  | T4 (Turing 7.5) | 378 ok / 17 OOM | 170/239 | 16/44 | 130 MiB | 15,500,771,328 |
  | A10G (Ampere 8.6) | 382 / 13 | 286/292 | 48/52 | 308 MiB | 23,363,977,216 |
  | L4 (Ada 8.9) | 382 / 13 | 266/275 | 48/52 | 232 MiB | 23,417,978,880 |
  | L40S (Ada 8.9) | 390 / 5 | 269/275 | 48/52 | 510 MiB | 47,132,966,912 |
  | RTX PRO 4500 (Blackwell 12.0, torch 2.7.1) | 387 / 8 | 255/265 | 46/48 | 320 MiB | 33,354,285,056 |
  | L4 gr6.4xlarge (반복) | 382 / 13 | 265/273 | 48/52 | 232 MiB | 23,417,978,880 |

  - A10G, L4, L40S, Blackwell: 내림차순 규칙이 3090처럼 성립한다.
    - 판별용 조건에서 안 맞은 것은 전부 ±2 MiB였다. 주소 규칙이 틀렸다면 20~226 MiB가 나와야 한다.
  - T4: 판별할 수 없다.
    - T5(SDPA 미사용)는 96/97이 3090과 같다.
    - SDPA 모델만 24·48·96 MiB씩 작다. 추정 원인은 flash attention을 못 써서 요청 목록 자체가 다른 것이다(미확인).
  - gr6(L4)는 g6와 같은 GPU를 다른 인스턴스에서 잰 것이다.
    - reserved가 376/382 조건에서 g6와 바이트까지 같았다. 나머지 6개는 실행 간 변동(−92 ~ +78 MiB)이다.
    - OOM 13개와 outside 232 MiB도 같았다. 같은 GPU 종류면 인스턴스가 달라도 결과가 재현된다.
  - cuBLAS: 5종 모두 디코더 조건 190~192개에서 (측정 시작 할당량 − 로드 직후 할당량)이 3090과 같았다 → 8,519,680 B
  - outside: 모델 유형과 상관없이 거의 상수다.
    - L4는 LLM·T5 317개 전부 232 MiB였고, 합성곱 모델(ViT 포함)은 234~236 MiB였다. cuDNN 로드 때문으로 추정한다.
    - 범위: T4 130~132, A10G 308~312, L4 232~236, L40S 510~514, Blackwell 318~326 MiB
- 반영
  - `gpu_profiles.json`에 5종의 실측 칸을 채움
  - `gpu_rules.VERIFIED` 확장
  - 커밋 `fbb1b19`

### 7.8 cuDNN 작업 공간 표, 일반화 시험, deviceless 준비 (11-20 ~ 11-23, INUOS + INUGPU)

- 1단계 (CPU 라이브러리 버퍼 제거, GPU 정보 불필요): `src/cgf680/layer_shadow.py`, `replay4.py` (drop="conv")
  - 합성곱 60개 조건, 100 MB 이내: 3090 41→50, T4 42→51, L4 46→55
- 2단계 (GPU별 단일 계수): 실패. 최적 계수가 0이었다. 작업 공간이 형태마다 달라서다.
- 3단계 (세대별 형태 측정 표): `gpu_probe/cudnn_ws_probe.py` v3 (INUGPU 제안 반영)
  - 형태만 GPU에서 실행해서 1순위 작업 공간과 대체 후보를 잰다. 3,253개에 91초 걸렸다.
  - + GPU 전용 라이브러리 요청 2개(cuBLASLt 1 MiB, max_pool2d int64 인덱스): 3090·A10G 60/60, 바이트 일치 58, 최대 +4 MiB
  - 다른 세대(Ada, Blackwell)에는 그 세대의 표가 필요하다. 3090 표를 쓰면 Ada는 오히려 나빠진다.
  - 상세: `results/reserved_v3/cudnn_fit/README_KO.md`
- 일반화 시험 (규칙에 안 쓴 모델 6개, 96개 조건, 3090): `results/reserved_v3/newmodels_eval/README_KO.md`
  - CNN 3개 48/48 바이트 일치 (새 형태 953개를 측정한 뒤)
  - LLM 3개 46/48. qwen2.5-0.5b 2개가 −610 MiB(FP32 그림자 현상으로 추정, AWS fp16_real로 확인 중), smollm2 1개 +58 MiB
- 격자 규칙 (측정한 형태가 없을 때 가장 가까운 격자 형태의 배수를 씀): 80%가 1 MiB 이내. 새 형태의 27%는 격자에 없는 종류였다. `cudnn_fit/grid_rule/README_KO.md`
- HF 합성곱 층 목록: 구조가 다른 config 1,021개 → meta 장치 실행 623개(다운로드 96.6%) → 층 구성 1,354개, 종류 23개. `results/conv_catalog/README_KO.md`
- cuBLASLt 1 MiB는 bias가 있는 linear 전부(LLM·ViT)에서 생긴다. T5에는 없다. small pool이라 영향은 2 MiB 이하다. `results/reserved_v3/lib_requests_llm/README_KO.md`
- 대체 계획 시뮬레이터 `replay5.py` — **GPU 검증 15/15 일치** (`results/reserved_v3/fallback/README_KO.md`)
  - 메모리가 충분하면 replay3와 108/108 같다.
  - 메모리 상한 실측과 비교했다(측정 전에 예측 기록, resnet-50 7개 + regnet 8개 지점). 성공/OOM, reserved, allocated가 모두 바이트 단위로 같았다.
  - 대체 계획을 모르면 최소 필요 메모리를 과대평가한다: resnet-50 b128 r512 +514 MiB, regnet +1,026 MiB
  - 11-16에 원인을 모르던 R_min 차이(resnet-50 예측 5,142 vs GPU 4,642)가 이것으로 설명된다.
- cuDNN 버전 영향 (같은 3090·torch 2.6, 버전 6개): `results/reserved_v3/cudnn_version/README_KO.md`
  - 형태의 85~87%는 버전이 바뀌어도 같다. 버전은 묶음으로 바뀐다: 9.1 | 9.5=9.7 | 9.10=9.12 | 9.28
  - 9.1 표로 다른 버전을 예측하면 CNN 108개 중 105~108개가 100 MB 이내다. 최대 −1,538 MiB (9.10, resnet-50 b128 r512)
- cuDNN deviceless (NVIDIA 공식, 9.8 API·9.12 정식, Ampere 이상)
  - CPU 전용에서 동작했다(libcuda stub 사용, check_support 생략). 후보 목록은 GPU 쪽의 부분집합이다(eng3/4/6/58 빠짐).
  - PyTorch가 실제로 고르는 계획과 1순위가 맞는 것은 36%뿐이다. 선택 규칙을 모델링하지 못해 예측에는 측정 표를 쓴다.
  - PyTorch 휠의 cuDNN: 2.6=9.1.0, 2.7=9.5.1 (cu128 이미지 9.7.1), 2.8~2.10=9.10.2

## 8. 보고서에 쓸 핵심 수치

### 8.1 reserved 예측 v3 정확도 (RTX 3090, 519개)
- §7.3 표. 일반 조건 96.8%, 전체 91.9% (100 MB 이내)

### 8.2 AWS GPU별 정확도 (v3 R_inf, 재시도 없는 조건, CNN 제외, INUOS 예측 기준, 100 MB 이내)
- A10G 299/309, L4 303/309 (gr6 302/309), L40S 308/319, RTX PRO 4500 306/314, T4 283/302, RTX PRO 6000 (g7e, 11-26) 311/321
- R_at(GPU별 실효 용량으로 재시도까지 재현)은 아직 GPU별로 다시 돌리지 않았다.

### 8.3 PPT 예시 (팀이 정한 조건)
- **g6.xlarge(L4), OPT-125M, 배치 8, 입력 512, 출력 128, FP16**

  | 기준 | 실측 | 예측 (AWS FP16 실행 + v3) | 예측 (INUOS 환산 + v3) |
  |---|---|---|---|
  | reserved | 918,552,576 B (876 MiB) | 918,552,576 B, 오차 0 (0.00%) | 916,455,424 B, −2 MiB (−0.23%) |
  | NVML 전체 (예측 = reserved + 232 MiB) | 1,161,822,208 B (1,108 MiB) | 오차 0 (0.00%) | −2 MiB (−0.18%) |

  - outside 232 MiB는 같은 g6 측정의 중앙값이다(GPU별 사전 측정 상수). 범위는 232~236 MiB다.
- **cuDNN 한계 예시: g6.xlarge, ResNet-18, 배치 128, 해상도 512**
  - 실측 reserved: 3,474,980,864 B (3,314 MiB), NVML 3,720,347,648 B (3,548 MiB)
  - INUOS 환산 + v3: 2,290 MiB → **−1,024 MiB (−30.9%)**, NVML 기준 −1,026 MiB (−28.9%). 원인은 cuDNN 작업 공간 누락
  - AWS FP16 실행 + v3: 3,506 MiB → +192 MiB (+5.8%)
    - CPU FP16 oneDNN 버퍼 과대분(약 +1,216)이 cuDNN 누락분(−1,024)을 우연히 상쇄한 값이다.
    - 정확도로 쓰면 안 된다. 같은 방식으로 MobileNet b128 r512는 +4,314 MiB다.
  - cuDNN 작업 공간은 GPU마다 다르다. ResNet-50 b128 r512의 실측은 3090·A10G 6,168 MiB, L4 4,372 MiB이고, 예측은 4,370 MiB다.

### 8.4 CPU 실행 방식 비교 (v3 R_inf, 재시도 없는 조건, 100 MB 이내 개수, INUOS / AWS)

| GPU | LLM·ViT | T5 | CNN |
|---|---|---|---|
| T4 | 166 / 162 (185) | 117 / 112 (117) | 42 / 41 (60) |
| A10G | 184 / 186 (191) | 115 / 110 (118) | 41 / 42 (60) |
| L4 | 186 / 186 (191) | 117 / 112 (118) | 46 / 44 (60) |
| L40S | 187 / 188 (195) | 121 / 115 (124) | 46 / 44 (60) |
| Blackwell | 189 / 189 (194) | 117 / 112 (120) | 37 / 45 (60) |

- 원본: `results/reserved_v3/aws_gpu_check/aws_vs_inuos_rinf.txt`
- 결론
  - 실제 FP16 실행은 LLM에서는 환산 방식과 같은 수준이다.
  - T5는 약간 나쁘다. CPU FP16 행렬곱 버퍼 때문으로 추정한다(미확인).
  - CNN은 CPU 버퍼 제거 없이는 비교할 의미가 적다.

### 8.5 합성곱 모델 최종 (60개 조건, 1단계 + 세대별 cuDNN 표 + cuBLASLt·max_pool, 2026-11-26)

| GPU (세대) | 표 | 100 MB 이내 | 바이트 일치 | 오차 범위 |
|---|---|---|---|---|
| RTX 3090 (Ampere) | 3090 | 60/60 | 58 | 0 ~ +4 MiB |
| A10G (Ampere) | 3090 | 60/60 | 58 | 0 ~ +4 |
| L4 g6·gr6 (Ada) | g6 L4 | 60/60 | 58 | 0 ~ +4 |
| L40S (Ada) | g6 L4 | 60/60 | 58 | 0 ~ +4 |
| T4 (Turing) | g4dn T4 | 60/60 | 58 | 0 ~ +4 |
| RTX PRO 4500 g7 (Blackwell) | g7e RTX PRO 6000 | 60/60 | 53 | −8 ~ +2 |
| RTX PRO 6000 g7e (Blackwell) | g7e | 60/60 | 53 | −8 ~ +2 |

- 3090 표를 다른 세대에 쓰면 Ada 48/60, Blackwell 55/60이었다. 세대별 표가 필요하고, 같은 세대 안에서는 표 하나로 된다(g7은 측정 없이 g7e 표로 60/60).
- 원본: `results/reserved_v3/cudnn_fit/pergpu/README_KO.md`, `final_eval_pergpu_summary.json`

### 8.6 처음 보는 모델 6개 (RTX 3090, 96개 조건)

| 모델 | 100 MB 이내 (INUOS) | 바이트 일치 | AWS 실제 FP16 실행 |
|---|---|---|---|
| convnext-tiny-224 | 16/16 | 16 | — |
| efficientnet-b0 | 16/16 | 16 | — |
| regnet-y-040 | 16/16 | 16 | — |
| pythia-410m | 16/16 | 15 | 16/16 |
| smollm2-360m | 16/16 | 15 (+58 MiB 1개) | 16/16 (+58 그대로) |
| qwen2.5-0.5b | 14/16 | 12 (−610 MiB 2개) | 16/16 (0 / −2 MiB) |

- 원본: `results/reserved_v3/newmodels_eval/summary.json`, `aws_fp16_real/compare.json`

### 8.7 메모리 상한에서 cuDNN 대체 계획 (RTX 3090, 측정 전 예측 기록)
- resnet-50 b128 r512: 상한 4,642~5,130 MiB에서 reserved 4,630 / allocated 3,835.1 MiB, 4,610에서 OOM → replay5 7/7 일치. 기존 replay3는 5,142 아래를 OOM으로 예측
- regnet-y-040 b128 r512: 상한 4,872~5,640에서 4,870 / 4,849.7, 4,840에서 OOM → 8/8 일치
- 대체 계획을 모르면 최소 필요 메모리를 0.5~1 GB 과대평가한다.

### 8.8 cuDNN 버전 영향 (같은 3090·torch 2.6)
- 형태의 85~87%는 버전과 무관. 9.1 표로 다른 버전을 예측하면 CNN 108개 중 105~108개가 100 MB 이내(최대 −1,538 MiB)

## 9. 코드 계보 (원래 무엇이었고, 어떻게 바뀌었나)

### 9.1 바꾸지 않은 원본
- **xMem** (`perf_estimator/allocator`, commit `1ee4bd08`)
  - PyTorch CUDA caching allocator 시뮬레이터. 원래 학습용으로 만들어졌다.
  - 파일 sha256을 `provenance/xmem-allocator-lock.json`으로 고정하고 실행 때마다 검사한다.
  - 원본이 다루지 않는 것
    - KV cache 같은 추론 생성 단계: 학습용이라 범위 밖
    - cuBLAS 작업 공간, 실제 드라이버의 세그먼트 주소 순서, 용량 제한에 따른 재시도: 학습 규모에서는 영향이 작아 단순화된 것으로 추정
  - 바꾼 방식: xMem 파일은 그대로 두고, 재생할 때 인스턴스 단위로 주소 정책을 바꾸고, 요청을 추가하고, 용량을 인자로 준다(`replay3.py`).
- **GPU 측정 작업자** `vram_worker.py` (INUGPU, sha256 `d06f0f15…`): 정답 코드다. CPU 쪽이 이 절차를 따라간다.

### 9.2 INUOS 코드의 변화 (`prediction/local-cpu-vm/03-fp16-cpu-prediction/`, 브랜치 `exp/fp16-680-cpu-20261111`)

| 단계 | 파일 | 원래 → 바뀐 것 | 이유 | 커밋 |
|---|---|---|---|---|
| 기반 | `02-reserved-diag/src/cgf_diag/replay.py`, `events.py` | 이전 실험의 순서 재생기 | xMem에 요청을 순서대로 넣는 공통 코드 | (이전 레포) |
| CPU 작업자 v1 | `cpu_worker.py`, `protocol.py`, `shadow.py` | 새로 작성: warmup 3 + measure 1을 한 프로파일러 세션에서 기록, FP32 실행 → FP16 크기 환산 | INUOS CPU는 FP16이 느림 | `4b5465d`, `7332115` |
| CPU 작업자 v2 | `cpu_worker2.py`, `protocol2.py`, `load_emu.py` | v1 → GPU 작업자와 같은 입력·생성 설정, 로드 단계 합성, 실행 1회 기록 후 4번 재생 | GPU 측정 프로토콜 v2와 같게 하려고 | `0d39be7` |
| 스케줄러 | `run395.py` → `run_l256.py` → `run_aws.py` | 병렬 슬롯, 메모리 기반 진입 제어, 코어 축소 병렬(AWS) | 실행 시간과 메모리 부족 대응 | `912f7fe`, `7acc8be`, `7b9a5c4`, `bfa58c4` |
| 용량 재생 | `tools/replay_capacity.py` | 1 TiB 대신 GPU 용량으로 재생 | 캐시 반환 재현 | `912f7fe` |
| reserved v3 | `replay3.py`, `gemm_probe.py`, `gemm_shadow.py`, `tools/reserved_predict.py`, `tools/compare_reserved.py` | v2 재생 → 내림차순 주소, cuBLAS 삽입, 실효 용량, R_inf/R_at/R_min, 가벼운 trace | 가이드 검증 결과 반영 | `2c66f7b`, `4b47bdb` |
| 환산 규칙 v2 | `rule2_shadow.py`, `cpu_worker2.py --shadow-rule v2` | 내부 임시 텐서 태그를 입력에서 상속 | T5 24/48 MiB 원인 | `bfa58c4` (효과 확인 `f0f0812`) |
| 실제 FP16 실행 | `cpu_worker2.py --dtype-mode fp16_real`, `gemm_shadow.FirstGemmRecorder` | 환산 없이 실행하고 첫 GEMM 위치 기록 | AWS FP16 CPU용 | `bfa58c4` |
| 층 분해 | `results/reserved_v3/confirm/hybrid/*.py` | 분석 스크립트(예측기 아님) | 합성곱 오차 원인 | `93b2189`, `cc797f0`, `78d3a69` |
| GPU별 상수 | `gpu_rules.py`, `manifest/gpu_profiles.json`, `reserved_predict --gpu a,b,c` | 상수 8,519,680 → 모델명으로 계산, 여러 GPU 동시 예측 | 사용자 지시, 다른 GPU 대응 | `1b01701`, `fbb1b19` |
| 층 구분 + CPU 버퍼 제거 | `layer_shadow.py`, `replay4.py` (drop="conv"), `tools/run_layer_labels.py` | 요청마다 소속 연산·합성곱 인자를 붙이고, 합성곱 안의 CPU 라이브러리 요청을 뺀다. replay3는 그대로 | 합성곱 과대예측(MobileNet) | `aef8aaf` (GitHub) |
| cuDNN 형태 측정 | `gpu_probe/cudnn_ws_probe.py` v1→v3, `run_probe.sh` | 모델 없이 합성곱 형태만 GPU에서 실행해 작업 공간과 대체 후보를 잰다. v3는 INUGPU 제안 | GPU 세대별 표 | `4a41cdb` |
| 최종 평가 | `results/reserved_v3/cudnn_fit/final_eval.py`, `newmodels_eval/newmodels_eval.py` | 1단계 + 표 + cuBLASLt·max_pool 규칙을 넣어 재생 | 합성곱 오차 해결 | `3fe8a15`, `553afb0` |
| 대체 계획 | `replay5.py` | replay3 + 작업 공간 할당 실패 시 다음 후보(형태별 기억) | 메모리가 빠듯한 경우 | `553afb0`, `029789f` |
| deviceless | `gpu_probe/deviceless/deviceless_query.py` | GPU 속성 파일만으로 cuDNN 후보 조회 (예측에는 미사용) | 측정 없이 표를 얻을 수 있는지 시험 | `029789f` |

- 원칙: 생산 예측값(`result.json` v2)과 과거 결과는 덮어쓰지 않는다. 바꾼 것은 새 파일이나 옵션으로 추가했다. 그래서 v2와 v3가 함께 남아 있다.

## 10. 한계와 미해결 (보고서에 명시할 것)

1. **cuDNN 작업 공간** (11-23 갱신)
   - CPU만으로는 알 수 없어 GPU 세대 × cuDNN 버전 묶음마다 합성곱 형태를 측정한 표가 필요하다(§7.8). Ampere·Ada·Turing·Blackwell 표를 만들었고 GPU 7종 모두 60/60이다(§8.5).
   - 표에 없는 형태는 격자 규칙으로 추정할 수 있지만 정확도가 떨어진다(80%가 1 MiB 이내, 새 종류는 불가).
   - NVIDIA deviceless(CPU 조회)는 PyTorch의 선택을 36%만 맞혀 쓰지 못했다.
2. **CPU 라이브러리 버퍼 제거·cuDNN 표·대체 계획은 평가 스크립트와 replay4/replay5에 구현했다.** 사용자용 단일 명령으로는 묶지 않았다.
3. **재시도 조건**(용량 근처): v3로 60.9%. 두 번째 캐시 반환부터 갈라지며 원인 미확정이다.
4. **qwen 특정 조건 −560 / −610 MiB**: 원인 확인(FP32 실행 시 RMSNorm `.to(float32)` 해제 시점, §7.8·§8.6). 실제 FP16 실행으로는 사라진다. 보정은 하지 않기로 했다(§0.4). smollm2 +58 MiB 1개는 미확인
5. **T4의 SDPA 모델**: 요청 목록 차이로 추정. T4 trace 필요
6. **내림차순 주소 규칙**: 3090에서 직접 확인했고, 다른 4종은 간접 확인이다. 드라이버는 570과 595 두 가지만 봤다.
7. **반복 측정 변동**: 같은 조건에서 최대 4.5%. 이보다 작은 오차는 의미가 없다.
8. **CPU 메모리 한계**: INUOS(47 GiB)에서는 큰 조건 9개를 예측하지 못했다. 큰 RAM 인스턴스에서는 가능하다(AWS 395/395).

## 11. 향후 과제 (실험 종료 2026-11-23, 논문의 Future work)

| 항목 | 필요한 것 |
|---|---|
| (완료 11-26) Ada·Blackwell·T4 cuDNN 표 → §8.5 | - |
| 예측기를 사용자용 단일 명령으로 묶기 (표·규칙·replay5) | 개발 |
| qwen FP32 그림자 보정 (RMSNorm 해제 시점) — 보정하지 않기로 결정 | — |
| smollm2 +58 MiB, AWS FP16 실행의 T5 악화, 재시도 조건, T4 SDPA 원인 | GPU trace |
| deviceless에서 PyTorch 선택 규칙 재현 | cuDNN 9.12 이상을 기본 탑재한 PyTorch |
| (완료 11-26) g7e(RTX PRO 6000) 실측·비교 → §0.35 | - |

## 12. 운영 제약 (사용자 지시, 실험 해석에 영향)

- 생산 예측 로직과 과거 결과는 보존한다. 원인 확인 전에는 보정식을 넣지 않는다.
- 실패 조건을 낮은 배치나 다른 dtype으로 몰래 바꾸지 않는다. 모델 ID나 후보를 자동으로 바꾸지 않는다.
- 허용 오차는 100 MB다. INUGPU 가이드의 32 MiB는 함께 표기만 한다.
- INUGPU는 읽기 전용이다. GPU 작업은 사용자 허락이 있어야 한다.
- 키, 토큰, IP, 인스턴스 ID 같은 접속 정보는 기록하지 않는다. AWS 로그의 인스턴스 ID와 내부 호스트명은 지우고 올렸다.

## 13. 파일 지도 (자주 찾는 것)

| 찾는 것 | 위치 |
|---|---|
| 3090 GPU 조건별 실측 | `measurement/local-gpu-vm/05-fp16-vram/results/gpu_rtx3090_main395_L256_per_condition.csv` (L1024도 같은 폴더) |
| AWS GPU 조건별 실측 | `measurement/aws/<instance>/runs/<label>/<condition_id>/measure.json`, 요약 `logs/main395_*_summary.md` |
| INUOS CPU 결과 | `prediction/local-cpu-vm/03-fp16-cpu-prediction/runs/main395_l256/cases/<id>/result.json` |
| AWS CPU 결과 | `prediction/aws-r7i-cpu/runs/aws_fp16_l256/cases/<id>/result.json` (요청 목록 npz는 제외) |
| v3 예측 (INUOS) | `prediction/local-cpu-vm/03-fp16-cpu-prediction/results/reserved_v3/rtx3090_v3/predictions.jsonl` |
| v3 예측 (AWS CPU) | `prediction/aws-r7i-cpu/results/reserved_v3/aws_fp16_l256/predictions.jsonl` |
| 3090 비교 보고서 | `prediction/local-cpu-vm/03-fp16-cpu-prediction/handoff_codex/reserved_prediction_report.md`, `_per_condition.csv` |
| AWS GPU 비교 | `prediction/local-cpu-vm/03-fp16-cpu-prediction/results/reserved_v3/aws_gpu_check/` (`per_condition.csv`, `summary.json`, `aws_vs_inuos_rinf.txt`) |
| GPU별 상수 | `prediction/local-cpu-vm/03-fp16-cpu-prediction/manifest/gpu_profiles.json`, `src/cgf680/gpu_rules.py` |
| 예측 입력 출처 | `prediction/local-cpu-vm/03-fp16-cpu-prediction/handoff_codex/reserved_prediction_inputs_KO.md` |
| 실험 과정 | `prediction/local-cpu-vm/03-fp16-cpu-prediction/results/reserved_v3/PROCESS_KO.md` |
| 이미지 재현 | `prediction/local-cpu-vm/03-fp16-cpu-prediction/docs/image/IMAGE_KO.md` |
| 합성곱 오차 줄이기·3090 cuDNN 표 | `prediction/local-cpu-vm/03-fp16-cpu-prediction/results/reserved_v3/cudnn_fit/README_KO.md`, `gpu_probe/` |
| 처음 보는 모델 6개 (+ AWS 실제 FP16) | `prediction/local-cpu-vm/03-fp16-cpu-prediction/results/reserved_v3/newmodels_eval/` |
| 대체 계획 GPU 검증 | `prediction/local-cpu-vm/03-fp16-cpu-prediction/results/reserved_v3/fallback/` |
| cuDNN 버전 영향, deviceless | `prediction/local-cpu-vm/03-fp16-cpu-prediction/results/reserved_v3/cudnn_version/`, `gpu_probe/deviceless/` |
| HF 합성곱 층 목록 | `prediction/local-cpu-vm/03-fp16-cpu-prediction/results/conv_catalog/` |
| qwen 원인 | `prediction/local-cpu-vm/03-fp16-cpu-prediction/results/reserved_v3/confirm/trace_confirm_20261120/README_KO.md` |
| 레포에 없는 큰 파일 | `EXCLUDED_LARGE_FILES.md`, `prediction/local-cpu-vm/EXCLUDED_LARGE_FILES.md`, `measurement/aws/README_KO.md` §4 |

## 14. 용어

| 용어 | 뜻 |
|---|---|
| allocated | 텐서가 실제로 쓰는 메모리 |
| reserved | 할당자가 GPU에서 잡아 둔 세그먼트 합. allocated + 쪼개진 빈 공간 |
| segment | cudaMalloc으로 받은 큰 덩어리. 작은 블록 풀은 2 MiB, 큰 요청은 별도 세그먼트 |
| 재시도(retry) | cudaMalloc 실패 시 캐시된 빈 세그먼트를 반환하고 다시 시도하는 동작(`num_alloc_retries`) |
| R_inf / R_at / R_min | 용량 무제한 / 해당 GPU 실효 용량 / OOM 없이 도는 최소 용량에서의 예측 reserved |
| 실효 용량 | cuda_total_memory − outside. 할당자가 쓸 수 있는 GPU 메모리 |
| shadow 태그 | FP32 실행 중 각 텐서가 FP16 실행이라면 어떤 dtype이었을지 표시하는 방식(h: 절반, f: FP32 고정, x: 비부동소수) |
| 모델 층 / 라이브러리 층 | 요청 목록 중 연산 출력 / 라이브러리가 따로 잡는 버퍼 |
| cuBLAS 작업 공간 | 첫 cuBLAS 호출 때 PyTorch가 잡고 놓지 않는 버퍼(8,519,680 B, Hopper 32 MiB) |
| cuDNN 작업 공간 | 합성곱 알고리즘이 쓰는 임시 버퍼. 크기는 GPU·cuDNN 버전·형태에 따라 다름 |

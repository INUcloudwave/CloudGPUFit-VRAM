# INUOS VM — CloudGPUFit CPU VRAM 예측 실험 이력 (2026-11-10 reserved diag 이전)

- 작성일: 2026-11-15. 작성 방식: VM 파일·git·로그를 읽기 전용으로 확인함.
- 범위: 2026-11-10 reserved diag(`/home/inuos/cgf-reserved-diag-20261110`), 2026-11-11 FP16-680(`/home/inuos/cgf-fp16-680-20261111`) **이전** 작업. 두 실험 자체는 다른 문서에서 다룸.
- 표기
  - "확인": 파일·git·로그에서 직접 확인함.
  - "추정": 근거는 있지만 직접 확인하지 못함.
  - "미확인": 근거 파일이 없음.
  - "(본 문서 계산)": 이번 작성 중 원본 JSON/CSV를 `python3 -I`로 읽어 집계한 값. 원본 문서에는 없는 값임.
- 개인정보(작성자 이름, 이메일, GitHub 계정, 키·토큰)는 넣지 않음. 저장소 URL과 Notion URL도 생략함. URL은 원본 파일에 있음(예: `aws-actual-20261110/source-manifest.json`).

---

## 0. 한눈에 보기

- 목표
  - 모델·dtype·배치·길이 조건을 받아 CPU에서 GPU VRAM을 예측함.
  - CPU에서 실제 추론을 `torch.profiler`로 실행해 텐서 storage 할당/해제 수명을 기록함.
  - 기록한 수명을 xMem `AllocatorSim`(PyTorch CUDACachingAllocator 재현)에 재생해 allocated/reserved peak를 구함.
  - CPU RAM이나 CPU 실행시간을 VRAM이나 GPU 응답시간으로 환산하지 않음.
- 예측기 버전은 네 단계로 바뀜.
  1. **v2(휴리스틱)**, 10-26 baseline9
     - FP32 forward(prefill, `use_cache=False`)를 xMem 학습 추정기 경로로 실행함.
     - dtype 환산 ×0.5, 과소예측 보정 +32MiB×scale, KV 해석식을 더함.
  2. **v3 `3-native-inference`**, 11-03 커밋 a95710e/7f822d7
     - native dtype `from_config` 모델로 full generate를 실행함.
     - warmup 1회 후 측정 3회를 같은 allocator에 연속 재생함.
  3. **v3 + recovery 패치**, 11-08
     - 예측 값 의미는 v3와 같음. 메모리 최적화 파서·phase 로그·컨테이너별 RAM 제한을 추가함.
  4. **v4 `4-native-inference-single-request`**, 11-08 singlepass
     - warmup 0회, 측정 1회로 바꿈. 이 버전으로 342조건 결과 패키지를 만듦(11-10).
- 최종 산출물: `/home/inuos/CloudGPUFit_CPU_predictions_20261110`
  - 342조건 중 성공 325, 실패 17(cgroup_oom 16 + timeout 1).
  - 이 패키지가 reserved diag의 CPU 기준선(AWS L4 g6.2xlarge 실측과 비교)으로 쓰임.

---

## 1. 타임라인

| 날짜/시각(KST) | 실험/작업 | 상태 | 주요 위치 |
|---|---|---|---|
| 2026-08-23 ~ 10-01 | repo 초기 배포, 웹 wizard·orchestrator·latency 예측기 통합(VRAM 예측 실험 아님) | 참고 | `/home/inuos/CloudGPUFit` (git 24f5369~9e6e798) |
| 10-26 18:00~18:10 | 커밋 930094a(고정 2GiB margin 제거), c25e26a/c6a1125(baseline9 실행기) | 커밋 | 같은 repo |
| 10-26 18:20 | baseline9 1차 실행: `git apply --reverse --check`가 컨테이너 안에서 exit 127로 실패 | 실패(0/9) | `/home/inuos/cloudgpufit-eval/run_20261026T092019Z_432983` |
| 10-26 18:22~18:38 | baseline9 2차 실행: v2 휴리스틱 예측기, FP16 9조건 | 완료 9/9 | `/home/inuos/cloudgpufit-eval/run_20261026T092222Z_434474` |
| 10-28~10-29 | Generate runtime(latency) 예측 커밋 d667003, 2d6198b(VRAM과 무관) | 커밋 | repo |
| 11-03 12:49 | 커밋 a95710e: native inference tracing(v3)과 GPU outside 보정 도입 | 커밋 | repo |
| 11-03 15:06~15:12 | native-v3 baseline9, threads=24 시험. 1건 성공 후 2번째 건에서 중단 | 중단(1/9) | `/home/inuos/cloudgpufit-predictions/native-v3-t24` |
| 11-03 15:22 | 커밋 7f822d7: jobs 병렬과 RAM 예약 스케줄링 추가 | 커밋(HEAD) | repo |
| 11-03 15:26 ~ 11-05 13:48 | native-v3 all342 실행(threads 4, jobs 6, ram-budget 36). host OOM 16건 발생 후 수동 중단 | 중단: 성공 244 / 오류 17 / 미실행 81 | `/home/inuos/cloudgpufit-predictions/native-v3-t4-j6` |
| 11-05 13:51 | 진단 자료 수집(kernel log, docker inspect, 컨테이너 소스·allocator 복사) | 완료 | `/home/inuos/cloudgpufit-cpu-diag-f9XTYy1G` |
| 11-08 15:27 | 2차 진단 수집(kernel log 3일치 비어 있음) | 완료 | `/home/inuos/cloudgpufit-cpu-diag-0XMg6UoG` |
| 11-08 15:52 | ChatGPT 작성 복구 패키지(fix.patch 등) | 패키지 | `/home/inuos/CloudGPUFit_CPU_recovery_20261108(.zip)` |
| 11-08 16:16 | Codex 인수인계 문서 | 문서 | `/home/inuos/CloudGPUFit_Codex_handoff_20261108.md` |
| 11-08 ~16:19~16:40 | recovery worktree 적용 + VM 검증(recovery-check-01~04, lifecycle 테스트) | 완료: 참조 5건 0 byte 차이, 실패 1건 복구 | `/home/inuos/CloudGPUFit-cpu-recovery-20261108`, `/home/inuos/cloudgpufit-predictions/recovery-*` |
| 11-08 ~17:33~17:41 | singlepass(v4: warmup 0 + 측정 1) 변경과 VM 검증(singlepass-check-01~06) | 완료: 6건 0 byte 차이 | `/home/inuos/CloudGPUFit-cpu-singlepass-20261108`, `.../singlepass-*` |
| 11-08 ~18:01~18:05 | 호스트 RAM 기준 병렬 admission scheduler와 검증 | 완료 | `.../parallel-check-01`, `.../parallel-audit-20261108` |
| 11-08 18:05:59 ~ 18:30:01 | all342 1차 병렬 실행(jobs 2). 10건 성공 후 SIGTERM으로 중단 | 중단(10 ok) | `.../parallel-all342-20261108-180550` |
| 11-08 18:30, 18:32 | 재실행 시도 2회가 인자 오류로 실패(pressure-kill 패치 적용 전) | 실패 | `.../parallel-all342-20261108-183011.console.log`, `-183226.console.log` |
| 11-08 18:32:59 ~ 11-10 02:59:53 | all342 2차 실행(남은 332건, jobs 5, pressure-kill, solo 재시도 36GiB) | 완료: 315 ok / 17 err | `.../parallel-all342-20261108-183250` |
| 11-10 10:41 | OOM 16건 무제한 RAM 재시도(assistant 시작). 사용자 요청으로 90초 만에 중단 | 중단 | `.../oom-unlimited-20261110-103935`, `.../oom-unlimited-audit-20261110-103935` |
| 11-10 10:44~10:52 | AWS 실측 tarball 6개 수집, 342 결과 패키지 export(10:51) | 완료 | `/home/inuos/cloudgpufit-comparison`, `/home/inuos/CloudGPUFit_CPU_predictions_20261110(.zip)` |
| 11-10 10:53:51 ~ 18:36:10 | OOM 16건 무제한 RAM 수동 재시도. 5건 처리 후 취소 | 중단: 2 ok / 2 OOM / 1 취소 | `.../oom-unlimited-manual-20261110-105344` |
| 11-10 이후 | reserved diag(범위 밖). 위 342 패키지와 singlepass 소스를 기준선으로 씀 | — | `/home/inuos/cgf-reserved-diag-20261110` |
| 11-12 | INUOS↔INUGPU 교차 검토. FP16-680/RTX3090 계열이며 342 실험이 아님 | 질문 대기 | `/home/inuos/cloudgpufit-review` |

---

## 2. 공통 실행 환경과 조건 계획

### 2.1 호스트와 컨테이너

- 호스트 `INUOSVM-1`. QEMU VM, Intel Xeon (Cascadelake) 24 vCPU(소켓 24 × 코어 1), RAM 약 47GiB(MemTotal 50,510,213,120 B), swap 없음.
  - 근거: `recovery-audit-20261108/lscpu.json`, manifest `last_admission_memory`.
  - CPU flags에 avx512 계열은 있으나 avx512_fp16/amx는 없음. FP16 CPU 연산이 느린 배경이며, FP16-680 메모에서 "FP16 hw 없음"으로 정리됨.
- 기존 예측 컨테이너 `xmem-cpu`
  - base 이미지는 `orchestrator/setup_manager.py:20`의 `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`.
  - 실제 컨테이너 이미지 ID `sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee`.
  - Python 3.11.11, torch 2.6.0+cu124, CUDA runtime 12.4, Transformers 4.52.4, GPU 없음(`gpu_available=false`).
  - Docker `Memory=0`, `MemorySwap=0`, 즉 RAM 제한 없음(`cloudgpufit-cpu-diag-*/container.txt`).
  - bind mount 2개, 둘 다 RW
    - `/home/inuos/xmem-work` → `/work`
    - `/home/inuos/.cache/huggingface` → `/root/.cache/huggingface`
  - 컨테이너는 평소 `sleep infinity`로 대기하고 `docker exec`로 예측을 실행함.
- xMem: `/home/inuos/xmem-work/xMem` @ `1ee4bd08dda277a6248abee5b6d40e0d6067efaa`(upstream xMem).
  - `core_inference.patch` 적용 흔적 있음: `exp/trainer/train_loop.py`, `trainer.py`, `perf_estimator/estimator.py` 수정됨.
  - v3 소스 4개가 미추적 파일로 복사돼 있음: `predict_vram.py` sha `1223f582…`, `cpu_tensor_trace.py` sha `1ea7c371…`, `vram_policy.py`, `gpu_overheads.json`.
  - `inference_trace_runtime.py`는 없음. recovery/singlepass 실행은 이 디렉터리를 동기화하지 않고 `/recovery` 별도 mount를 씀.
  - allocator 파일 8개 hash는 `benchmarks/aws342/xmem-allocator-lock.json`으로 고정함(allocator.py `7ac16de1…`, simulator.py `487e990a…` 등).
- recovery/singlepass용 로컬 snapshot 이미지: `cloudgpufit-cpu-validation:20261108` = `sha256:875290b5854ad6dbc8ea554fecb292dd5ffdfea2a03e8c4e8179a3c827968fa5`.
  - `xmem-cpu`의 writable layer를 `docker commit`해 만든 로컬 전용 이미지임. 업로드하지 않음.
  - writable layer는 약 5.2GB로 기록됨.

### 2.2 342조건 계획 (`benchmarks/aws342/selected-plan.json`, sha256 `e85edc8b…`)

- AWS GPU 벤치마크 계획에서 가져온 core suite 342조건. 9모델 × FP32/FP16/BF16.
  - decoder_llm 126 = 3모델 × 3dtype × 14
  - text_encdec 135 = 3 × 3 × 15
  - vision_cls 81 = (small, medium 각 11 + large 5) × 3dtype

| 유형 | small | medium | large |
|---|---|---|---|
| decoder_llm | openai-community/gpt2 | gpt2-medium | gpt2-xl |
| text_encdec | google-t5/t5-small | t5-base | t5-3b |
| vision_cls | google/mobilenet_v2_1.0_224 | microsoft/resnet-50 | google/vit-large-patch16-224(조건 01~05만) |

- revision은 `benchmarks/aws342/revisions.json`(sha256 `4061fddb…`)으로 고정함. 예: gpt2 `607a30d7…`, t5-3b `bed96aab…`.
- 조건 번호 → 파라미터. 번호만 보고 추정하지 말고 plan을 읽어야 함.
  - decoder_llm(B, seq_len, max_new_tokens)
    - 01 (1,256,64), 02 (2,256,64), 03 (4,256,64), 04 (8,256,64)
    - 05 (1,128,64), 06 (1,512,64), 07 (1,768,64)
    - 08 (1,256,16), 09 (1,256,128), 10 (1,256,256)
    - 11 (4,512,128), 12 (4,768,128), 13 (8,128,16), 14 (1,768,256)
  - text_encdec(B, enc, dec, new)
    - 01 (1,256,1,64), 02 (2,…), 03 (4,…)
    - 04 enc128, 05 enc512, 06 enc1024
    - 07 dec32, 08 dec128
    - 09 new16, 10 new128, 11 new256
    - 12 (2,512,1,128), 13 (2,1024,1,128), 14 (4,128,1,16), 15 (1,512,128,128)
  - vision_cls(B, res)
    - 01~05: B 1/2/4/8/16 @224
    - 06 (1,128), 07 (1,384), 08 (1,512), 09 (4,384), 10 (4,512), 11 (8,128)
- GPU 쪽 설정(plan의 `runner_settings`/`execution_profiles`)
  - seed 20261006, eval, inference_mode, use_cache, greedy, beams 1, min=max new tokens.
  - 조건마다 새 프로세스. `allocator_cache_policy: retain_within_case`.
  - 프로필 `local_core`: warmup 1, 모드 performance/memory/…, memory 모드 3회. `aws_core`: repeats 3, warmup 10, iterations 100.
  - 다만 GPU 비교 대상 L4 실측의 실제 실행 절차는 reserved diag 문서 기준으로 "performance → memory 연속, 캐시 유지"임(§5 참조).

---

## 3. 실험별 상세

### 3.1 [10-26] baseline9, v2 휴리스틱 예측기 (`/home/inuos/cloudgpufit-eval`)

- 목적: 고정 2GiB margin을 제거한 예측기(커밋 930094a)로 9모델 FP16 기준조건(01)을 CPU에서 예측하고 결과를 묶어 전달함.
- 실행기: `tools/run_prediction_baseline9.py`(c25e26a 작성, c6a1125에서 safe.directory 수정). 결과 폴더에 `runner.used.py`로 복사됨(git c6a1125 버전과 동일 확인).
- 설계
  - FP16 01조건만 실행. LLM은 B1/seq256/new64/generate, T5는 enc256/dec1/new64/generate, Vision은 224/prefill.
  - `gpu_memory_gb=24`(xMem 시뮬레이션 용량).
  - 예측기 `predict_vram.py`(sha `259151…`, c6a1125 버전)
    - 모델을 FP32 `from_config`로 만듦.
    - xMem `ModelTrainer(on_cpu=True, iterations=2)` + `ProfilerPlugin`으로 추론 루프를 프로파일링함(`core_inference.patch`의 `transformer_infer_loop`, `use_cache=False` prefill).
    - xMem `XMem.estimate()`의 `memory.tensor` 값에 FP16/BF16이면 ×0.5를 곱함(`DTYPE_SCALE`).
    - `UNDERESTIMATE_CONSTANT` 32MiB×scale을 더함.
    - generate 모드면 KV를 해석식 `2·L·H_kv·d·(len+new)·B·bytes`로 더함.
    - `required = corrected + kv`.
- 실행 방법
  - 컨테이너 `xmem-cpu` 안에서 다음 순서로 진행함.
    1. `git apply --reverse --check`로 patch 적용 여부 확인
    2. `docker cp`로 예측기 복사 후 hash 확인
    3. 조건별 `docker exec … python -u predict_vram.py --json -` 순차 실행
  - commit 메시지상 sudo로 detached 실행됨. 실제 실행 명령 기록은 bash_history에 없음(미확인).
- 결과
  - 1차(`run_20261026T092019Z_432983`): `git apply --reverse --check`가 컨테이너 안에서 exit 127로 실패해 0건임.
    - exit 127은 command not found임. 컨테이너에 git이 없었던 것으로 추정함.
    - 2분 뒤 같은 커밋으로 2차가 성공함. 그 사이 조치 내용은 미확인.
  - 2차(`run_20261026T092222Z_434474`): 9/9 성공, 총 971초.

| case(FP16 01) | 예측 required(GB) | allocated_scaled(B) | KV 해석식(B) | 시간(s) |
|---|---:|---:|---:|---:|
| gpt2 | 0.29 | 282,182,656 | 11,796,480 | 30.6 |
| gpt2-medium | 0.74 | 748,975,104 | 31,457,280 | 67.5 |
| gpt2-xl | 3.12 | 3,238,257,408 | 98,304,000 | 213.6 |
| mobilenet_v2 | 0.03 | 12,558,336 | — | 23.4 |
| resnet-50 | 0.07 | 56,999,936 | — | 20.1 |
| vit-large | 0.59 | 612,687,872 | — | 80.1 |
| t5-small | 0.14 | 129,927,936 | 798,720 | 41.5 |
| t5-base | 0.45 | 468,615,424 | 2,396,160 | 102.6 |
| t5-3b | 5.46 | 5,821,949,696 | 25,559,040 | 391.4 |

- 이 방식의 문제는 a95710e의 `docs/VRAM_INFERENCE_REVIEW.md`에 정리돼 있음.
  - 모든 텐서를 FP32→FP16으로 일괄 절반 환산함.
  - 32MiB 경험 보정이 들어감.
  - prefill과 KV 해석식을 단순 합산함. T5 cross-KV가 누락될 수 있음.
  - 학습 추정기의 가정이 섞임.
  - reserved(segment)를 쓰지 않음.
- 출력 파일(run 폴더)
  - `manifest.json`, `summary.csv`, `predictions.jsonl`
  - `<case>.input.json`, `<case>.log`, `<case>.prediction.json`
  - `predict_vram.used.py`, `core_inference.used.patch`, `runner.used.py`, `prediction-results.tar.gz`
- 상위 폴더: `LATEST.txt`, `latest-predictions.tar.gz`, `background_*.log`, `launcher.pid`(434471).

### 3.2 [11-03] native v3 도입 (커밋 a95710e → 7f822d7)

- a95710e "Replace heuristic VRAM corrections with native inference tracing and scoped GPU overhead calibration": 26파일, +1678/−395.
  - 신규 파일
    - `cpu_tensor_trace.py`, `vram_policy.py`, `gpu_overheads.json`
    - `benchmarks/aws342/{selected-plan,revisions,xmem-allocator-lock}.json`
    - `tools/run_prediction_suite.py`, `tools/compare_vram_predictions.py`, `tools/calibrate_gpu_overheads.py`, `tools/smoke_native_inference.py`
    - `docs/VRAM_INFERENCE_REVIEW.md`, 테스트 3종
  - `predict_vram.py`
    - DTYPE_SCALE, 보정 상수, KV 해석식을 제거함.
    - native dtype `from_config`, `attn_implementation="eager"`, seed 20261006, `torch.set_num_threads(threads)`를 씀.
  - `tools/run_prediction_baseline9.py`는 9줄짜리 호환 진입점으로 축소됨(suite에 `--case-set baseline9` 전달).
- 7f822d7 "Run isolated CPU predictions concurrently with bounded jobs and RAM reservations"
  - suite에 `--jobs`, `--ram-budget-gb`, `ram_reservation_gb()`(small 6 / medium 10 / large LLM 20·Vision 16·T5 28), ThreadPool 병렬을 추가함.
  - `cpu_tensor_trace.py`: `Config(run_id=uuid)`를 줘서 병렬 실행 시 xMem 임시 경로가 충돌하지 않게 함.
- GPU outside 보정(`gpu_overheads.json`, calibration_id `outside-bd902817a45c2b20`)
  - 정의: 같은 샘플의 NVML process bytes − PyTorch reserved.
  - 산정: FP16 조건01 9개의 case 중앙값들의 중앙값. 상한은 학습 샘플 최대값.

| GPU | outside 중앙값 | 학습 샘플 최대 | 지원 dtype |
|---|---:|---:|---|
| NVIDIA T4 | 132 MiB | 132 MiB | fp16, fp32 |
| NVIDIA A10G | 308 MiB | 310 MiB | fp16, fp32, bf16 |
| NVIDIA L4 | 232 MiB | 234 MiB | fp16, fp32, bf16 |
| RTX 3090 | 312 MiB | 314 MiB | fp16, fp32, bf16 |

- L40S는 보정값이 없어 제외함.
- 예측 NVML = reserved + outside 중앙값. 후보 필터용 required = reserved + 학습 최대값.

### 3.3 [11-03] native-v3 baseline9 threads=24 시험 (`/home/inuos/cloudgpufit-predictions/native-v3-t24`)

- 명령(bash_history): `python3 -u tools/run_prediction_suite.py --output-root ~/cloudgpufit-predictions/native-v3-t24 --case-set baseline9 --threads 24`
- git a95710e, 이미지 `77f17f84…`, identity threads=24.
- 결과
  - gpt2 FP16 01만 성공: allocated 310,894,080 B, reserved 346,030,080 B, 344.2초, tensor lifetime 99,267개.
  - 2번째 gpt2-medium FP16 01은 log 0 byte 상태에서 멈춤. manifest는 `running`, completed 1. 15:12 이후 기록이 없음.
  - 중단 사유 미확인. 15:22에 7f822d7 커밋 후 threads 4·jobs 6으로 재시작함(추정).

### 3.4 [11-03 ~ 11-05] native-v3 all342 (`/home/inuos/cloudgpufit-predictions/native-v3-t4-j6`)

- 목적: v3 예측기로 342조건 전체를 CPU에서 예측함.
- 실행 명령(bash_history). 직전에 `git pull --ff-only origin codex/vram-native-inference-20261103` 실행함.
  ```
  python3 -u tools/run_prediction_suite.py --output-root ~/cloudgpufit-predictions/native-v3-t4-j6 \
    --case-set all --threads 4 --jobs 6 --ram-budget-gb 36 --prepare 2>&1 | tee .../console.log
  ```
- 설계
  - git 7f822d7, 예측기 sha `1223f582…`, trace sha `1ea7c371…`.
  - 조건마다 `docker exec -i -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=1 -w /work/xMem xmem-cpu python -u predict_vram.py --json -`로 공유 컨테이너 안에서 별도 Python을 시작함.
  - 프로토콜
    - warmup 1회(추적 밖) 후 profiler 안에서 측정 3회(`record_function("cloudgpufit_inference")` 3구간)를 실행함.
    - 3구간의 수명을 **한 allocator에 연속 재생**함. 평균을 내는 방식이 아님.
  - 상주 storage: 모델 parameter·buffer와 입력의 고유 storage를 첫 구간 시작 1µs 전에 할당된 것으로 합성함.
  - xMem 용량 `gpu_memory_gb=80`.
  - RAM 예약 36은 스케줄링용 추정치이며 실제 제한이 아님. 컨테이너 Memory=0.
- 결과(manifest와 파일)
  - manifest: completed 260, success 244, elapsed 154,837.26초(43.01시간), status `running`(갱신 안 된 상태).
  - 파일 기준: 성공 244, 오류 17(모두 `Prediction JSON missing; inspect the case log`), 미실행 81.
    - 미실행 81 = t5-3b 45건 전부 + gpt2-xl FP32/BF16 28건 + XL FP16 07~14 8건.
  - 성공 분포 (본 문서 계산)
    - Vision 81 전부, T5 small/medium 90 전부 성공.
    - gpt2 FP16 12, FP32 11, BF16 14. gpt2-medium FP16 11, FP32 10, BF16 12. gpt2-xl FP16 3.
  - 가장 오래 걸린 성공 건
    - XL FP16 02: 20,957.7초. XL FP16 01: 11,292.6초.
    - medium FP16 04: 12,882.4초. 같은 조건의 FP32는 130.8초, BF16은 196.2초.
  - 가장 빠른 건: Vision small FP32, 약 3초.
- 오류 원인 (진단 문서와 kernel.log로 확인)
  - host global OOM kill 16건(`cloudgpufit-cpu-diag-f9XTYy1G/kernel.log`).
  - 오류 JSON 저장 시각과 0~2초 차이로 일대일 대응함. PID 직접 연결이 아닌 시각 기반 대응임.
  - 나머지 1건(XL FP16 06)은 사용자가 수동 중단한 것임. 오류 JSON 시각 11-05 13:48.

| 조건(core__decoder_llm__) | B/in/new | 희생 PID RSS(GiB) | kill 시각 |
|---|---|---:|---|
| small__fp32__11 / 12 / 14 | 4/512/128, 4/768/128, 1/768/256 | 18.51 / 17.51 / 16.69 | 11-03 15:56~15:58 |
| small__fp16__14 / 12 | | 14.57 / 14.88 | 11-03 16:04, 16:11 |
| medium__fp16__11 / 12 | | 28.73 / 29.45 | 11-03 22:36, 22:40 |
| medium__fp32__10 / 14 / 11 / 12 | 1/256/256 등 | 18.31 / 27.51 / 43.78 / 44.41 | 11-03 22:45~22:48 |
| medium__fp16__14 | | 20.98 | 11-03 22:46 |
| medium__bf16__11 / 12 | | 22.25 / 25.71 | 11-03 22:58, 23:00 |
| large__fp16__03 (B4/256/64) | | 46.25 | 11-04 20:04(약 11.44시간 실행 후) |
| large__fp16__04 (B8/256/64) | | 46.25 | 11-05 08:57(약 12.87시간 실행 후) |

- 진단에서 확인한 사항(handoff·diagnosis.md 요지, 코드로 대조함)
  1. `ram_reservation_gb`는 정밀도·배치·길이를 반영하지 않는 추정치임. RAM을 실제로 제한하지 않음. 단위도 GB/GiB로 정의되지 않음.
  2. 한 프로세스 안에서 모델·입력, native profiler 결과, 전체 trace JSON(`json.loads`), 수명 쌍·MemoryBlock, allocator 이력이 동시에 살아 있음.
  3. `selected_lifetimes`가 전체 이벤트를 timestamp로 정렬함.
  4. upstream `Trace._record`가 이벤트마다 `list(self._segments)`와 메시지를 저장함. O(E×S) 참조 비용이 들며 peak 계산에는 필요 없음.
  5. 실행기가 결과 JSON을 먼저 파싱하고 종료 코드를 나중에 봄. 그래서 `Prediction JSON missing`이 실제 원인을 가림.
  6. upstream `AllocatorSim.simulate`는 모든 예외를 `oom=True`로 바꿈.
  7. `gpuGetDeviceCount failed with code 35` 경고는 성공 244건 로그에도 모두 있음. 실패 원인이 아님.
  8. XL 구간은 단일 실행 중에도 OOM이 났음. 병렬 수를 줄이는 것만으로는 해결되지 않음.
- 진단 폴더 내용
  - `/home/inuos/cloudgpufit-cpu-diag-f9XTYy1G`(11-05 13:51:06 수집)
    - `kernel.log`: `journalctl -k --since "3 days ago"`, 1620줄
    - `container.txt`
    - `predict_vram.py`, `cpu_tensor_trace.py`: manifest hash와 일치
    - `allocator/`: lock hash와 일치
    - `diagnostics.tar.gz`: native-v3-t4-j6 결과 807 엔트리
  - `/home/inuos/cloudgpufit-cpu-diag-0XMg6UoG`(11-08 15:27:48 수집): 같은 구성, `kernel.log`는 "-- No entries --", tar 없음. 소스 hash는 11-05와 같음.

### 3.5 [11-08] 복구 패키지 (ChatGPT 작성) — `/home/inuos/CloudGPUFit_CPU_recovery_20261108`

- 파일: `README.md`, `diagnosis.md`, `fix.patch`(60,459 B), `prepare_recovery.py`, `checksums.json`(base_commit 7f822d7).
- ZIP 3종(목록만 확인)
  - `CloudGPUFit_CPU_recovery_20261108.zip`(15:52): 위 5파일.
  - `CloudGPUFit_CPU_recovery_VM_validated_20261108.zip`(16:40): README, prepare_recovery.py, VM_VALIDATION.md, fix.patch(76,312 B), validated-source.json, summary.json, remaining-after-validation.json, preservation-check.json, patch-application-check.json, lifecycle-results.json, vm-tests.log, host-tests.log, checksums.json.
  - `CloudGPUFit_CPU_singlepass_validated_20261108.zip`(17:41): README, prepare_recovery.py, VM_VALIDATION.md, fix.patch(87,420 B), validated-source.json, summary.json, preservation-check.json, vm-tests.log, checksums.json.
- `prepare_recovery.py`: `git worktree add --detach <target> 7f822d7` 후 `git apply --check`와 `git apply`를 실행함. fetch/push는 하지 않음.
- 처음 `--repo .`로 실행했을 때 `fatal: not a git repository`가 남. 저장소 밖에서 실행한 것이 원인이었고, 절대경로로 다시 실행해 worktree를 만듦.
- 패치 내용(요지)
  - 신규 `inference_trace_runtime.py`: 스트리밍 JSON 파서, phase/RSS 로그, `SummaryTrace`, 오류 분류.
  - `cpu_tensor_trace.py`: profiler 수명을 helper scope로 분리하고 메모리 이벤트만 보관함.
  - `predict_vram.py`: phase, 오류 stage·traceback.
  - 신규 `tools/run_prediction_recovery.py`: 작업마다 일회용 컨테이너, RAM hard limit, swap 비활성, timeout, OOMKilled 구분, 정확 비교(`--compare-with`).
  - legacy suite 실행 차단(`--allow-unbounded-legacy-run` 필요).
  - 버전·cache key 갱신, 테스트 추가.
- ChatGPT 환경 검증: Python 3.12/torch 2.6.0+cpu, 회귀 39개와 통합 15개 통과. Docker는 mock임. VM 검증은 미완료 상태였음.

### 3.6 [11-08] recovery VM 검증 — `/home/inuos/CloudGPUFit-cpu-recovery-20261108`

- worktree 상태
  - detached @7f822d7, `.git`은 worktree 파일(`gitdir: /home/inuos/CloudGPUFit/.git/worktrees/...`).
  - 수정: cpu_tensor_trace.py, orchestrator/cache_key.py, orchestrator/setup_manager.py, predict_vram.py, tests/test_vram_prediction.py, tools/run_prediction_suite.py, vram_policy.py.
  - 추가: inference_trace_runtime.py, tools/run_prediction_recovery.py(342줄, 순차 전용), tests/test_memory_optimized_trace.py, docs 2개.
- VM에서 발견해 추가로 고친 문제(`docs/cpu_recovery_vm_validation_20261108.md`)
  1. Docker 29.7.2가 `commit --pause=false` 경고를 stdout에 출력함. 이 경고와 SHA가 이미지 ID로 쓰여 `invalid reference format`이 남. `docker image inspect`로 ID를 얻도록 수정함. recovery-check-01이 이 오류로 2건 모두 `runner_error`, 추론 없음.
  2. 새 predictor 파일을 읽기 전용 원본 아래 overlay하지 않고 별도 `/recovery` RO mount에서 실행하게 함. `PYTHONPATH=/recovery:/work/xMem`.
  3. `--snapshot-image`로 기존 snapshot을 재사용할 수 있게 함. 환경 일치를 확인하고 매번 5.2GB commit하는 것을 피함.
  4. `inference_trace_runtime.py`를 cache fingerprint에 포함함. 참조 비교 시 revision도 확인함.
  5. phase를 `measured_inference`와 `profiler_stop`으로 분리함.
  6. container ID, host PID/cgroup, 제한, mount, 최종 상태를 기록함.
- 실행 조건
  - 순차 실행, threads 4, `--cpus 4`, `--memory 32GiB`(memory.max=34359738368), swap 0, host reserve 6GiB, timeout 3600초.
  - `/work`, HF cache, `/recovery` 모두 RO. `--network none`.
- 테스트
  - host 40개 중 36 통과, 4 skip.
  - VM 격리 40/40 통과. 통합(작은 GPT-2/T5/ViT/ResNet/MobileNetV2 × 3dtype) 15/15 통과.
  - lifecycle 실측
    - exit 7, OOM 증거 없는 exit 137 → process error로 분류됨.
    - 64MiB 제한 작업에 256MiB 할당 → `OOMKilled=true`/`cgroup_oom`.
    - timeout과 SIGTERM 취소가 구분됨. 작업 컨테이너는 모두 제거됨.
- 원래 크기 결과(v3 의미: warmup 1 + 측정 3)

| case | allocated B (old → new) | reserved B (old → new) | 차이 | 총 s | peak RSS GiB | cgroup peak GiB |
|---|---:|---:|---|---:|---:|---:|
| decoder_llm small fp16 01 | 310,894,080 | 346,030,080 | 0/0 | 335.53 | 2.682 | 2.576 |
| decoder_llm small fp32 01 | 593,162,240 | 694,157,312 | 0/0 | 35.37 | 2.850 | 2.643 |
| text_encdec small fp32 01 | 252,521,472 | 278,921,216 | 0/0 | 24.87 | 1.876 | 1.662 |
| vision large fp32 01 | 1,229,447,168 | 1,237,319,680 | 0/0 | 11.53 | 2.143 | 1.909 |
| vision small fp32 01 | 31,478,784 | 52,428,800 | 0/0 | 6.03 | 0.659 | 0.421 |
| decoder_llm small fp32 12 (과거 OOM) | 참조 없음 → 1,479,036,928 | → 1,965,031,424 | 비교 불가 | 111.38 | 30.744 | 30.619 |

- phase 시간(초)
  - gpt2 fp16 01: init 13.8, warmup 71.7, 측정 3회 223.2, profiler stop 6.4, export 2.3, parse 6.5, matching 0.3, allocator 5.7.
    - FP16에서는 실제 CPU 추론 시간이 지배적임.
  - gpt2 fp32 12: warmup 끝 RSS 2.116GiB. 측정 추론이 끝난 시점에 이미 30.400GiB였음.
    - 즉 RAM 증가는 export/parse/allocator 이전, **profiler가 켜진 추론 구간 안에서** 일어남.
    - profiler 보관분, 텐서 workspace, CPU allocator 캐시 중 무엇 때문인지는 분리하지 못함.
- 결과 위치: `/home/inuos/cloudgpufit-predictions/recovery-check-01`(Docker 29 실패 보존), `-02`(GPT-2 2건), `-03`(T5 1건 + Vision 2건), `-04`(fp32 12 복구), `recovery-audit-20261108`.
  - audit에는 `before.json`(원본 787파일 hash), lifecycle 결과, cgroup/mount 증거, `historical-costs.json`, `fp16-resource-samples.jsonl`, `remaining-after-validation.json`(남은 97건) 등이 있음.
- 결론(문서 기록)
  - 남은 조건 97건 = 원래 오류 16 + 미실행 81.
  - 큰 GPT-2 XL/T5-3B를 바로 돌리는 것은 시기상조로 판단함.

### 3.7 [11-08] singlepass v4 (warmup 0 + 측정 1) — `/home/inuos/CloudGPUFit-cpu-singlepass-20261108`

- 목적
  - 한 요청의 peak 예측에는 반복이 필요 없다는 판단으로 기본값을 warmup 0회, 측정 1회로 정의함.
  - 별도 버전 `4-native-inference-single-request`로 분리함.
  - full generate, native dtype, KV 추적, 입력 shape, seed, allocator lock은 그대로 둠.
- 342 실행에 쓰인 predictor 소스가 이 worktree임.
  - 5파일 hash가 패키지 `source/predictor/`와 일치함(본 문서에서 cmp로 확인).
  - predict_vram `7fd049d5…`, cpu_tensor_trace `e16048e3…`, inference_trace_runtime `6c999a90…`, vram_policy `28cde527…`, gpu_overheads `8cd43c6d…`.
- git 상태: detached @7f822d7 + 미커밋 변경(§3.7.1).
- VM 검증
  - 테스트: VM 43개, 통합 15개 통과.
  - 대표 6건이 이전 1+3 결과와 allocated/reserved 모두 0 byte 차이(`singlepass-check-01~06`).
  - 실행 조건: 같은 snapshot, 4 threads, 32GiB, 순차.

| case | 1+3 시간 s | 0+1 시간 s | 비율 | peak RSS GiB (old → new) |
|---|---:|---:|---:|---|
| gpt2 small fp32 01 | 35.37 | 15.89 | 2.23x | 2.850 → 2.018 |
| mobilenet fp32 01 | 6.03 | 5.89 | 1.02x | 0.659 → 0.624 |
| t5-small fp32 01 | 24.87 | 12.06 | 2.06x | 1.876 → 1.167 |
| vit-large fp32 01 | 11.53 | 10.13 | 1.14x | 2.143 → 1.869 |
| gpt2 small fp32 12 | 111.38 | 43.24 | 2.58x | 30.744 → **31.426** |
| gpt2 small fp16 01 | 335.53 | 104.55 | 3.21x | 2.682 → 1.564 |

- 해석(문서 기록)
  - 반복 제거로 시간은 줄었지만 RAM 문제는 그대로임. fp32 12는 단일 추론에서도 31.4GiB임.
  - FP16 단일 generate도 약 79.9초 걸림.
- 추가 검증 (본 문서 계산)
  - v3 t4-j6 성공 244건과 v4 패키지 값을 같은 case_id로 비교함.
    - allocated·reserved 모두 같은 건: 232건.
    - 하나라도 다른 건: 12건.
  - 다른 12건은 모두 v4 ≤ v3임.
    - T5-base fp16/bf16 03·05·15: reserved −12,582,912 B(−12MiB). allocated는 같음.
    - gpt2 small fp16/bf16 11·13: allocated −1,048,576 B. reserved는 같음.
    - resnet bf16 04: allocated −212,480 B.
    - mobilenet fp32 07·11: reserved −14,680,064 B. 07은 allocated도 −98,816 B.
    - mobilenet fp16 08: allocated −1,008,640 B.
  - 따라서 "1+3과 0+1이 byte 단위로 같다"는 대표 6건에서만 성립함. 전체에서 항상 같지는 않음.
- 산출물
  - `/home/inuos/cloudgpufit-predictions/singlepass-check-01..06`
  - `singlepass-audit-20261108`: `run_representatives.py`, `summary.json`, `fix.patch`, `validated-source.json`, `github-publication.json`
- `docs/cpu_single_request_validation_20261108.md`, `docs/cpu_recovery_errors_20261108.md`
- GitHub 반영 기록
  - `singlepass-audit/github-publication.json`: 원격 브랜치 `codex/vram-native-inference-20261103`에 commit `6ca425ca…`로 16파일을 올렸고 `remote_content_matches: true`라고 기록됨.
  - `parallel-audit/github-publication.json`: commit `c153bb22…`(scheduler 4파일)로 기록됨.
  - 로컬 `/home/inuos/CloudGPUFit`에는 두 객체가 없음. origin ref도 7f822d7에 머물러 fetch되지 않음. 원격 실제 내용은 **미확인**.
  - 이후 적용한 `cgf-pressure-kill.patch`와 11-10의 무제한 RAM 변경은 publication 기록이 없음. 원격 미반영으로 추정함.

#### 3.7.1 singlepass worktree 미커밋 diff (7f822d7 대비)

| 파일 | 변경 |
|---|---|
| `cpu_tensor_trace.py` (75줄 변경) | `json.loads` 전체 로드 → `read_memory_events` 스트리밍. profiler를 `capture()` 안 로컬로 두고 start/stop을 명시하고 phase로 감쌈. `selected_lifetimes(events, expected_invocations)`. `trace_and_estimate(..., trace_output, allocator_history, warmup_invocations=0, measured_invocations=1)`. allocator는 `inference_trace_runtime.simulate`(SummaryTrace)로 실행. trace_summary에 `request_policy`, `allocator_initial_state`, `total_profiler_events`, `allocator_history_recorded` 추가 |
| `predict_vram.py` | `phase('model_initialization')`. trace 옵션 4개를 CLI/JSON으로 전달(`--warmup_invocations` 기본 0, `--measured_invocations` 기본 1, `--trace_output`, `--allocator_history`). 오류 JSON에 `stage`(LAST_PHASE)와 `traceback` 추가 |
| `vram_policy.py` | `VERSION = "4-native-inference-single-request"` |
| `orchestrator/cache_key.py` | `VRAM_PREDICTOR_VERSION = "4-native-inference-single-request-outside-bd902817a45c2b20"`. fingerprint에 inference_trace_runtime.py 추가 |
| `orchestrator/setup_manager.py` | `_sync_predictor` 대상에 inference_trace_runtime.py 추가 |
| `tools/run_prediction_suite.py` | `--allow-unbounded-legacy-run`이 없으면 실제 실행을 거부함(preflight만 허용). identity 소스에 inference_trace_runtime.py 추가 |
| `tools/smoke_native_inference.py` | measured==1, warmup==0, request_policy=='cold_single_request' assert |
| `tests/test_native_trace_and_suite.py`, `tests/test_vram_prediction.py` | 위 정책에 맞춘 테스트 수정 |
| 미추적 신규 | `inference_trace_runtime.py`, `tools/run_prediction_recovery.py`(500줄, 병렬·pressure-kill·무제한 모드), `tools/prediction_scheduler.py`(219줄), `tests/test_memory_optimized_trace.py`, `tests/test_prediction_scheduler.py`, `tests/test_unlimited_recovery.py`, `cgf-pressure-kill.patch`(적용됨), docs 5개 |

- `cgf-pressure-kill.patch`(560줄)는 tests/test_prediction_scheduler.py, tools/prediction_scheduler.py, tools/run_prediction_recovery.py를 수정함.
  - 현재 파일에 `JobRegistry`, `mark_pressure_killed`, `retry_reason` 등이 있어 **적용된 상태**임. bash_history에도 `git apply cgf-pressure-kill.patch`가 있음.
- `tools/run_prediction_recovery.py`의 mtime은 11-10 10:39임. 11-10 무제한 RAM 지원 변경까지 반영된 상태임.
  - 342 실행에 쓴 버전(sha `e143f917…`)은 패키지의 `source/runner_for_332.py`와 `oom-unlimited-audit/run_prediction_recovery.before.py`에 있음.
  - 현재 버전 sha는 `26278840…`. diff는 `classify(memory_limit_bytes)`, `--memory-gib 0` 허용, 무제한일 때 `--memory` 생략, `oom_killed_without_job_limit` 분류, 무제한 1회 시도뿐임.
  - `tools/prediction_scheduler.py`는 패키지 `scheduler_for_332.py`와 동일함.

### 3.8 [11-08] 병렬 admission scheduler 검증 (`parallel-check-01`, `parallel-audit-20261108`)

- 정책(`docs/cpu_parallel_admission_20261108.md`)
  - 신규 작업은 호스트 `MemTotal−MemAvailable ≤ 20GiB`이고 가용 ≥ reserve일 때만 시작함.
  - 기본 최대 2작업, 시작 간격 5초. 대기 시간은 timeout에 포함하지 않음.
  - 20GiB는 시작 기준일 뿐이며 시작 후 RAM 증가는 막지 못함.
- 검증
  - 호스트 테스트 50개 중 5개 skip. VM 50/50 통과.
  - 실제 병렬: gpt2 small fp32 01과 t5-small fp32 01을 겹쳐 실행함. 둘 다 0 byte 차이.
  - timeout과 SIGTERM 취소 테스트에서 컨테이너가 정리됨.
- audit: `check_parallel_lifecycle.py`, `cgf-parallel-cancel-*.json`, `cgf-parallel-timeout-*.json`, `parallel-lifecycle-results.json`, `github-publication.json`(runner sha `353920c0…`, scheduler `c584e94a…`. pressure-kill 이전 버전).

### 3.9 [11-08 ~ 11-10] all342 v4 실행 (342 CPU 예측의 본 실행)

#### 1차: `parallel-all342-20261108-180550`

- 명령
  ```
  cd /home/inuos/CloudGPUFit-cpu-singlepass-20261108
  python3 -u tools/run_prediction_recovery.py --all --output-root "$cgf_out" \
    --snapshot-image cloudgpufit-cpu-validation:20261108 --jobs 2 --threads 4 \
    --start-below-used-gib 20 --memory-gib 32 --host-reserve-gib 6 --timeout-seconds 21600
  ```
- runner sha `353920c0…`(pressure-kill 이전), scheduler `c584e94a…`.
- 결과: 18:05:59 시작. gpt2 small FP16 01~10 성공. 11·12 실행 중 18:30:01에 `cancellation_requested`(SIGTERM).
  - 12는 `missing_prediction_json`(stage measured_inference). 11은 결과 파일 없음.
  - manifest `runner_failed`, completed 11, success 10.
- 중단 이유: bash_history 순서상 pressure-kill 패치를 적용하고 jobs 5로 재시작하려던 것으로 추정함.
- 18:30:11, 18:32:26 재시작 시도: `unrecognized arguments: --kill-below-available-gib 4 --solo-memory-gib 36`. 패치 적용 전 runner로 실행해 실패함.

#### 2차: `parallel-all342-20261108-183250` (남은 332건)

- 명령(bash_history)
  ```
  python3 -u tools/run_prediction_recovery.py --remaining-from "$prev" --output-root "$cgf_out" \
    --snapshot-image cloudgpufit-cpu-validation:20261108 --jobs 5 --threads 4 --start-below-used-gib 20 \
    --memory-gib 32 --host-reserve-gib 6 --kill-below-available-gib 4 --poll-seconds 0.5 \
    --solo-memory-gib 36 --timeout-seconds 21600
  ```
  - `$prev` = 1차 폴더. 그 폴더의 성공 10건을 제외해 332건을 선택함.
- 설정(manifest로 확인)
  - runner sha `e143f917…`, scheduler `3e7c6413…`.
  - 작업 제한
    - jobs 5, threads 4(`--cpus 4`), 작업당 RAM 32GiB, swap 없음
    - timeout 21,600초(6시간), oom_score_adj 1000, network none
    - 소스/캐시 RO mount, `HF_HUB_OFFLINE=1`
  - admission: used ≤ 20GiB이고 available ≥ 6GiB.
  - pressure-kill: 가용 < 4GiB이면 RAM을 가장 많이 쓰는 작업 1개를 SIGKILL함. 큐가 빈 뒤 혼자 1회 재시도하며 이때 제한은 36GiB임.
  - cgroup_oom 건도 solo 한도(36) > 기본 한도(32)이므로 1회 재시도함.
- 결과
  - 18:32:59 시작, 11-10 02:59:53 종료(116,814초 ≈ 32.4시간). `complete_with_errors`.
  - 315 ok / 17 error(cgroup_oom 16, timeout 1). pressure_kills 36, solo_retries 41, leftover 컨테이너 0.
  - scheduler 이벤트: start 373, finished 332, wait 227, requeued_solo 41, memory_pressure_kill 36.
  - 재시도 결과 (본 문서 계산)
    - pressure-kill 후 재시도 36건: 성공 24, 실패 12.
    - cgroup_oom(32GiB) 후 재시도 5건: 모두 다시 실패.

#### 최종 집계 (패키지: 1차 성공 10 + 2차 332 = 342, 중복·누락 없음)

- 성공 325, 실패 17, 미실행 0. 모든 성공 건은 `prediction_version=4-native-inference-single-request`, warmup 0, measured 1.
- 시도 횟수: 성공 중 1회 301, 2회 24.
- 모델·dtype별 성공률(`model_dtype_summary.csv`)

| 모델 | fp32 | fp16 | bf16 | 성공 중앙값 시간(s) fp32/fp16/bf16 |
|---|---|---|---|---|
| gpt2 | 14/14 | 14/14 | 14/14 | 18.5 / 196.3 / 27.5 |
| gpt2-medium | 12/14 | 13/14 | 14/14 | 36.0 / 772.5 / 57.9 |
| gpt2-xl | 12/14 | 11/14 | 11/14 | 95.5 / 4096.3 / 183.6 |
| mobilenet_v2 | 11/11 | 11/11 | 11/11 | 6.1 / 11.2 / 6.1 |
| resnet-50 | 11/11 | 11/11 | 11/11 | 6.9 / 17.1 / 7.4 |
| vit-large | 5/5 | 5/5 | 5/5 | 12.0 / 31.7 / 15.1 |
| t5-small | 15/15 | 15/15 | 15/15 | 13.5 / 24.7 / 17.3 |
| t5-base | 15/15 | 15/15 | 15/15 | 21.2 / 47.6 / 31.0 |
| t5-3b | 12/15 | 13/15 | 14/15 | 86.6 / 308.9 / 175.7 |

- 최종 실패 17건. 모두 2번째 시도이며 제한 36GiB였음.
  - stage는 measured_inference. t5-3b fp32 11만 profiler_stop.
  - `max_logged_cpu_rss_GiB`는 phase 경계에서만 기록됨. 그래서 대부분 낮게 나옴(예: 1.3~11GiB). 실제 peak가 아님.

| case | 종류 | B/len/new | 최종 시도 s |
|---|---|---|---:|
| decoder_llm medium fp16 12 | cgroup_oom | 4/768/128 | 4116.7 |
| decoder_llm medium fp32 11 / 14 | cgroup_oom | 4/512/128, 1/768/256 | 39.0 / 41.8 |
| decoder_llm large fp16 11 / 14 | cgroup_oom | 4/512/128, 1/768/256 | 17933.8 / 7325.5 |
| decoder_llm large fp16 12 | **timeout(6h)** | 4/768/128 | 21604.2 |
| decoder_llm large fp32 10 / 14 | cgroup_oom | 1/256/256, 1/768/256 | 88.1 / 65.8 |
| decoder_llm large bf16 11 / 12 / 14 | cgroup_oom | | 216.8 / 238.3 / 195.5 |
| text_encdec large fp16 12 / 13 | cgroup_oom | 2/512/1/128, 2/1024/1/128 | 385.4 / 823.5 |
| text_encdec large fp32 03 / 11 / 12 | cgroup_oom(11은 profiler_stop) | 4/256/1/64, 1/256/1/256, 2/512/1/128 | 69.8 / 110.9 / 67.5 |
| text_encdec large bf16 13 | cgroup_oom | 2/1024/1/128 | 205.5 |

- 기준 조건 01 예측값(MiB, allocated / reserved, 본 문서 발췌)
  - gpt2: fp16 296.5/330, fp32 565.7/662, bf16 345.7/380
  - gpt2-medium: fp16 768.9/830, fp32 1500.8/1538, bf16 818.2/880
  - gpt2-xl: fp16 3225.0/3374, fp32 6401.0/6472, bf16 3274.5/3424
  - mobilenet: fp16 13.3/30, fp32 30.0/50, bf16 29.0/50
  - resnet-50: fp16 56.1/76, fp32 113.3/144, bf16 66.5/76
  - vit-large: fp16 589.5/606, fp32 1172.5/1180, bf16 589.5/606
  - t5-small: fp16 123.5/134, fp32 240.8/266, bf16 123.5/134
  - t5-base: fp16 455.3/482, fp32 874.6/984, bf16 455.3/482
  - t5-3b: fp16 5560.5/6066, fp32 11120.8/11136, bf16 5561.8/6086
- 성공 건 중 RAM이 높았던 것(`max_logged_cpu_rss_GiB`)
  - t5-3b fp32 02: 36.07. medium bf16 12: 34.13. xl fp16 10: 33.77. xl bf16 10: 33.40.
  - t5-3b fp16 03: 32.19. gpt2 fp32 12: 31.43.
- 가장 오래 걸린 성공 건
  - xl fp16 04: 19,738초. xl fp16 03: 10,555초. xl fp16 13: 7,582초. xl fp16 07: 5,487초.

### 3.10 [11-10] 342 결과 패키지 export — `/home/inuos/CloudGPUFit_CPU_predictions_20261110`

- 생성 스크립트: `source/export_cpu_results.py`(= `/home/inuos/cloudgpufit-comparison/cpu-export-tools/export_cpu_results.py`, 동일 확인).
  - 추론이나 AWS 작업은 실행하지 않음.
  - 실행 전 assert: 두 run의 identity, plan, revisions 동일. 342 case 정확히 매칭. 성공 건은 v4이고 warmup 0, measured 1.
- 생성 시각 2026-11-10T10:51:01+09:00. zip sha256 `e2ab0e6e…`(6,843,162 B). `export-validation.json` status verified.
- 파일
  - `README_먼저읽기.md`: 집계 범위, 비교 방법, 단위, 무결성.
  - `CPU_predictions.xlsx`: 시트 요약 / 성공325 / 전체342 / 실패17 / 모델정밀도별 / GPU별요구량(1300행 = 325×4 GPU) / 컬럼설명 / 출처.
  - `cpu_predictions_all.csv`(342), `cpu_predictions_success.csv`(325), `cpu_predictions_failures.csv`(17), `cpu_predictions.json`, `gpu_requirements.csv`, `model_dtype_summary.csv`.
  - `raw_predictions_with_requests.json`, `raw/predictions/`(prediction/input JSON 342쌍), `raw/logs/`(최초와 solo 재시도 로그), `raw/runs/`(두 run의 manifest/scheduler).
  - `prediction_provenance.json`: 원본 경로와 hash. `export_manifest.json`, `SHA256SUMS.txt`.
  - `source/`: predictor 5파일, `runner_for_332.py`, `scheduler_for_332.py`, `selected-plan.json`, `revisions.json`, `xmem-allocator-lock.json`.
  - `commands/retry_oom_unlimited.sh`: 사용자 직접 실행용으로 만든 것이며 export 시 실행하지 않음.
- CSV 주요 컬럼
  - 식별·입력: case_id, status, workload_type, model, size, dtype, batch_size, seq_len, encoder_seq_len, decoder_seq_len, max_new_tokens, resolution, mode
  - 예측값: `predicted_allocated_bytes`, `predicted_reserved_bytes`(+MiB/GiB), model_storage_bytes, resident_storage_bytes, input_storage_bytes
  - KV: self/cross_kv_storage/logical_bytes
  - trace: warmup/measured_invocations, tensor_lifetime_count, profiler_event_count
  - 시간: final_attempt_seconds, previous_attempt_seconds, recorded_attempts_seconds, attempt_count
  - 자원: max_logged_cpu_rss_GiB, max_logged_cgroup_peak_GiB, job_ram_limit_GiB
  - 오류: error_kind, error_stage, error, exit_code, oom_killed
  - 출처: prediction_version, revision, actual_input_shapes_json, source_run, cpu_threads, source_jobs, raw_prediction, raw_input, final_log, input_config_sha256
- README의 비교 지침
  - CPU allocated ↔ GPU `inference_peak_allocated_bytes`, CPU reserved ↔ GPU `inference_peak_reserved_bytes`.
  - NVML은 reserved + outside ↔ `nvml_process_sampled_peak_bytes`.
  - 실패 건은 null로 둠(0이 아님). AWS 실측과 오차 통계는 포함하지 않음.
  - 차이 요인을 명시함: CPU는 from_config 랜덤 가중치, GPU는 pretrained. CPU는 빈 allocator에서 재생, GPU는 앞 단계 캐시를 이어받음.
- 같은 predictions 342쌍이 `/home/inuos/cloudgpufit-comparison/native-v4-vs-aws-20261110/predictions`(684파일)에도 복사돼 있음. `prediction-provenance.json`, `example-actual-memory.json`(L4 memory 모드 예시 1건) 포함.

### 3.11 [11-10] OOM 16건 무제한 RAM 재시도

- 무제한 모드 runner 변경 내용(§3.7.1)
  - `--memory-gib 0`이면 `--jobs 1 --kill-below-available-gib 0 --solo-memory-gib 0`이 필요함.
  - OOM은 `oom_killed_without_job_limit`으로 분류함. 재시도하지 않음.
- assistant 시작분: `oom-unlimited-20261110-103935`, audit는 `oom-unlimited-audit-20261110-103935`.
  - canary로 memory.max=max, swap.max=max, oom_score_adj 1000을 확인함.
  - 10:41:05에 medium fp16 12를 시작했으나 90.8초 후 `cancelled`(exit 130).
  - `stopped-at-user-request.json`: "User requested commands only; stop assistant-started retry".
  - audit에 `change.json`(runner before/after sha, 16 case 명령), `selection.json`, `previous-output-checksums.json`, `run.sh`가 있음.
- 사용자 수동 실행: `oom-unlimited-manual-20261110-105344`
  - 스크립트: `/home/inuos/cloudgpufit-predictions/retry_oom_unlimited_20261110.sh`
    - 패키지판과 차이: `--host-reserve-gib 0`, `--start-below-used-gib <MemTotal GiB 올림>`.
    - 2차 run에서 cgroup_oom 16건을 자동으로 고르고 `--jobs 1 --memory-gib 0 …`으로 실행함.
  - 10:53:51 시작, 18:36:10 cancellation(27,740초). manifest `runner_failed`, completed 5.

| case | 결과 | allocated B | reserved B | s |
|---|---|---:|---:|---:|
| medium fp16 12 | **ok** | 1,502,986,240 | 1,889,533,952 | 4042.0 |
| medium fp32 11 | oom_killed_without_job_limit(host OOM) | — | — | 59.4 |
| medium fp32 14 | **ok** | 1,829,330,944 | 2,011,168,768 | 105.0 |
| large fp16 11 | oom_killed_without_job_limit | — | — | 18601.0(약 5.2h) |
| large fp16 14 | missing_prediction_json(취소로 중단, measured_inference 중) | — | — | 4932.1 |

- 이 2건 성공은 342 패키지에 **포함되지 않음**(패키지는 10:51 생성).
- 18:36 취소 사유는 미확인. reserved diag 작업과 시기가 겹침(추정).
- bash_history에 `tmux kill-session -t retry`, `cgf-diag` 관련 kill 기록이 있음.

### 3.12 [11-10] AWS 실측 자료 수집 — `/home/inuos/cloudgpufit-comparison`

- `aws-actual-20261110/`: 342조건 GPU 실측 tarball 6개. Notion 첨부에서 받음. 파일별 sha256은 `source-manifest.json`에 있음.
  - `g4dn-2xlarge-all342-20261102.tar.gz`(T4)
  - `g5-2xlarge-all342-20261102.tar.gz`(A10G)
  - `g6-xlarge-all342-20261102.tar.gz`, `g6-2xlarge-all342-20261102.tar.gz`, `g6-4xlarge-all342-20261103.tar.gz`(L4)
  - `g6e-2xlarge-all342-20261103.tar.gz`(L40S)
- 이 폴더에는 압축 해제본이나 비교 결과가 없음.
  - L4 g6.2xlarge tarball은 reserved diag의 `inputs/aws_g6_2xlarge/`에 풀려 비교에 쓰임(`provenance/ENV_AND_PATH_REPORT.md` 기록).
  - GPU worker `bench_worker_v3.py` sha `4018cd8e…`.

### 3.13 [11-12] `/home/inuos/cloudgpufit-review` (참고, 342 계열 아님)

- INUOS → INUGPU 교차 검토. FP16 395조건(RTX 3090, FP16-680 계열)의 CPU manifest(`/home/inuos/cgf-fp16-680-20261111/handoff_codex/cpu_manifest.json`)와 GPU manifest를 비교함.
- 판정: 대상 조건과 경계 의미는 같음. 실행·요청 동등성은 입증되지 않음.
- 질문 INUOS-Q001~Q007 답변 대기. `received_reply: false`.
- 파일: `review_INUOS.md`, `comparison.md`, `comparison_details.json`, `paired_metrics.csv`, `case_crosswalk.csv`, `skip_events.csv`, `gpu_manifest.json`, `review_state.json`, `review_artifacts.sha256`.
- 상세는 FP16-680 문서 범위임.

---

## 4. 코드 설명

- 기준
  - 342 실행 버전 = singlepass worktree `/home/inuos/CloudGPUFit-cpu-singlepass-20261108`.
  - legacy는 `/home/inuos/CloudGPUFit` HEAD 7f822d7.
- 아래 줄번호는 실제 파일 기준임.

### 4.1 `predict_vram.py` (singlepass, 345줄, sha `7fd049d5…`)

- 역할
  - 조건 1개의 CPU 예측 엔트리포인트. 컨테이너 안 `/recovery/predict_vram.py`로 실행됨.
  - stdout 마지막에 결과 JSON 1개를 출력함. stderr에는 `CGF_PROGRESS` phase 로그를 출력함.
- CLI
  - 기본 형태: `python -u predict_vram.py --json -`(stdin JSON). 개별 플래그도 받음.
  - 플래그와 기본값
    - `--model`, `--workload_type {decoder_llm,text_encdec,vision_cls}`, `--dtype {fp32,fp16,bf16}`(기본 fp16)
    - `--batch_size`(1), `--seq_len`(128), `--encoder_seq_len`(512), `--decoder_seq_len`(128), `--resolution`(None→config image_size→224)
    - `--threads`(4), `--gpu_name`, `--mode {prefill,generate}`(prefill), `--max_new_tokens`(256), `--gpu_memory_gb`(80)
    - `--warmup_invocations`(0), `--measured_invocations`(1), `--trace_output`, `--allocator_history`
  - JSON 키는 argparse 결과 위에 덮어씀. `model_config`(HF config.json dict)도 JSON으로 전달됨.
- runner가 보내는 요청(`run_prediction_suite.request`)
  - `{model, workload_type, model_config, dtype, **params, mode: vision이면 prefill 아니면 generate, gpu_memory_gb: 80, threads}`

| 줄 | 동작 |
|---|---|
| 14-27 | import(`traceback`, `inference_trace_runtime.phase`, `vram_policy.memory_result`). `DTYPE_BYTES`, `PROFILE_ITERATIONS=2`/`SAMPLES_PER_LOADER=4`(legacy 상수. 실제로는 첫 배치만 씀) |
| 33-90 | 합성 Dataset. LLM: `Generator().manual_seed(idx)`, `randint(0, min(vocab,30000), (seq_len,))` + attention_mask 1. EncDec: enc/dec ids, `dec[0]=decoder_start_token_id`. Vision: `randn(C,res,res)`를 모델 dtype으로 변환 |
| 93-111 | `build_loader`/`build_encdec_loader`: n = batch×4, shuffle False |
| 114-118 | `WORKLOAD_MODEL_CLASS`: decoder_llm→AutoModelForCausalLM, text_encdec→AutoModelForSeq2SeqLM, vision_cls→AutoModelForImageClassification |
| 121-171 | `load_model`: `CONFIG_MAPPING[model_type].from_dict(model_config)`(네트워크 없음), `cfg.torch_dtype=target`, `_attn_implementation="eager"`, `model_class.from_config(cfg, torch_dtype, attn_implementation="eager")`, `eval()`. torchvision 별칭은 거부함 |
| 180-190 | `param_bytes`, `storage_sizes`: `(data_ptr, nbytes)`로 고유 storage의 nbytes 목록을 만듦(공유 storage 중복 제거) |
| 193-196 | 고정 경고 3개 |
| 200-213 | `run`: seed 20261006, `set_num_threads(threads)`, 입력 검증 |
| 215-223 | `phase('model_initialization')` 안에서 모델 생성. `model_storage_bytes` 계산 |
| 225-252 | workload별 loader. T5는 decoder_start→pad→0 순으로 선택함 |
| 254-262 | 경고. Vision 해상도가 config와 다르면 경고 추가 |
| 264-266 | `next(iter(loader))` 첫 배치만 입력으로 씀. vision은 `{"pixel_values": x}`. `actual_input_shapes` 기록 |
| 267-272 | `cpu_tensor_trace.trace_and_estimate(model, inputs, mode, max_new_tokens, storage_sizes, gpu_memory_gb, trace_output, allocator_history, warmup_invocations, measured_invocations)` |
| 274-275 | `vram_policy.memory_result(allocated, reserved, dtype, gpu_name)` |
| 277-298 | 결과 dict 반환 |
| 301-331 | argparse, `--json`을 덮어씀 |
| 333-341 | 예외 시 `{"status":"error","error","stage":LAST_PHASE,"traceback","input"}`. JSON 출력. exit 0(ok) 또는 1 |

- 출력 스키마(ok)
  - 기본: `status`, `input{model,type,batch_size,seq_len,encoder_seq_len,decoder_seq_len,resolution,dtype,mode,max_new_tokens}`
  - `vram_policy` 결과: `prediction_version`, `predicted_allocated_bytes`, `predicted_reserved_bytes`, `required_vram_bytes`, `required_vram_gb`, `required_vram_scope`, `target_gpu_estimate`, `gpu_requirements{GPU:{…}}`, `breakdown_bytes{predicted_allocated_native_dtype, predicted_reserved_native_dtype, empirical_tensor_correction:0, fixed_margin:0, outside_gpu_estimate}`
  - 나머지: `model_storage_bytes`, `trace_summary{…}`, `actual_input_shapes`, `inference_path`(full_generate_with_cache 또는 forward), `dtype_policy: native_cpu_no_rescale`, `environment{torch,transformers,cpu_threads,attention:eager,seed}`, `elapsed_seconds`, `warnings`
- 7f822d7(HEAD) 버전(332줄)과 다른 점: §3.7.1 참조. HEAD는 trace 호출 줄이 265이고 phase/traceback이 없음.

### 4.2 `cpu_tensor_trace.py` (singlepass, 150줄, sha `e16048e3…`)

- 역할: 추론 구간의 CPU 텐서 할당/해제를 profiler로 기록함. 수명 쌍을 만들고, 상주 storage를 합성해 xMem allocator로 재생함.

| 줄 | 동작 |
|---|---|
| 11, 14 | `phase/read_memory_events/simulate` import. `REGION="cloudgpufit_inference"` |
| 17-20 | `selected_lifetimes`: `cat=user_annotation, name=REGION` 이벤트로 측정 구간 `(ts, ts+dur)` 목록을 만듦. 개수가 expected와 다르면 오류 |
| 22-26 | 이벤트를 `ts`로 stable sort. `cpu_instant_event/[memory]`, `Device Type==0`(CPU), `Bytes≠0`만 사용 |
| 27-30 | 할당(Bytes>0)은 측정 구간 안 것만 live에 등록. 같은 주소가 해제 없이 재사용되면 오류 |
| 31-34 | 해제(Bytes<0)는 구간 밖이어도 live에 있으면 매칭. 크기 불일치는 오류 |
| 35-37 | 해제되지 않은 할당은 `(alloc, None)`으로 끝까지 유지. 비어 있으면 오류 |
| 40-46 | `trace_and_estimate` 시그니처. warmup ≥ 0, measured ≥ 1 검증 |
| 51-64 | `observe_cache` forward hook(generate 모드만): `past_key_values`의 self(layer[:2])와 cross(layer[2:4]) KV의 storage bytes와 logical bytes 최대값을 기록함. **진단용이며 예측값에 더하지 않음** |
| 66-80 | `invoke`: generate는 `do_sample=False, num_beams=1, use_cache=True, min_new_tokens=max_new_tokens=N, pad=pad or eos`. 출력 길이가 prefix+N이 아니면 오류. vision은 `model(**inputs)`, 그 외는 `model(**inputs, use_cache=False)` |
| 82-106 | `capture`: `inference_mode` 안에서 warmup N회(phase warmup, 추적 밖) → `torch.profiler.profile(CPU, profile_memory=True, record_shapes=False, with_stack=False)` `start()` → `record_function(REGION)`로 measured회 invoke → `stop()` → `export_chrome_trace`. profiler는 함수 로컬이라 파싱 전에 참조가 사라짐 |
| 108-112 | 임시 디렉터리(또는 `--trace_output`)에 trace를 저장하고 `read_memory_events`로 스트리밍 파싱함 |
| 113-115 | phase lifetime_matching. `del events` |
| 116-120 | 수명 → `MemoryBlock(CpuInstantNode(alloc))`, `set_free_node(free)` |
| 121-131 | 상주 storage 합성. 모델 parameter+buffer와 입력의 고유 storage마다 가짜 `[memory]` 할당 이벤트를 만듦(`ts = 첫 구간 시작 − 1`, 주소 −1, −2, …). 해제 없음. dtype 재환산 없음 |
| 132 | `blocks.sort(key=alloc_time)` |
| 133-135 | phase allocator_simulation. `simulate(blocks, gpu_memory_gb, Config(save2tmp=True, run_id=uuid), record_history)` |
| 136-139 | allocated = `max(_trace.max_usage_changes)`, reserved = `max(_trace.max_segment_changes)`. allocated가 상주 합보다 작으면 오류 |
| 140-150 | 반환: `memory{tensor, segment}`, `trace_summary{measured_invocations, warmup_invocations, request_policy(cold_single_request 또는 explicit_repeated_requests), allocator_initial_state:'empty_plus_resident_model_and_inputs', self/cross_kv_storage/logical_bytes, total_profiler_events, allocator_history_recorded, tensor_lifetime_count, resident_storage_bytes, input_storage_bytes, cpu_device_only, method:'native_inference_lifetimes_into_xmem_allocator', cross_kv_policy:'traced_with_full_generate_no_analytic_addition'}` |

- 의미상 주의
  - 초기 allocator는 비어 있고 모델·입력만 상주함. GPU의 로딩·검증 forward·warmup 캐시는 없음. reserved diag의 핵심 주제가 이것임.
  - 용량은 80GiB임. 넘으면 simulated_gpu_capacity_exceeded로 실패함.
  - recovery worktree 버전(1+3 고정)과의 diff: `selected_lifetimes`가 3구간 고정, warmup 1회 고정, `range(3)`.

### 4.3 `inference_trace_runtime.py` (193줄, sha `6c999a90…`, recovery와 singlepass 동일)

- 역할: 스트리밍 Chrome trace 파서, phase/자원 로그, 이력을 생략하는 allocator 실행, simulation 오류 분류.

| 줄 | 동작 |
|---|---|
| 15 | `LAST_PHASE` 전역(오류 stage 보고용) |
| 18-31 | `resource_snapshot`: pid, `ru_maxrss`, `/proc/self/status` VmRSS, `/sys/fs/cgroup/memory.current`, `memory.peak` |
| 34-36 | `progress`: stderr에 `CGF_PROGRESS {json}` 출력 |
| 39-51 | `phase(name)` context manager. phase_start / phase_end / phase_error와 경과시간 |
| 54-96 | `_JSONStream`: 64KiB 청크 버퍼와 `JSONDecoder.raw_decode`. 청크 경계에 걸린 부분 scalar는 받지 않음 |
| 99-137 | `iter_trace_events`: 최상위 객체에서 `traceEvents` 배열 원소를 파일 순서대로 yield함. 중복, 누락, trailing data는 오류 |
| 140-150 | `read_memory_events`: REGION user_annotation과 CPU(Device Type 0) Bytes≠0 `[memory]` 이벤트만 보관함. `(selected, total)` 반환 |
| 153-161 | `SummaryTrace(Trace)`: `_record`만 no-op으로 바꿈(세그먼트·usage counter는 upstream 유지) |
| 163-178 | `ReportingAllocator(CachingAllocator)`: malloc/free 예외를 `failure`에 저장 |
| 180-185 | `AllocatorSim` 생성 후 `_allocator`를 ReportingAllocator로 교체. history를 끄면 `_trace=SummaryTrace(_segments)` |
| 186-193 | `result.oom` 분류: `MemoryError('OOM when allocating …')` → `simulated_gpu_capacity_exceeded`. 그 외 예외 → `allocator_simulation_failed: <type>`. 원인 없음 → "not a confirmed capacity OOM" |

- phase 이름
  - model_initialization, warmup, measured_inference_and_profiler_stop(바깥), measured_inference, profiler_stop
  - trace_export, trace_parse, lifetime_matching, allocator_simulation
- 알려진 한계(문서 기록)
  - O(E) 이벤트와 allocator 맵은 여전히 보관함.
  - profiler가 켜진 추론 구간 안의 RAM 폭증은 이 파일로 해결되지 않음.

### 4.4 `vram_policy.py` (86줄, singlepass는 VERSION만 다름) + `gpu_overheads.json`

| 줄 | 동작 |
|---|---|
| 8-11 | `VERSION`(v3 `3-native-inference`, v4 `4-native-inference-single-request`), 보정 파일 경로, GPU 별칭 |
| 14-17 | `nonnegative_bytes`: 유한한 음이 아닌 값을 ceil |
| 20-48 | `process_estimate`: GPU 프로필이 없으면 `uncalibrated_gpu`, dtype 미지원이면 `unsupported_dtype`. 있으면 `estimated_nvml = reserved + outside_median`, `required = reserved + outside_max_training_sample` |
| 51-86 | `memory_result`: 4개 GPU 모두에 대해 `gpu_requirements`를 계산함. 대상 GPU가 없으면 top-level `required_vram_bytes`는 보정된 GPU 중 최대값(`maximum_among_calibrated_gpus`)임. breakdown의 보정·margin은 0 |

- 342 실행에서는 `gpu_name`이 없음. 그래서 top-level required는 4 GPU 중 최대값(대개 A10G/3090의 outside 상한)이 됨. 비교에는 `predicted_*_bytes`를 써야 함.

### 4.5 `tools/run_prediction_suite.py` (HEAD 7f822d7, 177줄): legacy 공유 컨테이너 실행기

- CLI
  - `--output-root`(필수), `--plan`, `--revisions`, `--case-set {baseline9,all}`
  - `--threads`(4), `--jobs`(1), `--ram-budget-gb`(36), `--prepare`, `--preflight-only`

| 줄 | 동작 |
|---|---|
| 21-34 | `save`(tmp 후 replace), `prediction_json`(로그에서 줄 시작 `{` 위치를 역순으로 raw_decode) |
| 37-41 | `request`(gpu_memory_gb 80, mode 결정) |
| 44-51 | `select_cases`: core 342 검증. baseline9 = fp16 & `__01` |
| 54-58 | `ram_reservation_gb`(스케줄링 추정치) |
| 61-76 | `bounded_results`: ThreadPool. 예약 합이 budget 이하일 때 제출 |
| 92-110 | 출력 lock, identity(소스 4개·plan·revisions sha, threads). identity가 다르면 거부. 컨테이너 실행 여부 확인, repo `.cpu-prediction.lock`, 실행 중 예측 확인 |
| 111-114 | `--prepare`: `setup_manager._apply_patch_if_needed`, `_sync_predictor`(소스를 `/work/xMem`에 복사) |
| 117-130 | 환경 고정 검사(torch 2.6.0+cu124/CUDA 12.4/TF 4.52.4/Py 3.11.11/GPU 없음), 컨테이너 소스 sha, xMem commit, allocator lock hash |
| 134-141 | manifest, invocations.jsonl, 모델 config(`hf_hub_download`, pinned revision) |
| 142-162 | `execute`: 이미 ok면 REUSE. 아니면 input.json 저장 → `docker exec … predict_vram.py --json -` → 로그에서 JSON 파싱 → 저장. 예외 시 error JSON(종료 코드보다 JSON 파싱을 먼저 함) |
| 163-174 | 완료 순 집계. manifest/summary.csv |

- 출력: `<case>.input.json{case_id,revision,request}`, `<case>.log`, `<case>.prediction.json`(예측기 결과 + case_id, revision), `manifest.json{identity,status,case_set,total_cases,jobs,ram_budget_gb,environment,image_id,allocator_lock,git_commit,completed_cases,success_count,elapsed_seconds}`, `summary.csv`, `invocations.jsonl`.
- singlepass 버전은 `--allow-unbounded-legacy-run`이 없으면 실행을 막음(+4줄). identity에 inference_trace_runtime 포함.

### 4.6 `tools/run_prediction_recovery.py` (singlepass 현재본 500줄, sha `26278840…`. 342 실행본 `e143f917…`)

- 역할
  - 선택한 조건을 조건마다 일회용 Docker 컨테이너(PID1 = predictor)에서 실행함.
  - 컨테이너마다 RAM/CPU 제한, timeout, 취소, OOM 분류, 정확 비교를 적용함. scheduler로 병렬 admission과 pressure-kill을 처리함.
- CLI
  - 필수: `--output-root`(새 경로만 허용). `--all` / `--case-id`(반복) / `--remaining-from` 중 하나
  - 대상 지정: `--container`(xmem-cpu), `--snapshot-image`, `--compare-with`
  - 실행 자원: `--threads`(4), `--jobs`(2), `--memory-gib`(32. 0이면 무제한), `--solo-memory-gib`(기본 = memory)
  - admission과 kill: `--start-below-used-gib`(20), `--poll-seconds`(1), `--launch-interval-seconds`(5), `--host-reserve-gib`(6), `--kill-below-available-gib`(4. 0이면 끔)
  - 시간: `--timeout-seconds`(3600)

| 줄 | 동작 |
|---|---|
| 20-25 | suite의 `ROOT, CONFIG, prediction_json, request, save, select_cases` 재사용. scheduler import. SOURCES 5개 |
| 40-49 | `last_phase`: 로그의 마지막 CGF_PROGRESS phase |
| 52-83 | `classify`: cancelled → timeout → OOMKilled(제한 0이면 `oom_killed_without_job_limit`, 아니면 `cgroup_oom`) → JSON 없음(`missing_prediction_json` 또는 `process_exit_without_json`) → JSON 있으면 종료 코드를 우선함(`process_exit`, `predictor_error`) |
| 86-96 | `stop_job`(docker stop -t3, 실패 시 kill), `kill_job`(즉시 SIGKILL) |
| 99-111 | `retry_reason`: 1차 실패만 대상. pressure-kill, cgroup_oom(solo 한도가 더 클 때), 137(SIGKILL) |
| 114-194 | `run_job`: `docker create` → inspect(제한, mount) → `docker start -a -i` → host PID/cgroup 수집 → stdin으로 payload 전달 → 1초 단위로 취소·timeout 확인 → 최종 State → classify → finally에서 `docker rm -f` |
| 197-212 | `job_command`: `--entrypoint python --network none --oom-score-adj 1000 --cpus T`, `/work`·HF cache·`/recovery`(=worktree) RO mount, `--memory/--memory-swap`(제한이 있을 때), env `OMP/MKL_NUM_THREADS=T, OPENBLAS=1, PYTHONDONTWRITEBYTECODE=1, HF_HUB_OFFLINE=1, PYTHONPATH=/recovery:/work/xMem`, `-w /work/xMem image -u /recovery/predict_vram.py --json -` |
| 215-230 | `choose_cases`(`--case-id` 또는 `--remaining-from`의 비성공 조건) |
| 233-250 | `compare_reference`: request·revision이 같은지 확인하고 allocated/reserved byte 차이 계산. equal / mismatch / incompatible_input |
| 276-286 | 인자 검증. 무제한이면 jobs 1 + kill 0 + solo 0. kill < reserve |
| 287-297 | 출력 경로가 새 경로인지 확인, plan/revisions, 조건 선택, `/tmp/cloudgpufit-cpu-recovery-<uid>.lock` |
| 299-324 | 원 컨테이너 Running 확인, 실행 중 예측이 없는지 확인, bind mount 경로, allocator lock hash, 환경 고정, snapshot 환경이 원본과 같은지 확인, 모델 config 수집 |
| 325-356 | 바이트 변환, 이미지 token, manifest 초기화(git, identity sha, plan/rev sha, runner/scheduler sha, 제한, 정책, selection) |
| 362-370 | snapshot 재사용 또는 `docker commit --pause=false` 후 `image inspect`로 ID 획득 |
| 375-383 | `job_ram`: host PID → cgroup anon RAM |
| 385-413 | `execute`: input.json(첫 시도), 컨테이너 이름 `cgf-cpu-<token12>-<n>[-aN]`, 로그 `<cid>[.attemptN].log`, registry 등록/해제, pressure-kill 표시, 메타 추가(`attempt`, `memory_limit_bytes`, `runner_elapsed_seconds`) |
| 415-426 | `retry_solo`: 무제한이면 재시도 없음. 1차 결과 요약을 `previous_attempt`로 보관 |
| 428-444 | `record`: reference 비교, prediction.json 저장, manifest 갱신. mismatch면 False 반환(신규 시작 중단) |
| 446-465 | `notify`: scheduler.jsonl 기록, WAIT/KILL/REQUEUE 출력 |
| 467-471 | `run_cases(...)` 호출 |
| 475-492 | finally: 남은 `cgf-cpu-<token>` 컨테이너 강제 제거, 임시 snapshot 삭제. status(interrupted / validation_mismatch / complete / complete_with_errors / runner_failed). 반환 130/2/0/1 |
| 495-500 | SIGTERM/SIGHUP → KeyboardInterrupt |

- 출력
  - `<case>.input.json{case_id,revision,request}`
  - `<case>.log`, `<case>.attempt2.log`
  - `<case>.prediction.json`: 예측기 결과 + `exit_code`, `container_exit_code`, `oom_killed`, `stage`, `container{container_id,container_name,memory_limit_bytes,memory_swap_bytes,mounts,nano_cpus,network_mode,host_pid,host_cgroup}`, `container_state`, `case_id`, `revision`, `attempt`, `memory_limit_bytes`, `runner_elapsed_seconds`, `reference_comparison`, (재시도 시) `previous_attempt`. 오류면 `error_kind`, `error`
  - `manifest.json`: 위 필드 + `pressure_kills`, `solo_retries`, `snapshot_image_id`, `last_admission_memory`, `running_cases`, `pending_cases`, `leftover_containers_removed`
  - `scheduler.jsonl`: start/wait/finished/memory_pressure_kill/requeued_solo/cancellation_requested와 메모리 상태
- recovery worktree 버전(342줄): 같은 구조의 순차 전용 버전. jobs/scheduler/pressure-kill/solo/무제한 옵션이 없음. CLI는 `--case-id|--remaining-from`, `--snapshot-image`, `--compare-with`, `--threads`, `--memory-gib`, `--host-reserve-gib`, `--timeout-seconds`.

### 4.7 `tools/prediction_scheduler.py` (219줄, sha `3e7c6413…` = 패키지 scheduler_for_332)

| 줄 | 동작 |
|---|---|
| 10-18 | `host_memory`: `/proc/meminfo` MemTotal/MemAvailable, used = total − available |
| 21-22 | `admission_allowed`: used ≤ threshold이고 available ≥ reserve |
| 25-65 | `_cgroup_dirs`, `job_ram_bytes`: cgroup v2 `memory.stat anon`(v1이면 total_rss/rss). 없으면 memory.current, 그다음 VmRSS. 프로세스가 없으면 None |
| 68-97 | `JobRegistry`: 실행 중 case → 컨테이너 이름, host PID. pressure-kill 표시를 1회 소비 |
| 100-110 | `choose_victim`: 측정 가능한 작업 중 RAM 최대 |
| 113-219 | `run_cases`: 메인 루프. 완료 작업 수거 → solo 재시도 판단(case당 최대 1회, 정상 큐가 비고 실행 중 작업이 없을 때만) → record가 False면 취소 → guard(available < kill 기준이면 victim SIGKILL, 30초 grace 동안 한 번에 1개) → admission(메모리, 시작 간격) → `wait(FIRST_COMPLETED, poll)`. 중단 시 취소 전파, `pool.shutdown(wait=True)` |

### 4.8 기타 파일

- `tools/run_prediction_baseline9.py`
  - c6a1125(180줄): 10-26 v2 실행기. 핵심 흐름은 §3.1과 같음.
    - 17-27: CASES 고정
    - 51-61: request
    - 85-108: margin 검사, patch reverse-check, docker cp, hash 확인
    - 109-148: 조건 루프
    - 155-176: summary/tar
  - HEAD(9줄): suite를 baseline9로 호출하는 래퍼.
- `predict_vram.py` c6a1125(418줄, v2)
  - 24-35: 상수(DTYPE_SCALE, UNDERESTIMATE 32MiB, iterations 2)
  - 189-215: xMem ModelTrainer 프로파일
  - 218-236: XMem.estimate
  - 239-250: KV 해석식
  - 263-380: run(환산·보정·KV 합산)
- `orchestrator/setup_manager.py`
  - `IMAGE`(20줄), `XMEM_COMMIT`(23줄)
  - `_create_container`(145~): docker pull/run sleep infinity
  - `_sync_predictor`(170~): 예측 소스를 `/work/xMem`에 원자적 복사
  - `_apply_patch_if_needed`/`_apply_one_patch`(229~): core_inference.patch
- `orchestrator/xmem_runner.py`: 웹 orchestrator 경로. `/work/xMem/predict_vram.py`를 컨테이너 안에서 호출하고 결과를 분류함. 342 실험에는 쓰지 않음.
- `tools/compare_vram_predictions.py`(a95710e): predictions 폴더와 GPU tarball을 받아 allocated/reserved/보정 NVML을 비교함(MAPE, 중앙 절대오차, 최대오차, 과소예측 비율). 실패 건은 남김. 이 VM에서 실행한 흔적은 미확인.
- `tools/calibrate_gpu_overheads.py`: outside 보정표 재현 도구.
- `tools/smoke_native_inference.py`: 작은 모델 5종 × 3dtype 통합 검사(15건).
- `source/export_cpu_results.py`(272줄): 두 run을 병합해 CSV/XLSX/JSON/zip을 만듦. 8-12줄에 경로 상수가 하드코딩됨(BASE/CODE/OUT/SOURCES). main은 94줄부터.
- `/home/inuos/cloudgpufit-predictions/retry_oom_unlimited_20261110.sh`: 2차 run에서 cgroup_oom 16건을 자동으로 고르고 무제한 RAM, jobs 1로 순차 실행함.

---

## 5. 이후 작업과의 연결

- 342 CPU 예측 패키지 → AWS L4 비교 → reserved diag
  - reserved diag는 `CloudGPUFit_CPU_predictions_20261110`을 CPU 운영 기준선으로 씀.
    - `src/cgf_diag/build_manifest.py`의 `PKG=.../source`, "4-native-inference-single-request" 사용.
    - `results/population_l4_vs_cpu/pop2.py`가 `cpu_predictions_success.csv`를 읽음.
  - predictor 소스 `/home/inuos/CloudGPUFit-cpu-singlepass-20261108`를 RO mount해 재사용함. 실행 이미지도 같은 `cloudgpufit-cpu-validation:20261108`임.
  - GPU 쪽은 `cloudgpufit-comparison/aws-actual-20261110/g6-2xlarge-all342-20261102.tar.gz`(L4)를 풀어 씀.
  - reserved diag 인계 문서(`handoff/REDESIGN_EXPERIMENT_PROMPT.md`)에 정리된 비교 결과(그 실험 범위)
    - L4 325조건 전부에서 memory 모드 측정 peak = 측정 시작 시 reserved였음. 즉 GPU 측정값은 앞 단계(로딩, 검증 forward, performance 모드) 캐시에 지배됨.
    - LLM/T5의 큰 reserved 오차(20% 이상)는 전부 GPU > CPU였고 allocated는 대부분 일치함.
    - 이 342 실험의 설계 특성 때문에 생긴 차이임: CPU는 빈 allocator + 상주 storage에서 1회 재생함.
  - 이 문서 범위의 결과 중 diag 해석에 필요한 전제
    1. CPU 값은 `cold_single_request` 정의임(warmup 0, measured 1, 초기 상태 = 빈 allocator + 상주 모델·입력).
    2. 가중치는 from_config 랜덤값이고 입력 난수 규칙이 GPU와 다름.
    3. 실패 17건은 null임.
    4. v3(1+3) 244건과 v4(0+1)는 12건에서 byte가 다름(본 문서 계산).
- FP16-680(11-11)과의 연결
  - 메모리 노트 기준으로 FP16-680은 이 singlepass predictor의 trace parser와 diag replayer를 재사용함.
  - 이 문서에서 미해결로 남은 두 문제를 그 실험이 다룸.
    - "profiler가 켜진 추론 구간의 RAM 폭증": USER_SCOPE profiler로 RSS 약 10배 감소.
    - "CPU FP16 느림": Cascade Lake에 FP16 hw 없음.
  - `cloudgpufit-review`(11-12)는 그 계열 결과에 대한 교차 검토임.

---

## 6. 미확인·주의 사항

- 10-26 baseline9 1차 실패(exit 127) 후 2분 사이에 한 조치: 미확인. 컨테이너에 git을 설치했다고 추정함.
- native-v3-t24 중단 사유: 미확인.
- native-v3-t4-j6 manifest의 `status: running`은 갱신되지 않은 값임. 실제로는 11-05 13:48께 수동 중단됨.
- 1차 병렬 run(18:05) 중단 사유: pressure-kill 패치 적용을 위한 것으로 추정함(bash_history 순서 근거).
- singlepass와 parallel 변경의 GitHub 반영(commit 6ca425c, c153bb2): audit 기록만 있음. 로컬에는 객체가 없어 미확인.
  - pressure-kill과 무제한 변경은 미커밋·미반영 상태임. 342 실행 소스의 기준은 git commit이 아니라 **identity hash**임.
- 실패 17건의 실제 peak RAM: 미확인. phase 경계 로그만 있음.
- 11-10 무제한 수동 run의 18:36 취소 사유: 미확인.
- 무제한 run에서 성공한 2건은 패키지에 없음.
- AWS 비교 수치(오차 통계): 이 범위의 폴더에는 없음. reserved diag 결과물에 있음.
- `cloudgpufit-comparison`에는 tarball만 있고 압축 해제본이 없음. tarball 내부는 이번에 확인하지 않음.
- 작성 중 `git status`를 실행해 worktree index의 stat 캐시가 갱신됐을 수 있음(내용 변경 없음). 그 밖의 파일 수정이나 실행은 하지 않음.

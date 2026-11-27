# 07. 코드 · 실행 환경 계보 — 무엇이 언제, 왜, 어떻게 바뀌었나 (INUGPU 중심, INUOS 요약 포함, 2026-11-18 기준)

- 이 문서의 독자: 보고서를 쓰는 사람이나 AI. "측정 코드가 처음엔 어땠고, 무엇 때문에 어떻게 바뀌었는지"를 한 곳에서 보게 한다.
- 원칙 (사용자 지시, 처음부터 끝까지 유지)
  - 생산 측정·예측 코드와 과거 결과는 덮어쓰지 않는다.
  - 바꿀 때는 새 파일·복사본·래퍼를 만들고, 원본은 `*.v<날짜>` 또는 `*.before-*`로 보존한다.
  - 실패 조건을 몰래 다른 배치나 dtype으로 바꾸지 않는다.
- 표기: sha는 sha256 앞 8자리. 파일 경로는 레포 기준(`measurement/local-gpu-vm/…`, `prediction/local-cpu-vm/…`)

---

## 1. 큰 흐름 (측정 설계가 바뀐 이유)

| 단계 | 기간 | 측정 코드 | 무엇이 문제였고 왜 바꿨나 | 문서 |
|---|---|---|---|---|
| ① 벤치 하네스 | 10-06 ~ 10-27 | `measurement/local-gpu-vm/01-cloudgpufit-bench/` v1 → v3 | 342조건 × 5모드(performance/memory/…)를 한 프로세스에서 이어서 실행. v1이 17.8시간 걸려서 v3(모델 공유 + 격리 + 반복 축소)로 줄임 (3.7시간) | `docs/01` |
| ② outside 연구 | 11-03 ~ 11-10 | `measurement/local-gpu-vm/02-outside-analysis/outside-lab/` (`probe.py`, `container_run.py`) | 예측 VRAM = reserved + outside 중 outside를 따로 정의하고 측정. probe에 GPU UUID 정규화를 추가했고, 이전판은 `probe.py.before-uuid-fix`로 보존 | `docs/02` |
| ③ reserved 오차 원인 분리 | 11-10 ~ 11-11 | `measurement/local-gpu-vm/04-reserved-diag/collector/` (INUOS 요청서 기반 수집기, 변경점 `COLLECTOR_CHANGES.md`) | CPU 예측(342)과 AWS L4 실측의 reserved 차이가 큼. 원인은 **실행 이력**이었다. 이전 측정은 같은 프로세스에서 검증 forward·performance 모드를 거친 뒤 memory를 잼 → 캐시가 이미 커져 있었음 | `docs/03`, `prediction/local-cpu-vm/docs/02` |
| ④ **VRAM 전용 측정 재설계** | 11-11 ~ | `measurement/local-gpu-vm/05-fp16-vram/src/vram_worker.py` + `src/run_conditions.py` | ③의 결론으로 "조건마다 새 컨테이너, 로드 → warmup 3 → measure 5, 검증 forward 없음, 캐시 유지"로 재설계. 이후 모든 GPU 정답값의 기준 | `docs/04` |
| ⑤ attention 기본값 전환 | 11-11 | 워커에 `--attn` 추가 (`1e86ee7e` → `d06f0f15`) | v1은 eager 고정. 실제 사용 환경(transformers 기본 = sdpa)과 달라서 파일럿 v2부터 기본값 사용. Qwen FP16 eager는 출력 NaN이라는 문제도 발견 | `docs/04` §2.2 ~ 2.3 |
| ⑥ 조건표 L1024 → L256 | 11-15 | 조건표 생성기 `generate_main_l256.py` | 1024 조건이 너무 오래 걸려서 사용자가 256으로 변경 결정 | `docs/04` §2.10 |
| ⑦ 분석용 래퍼 (측정 코드는 그대로) | 11-16 ~ 11-17 | `src_cap/cap_worker.py`, `src_env/outside_worker.py` + 러너 복사본 | R_min 실측(메모리 상한)과 세 시점 outside 기록이 필요. 워커를 고치지 않으려고 래퍼로 감쌈 | `docs/05`, `docs/06` |
| ⑧ 이미지 추가 | 11-17 | `cu128-v1`, `pt251-cu124-v1` | Blackwell 지원(cu128), PyTorch 배포판 비교(pt251) | `docs/06` |

---

## 2. GPU 측정 워커 `measurement/local-gpu-vm/05-fp16-vram/src/vram_worker.py`

| sha | 날짜 | 변경 | 이유 | 사용한 실행 |
|---|---|---|---|---|
| `afdddc88` | 11-11 | 최초판: measure / validate / trace 모드, 조건마다 1프로세스 | ④ 재설계 | 스모크 |
| `1e86ee7e` | 11-11 | validate를 원시 logits forward hook으로 변경 | `min_new_tokens`가 EOS를 −inf로 막아서, 후처리된 점수로는 유한성 검사가 틀렸음 | 파일럿 v1 |
| `d06f0f15` | 11-11 | `--attn eager|default` 추가 | ⑤ | 파일럿 v2 이후 **모든 실측** (3090 L1024·L256, T4 L256, 3090 cu128, envcmp, capsearch, g7) |

- `d06f0f15` 이후 워커는 바꾸지 않았다. 모든 결과의 `worker_sha256`가 같다.
- 동작 요약: `docs/04` §3.1과 `docs/ai-context/AI_CONTEXT_KO.md` §5
  - FP16, 입력 `randint`/`randn`(시드 20261006), greedy, `min_new_tokens = max_new_tokens`
  - TF32 off, `cudnn.benchmark=False`, `CUDA_MODULE_LOADING=LAZY`
  - 기준값은 `summary.first_measure_peak_reserved`

## 3. 러너와 래퍼

| 파일 | sha / 날짜 | 원본과의 관계 | 바뀐 점 | 쓰인 곳 |
|---|---|---|---|---|
| `src/run_conditions.py` | `eb0196ea` → `27fae8d2` → **`77e3f4a9`** (11-12 ~) | — | v2: `--attn`. 현재판: `--conditions`(조건표 선택), `--retry-status`, 결과 없는 종료를 러너가 기록. 이전판은 `run_conditions.py.v2pilot`로 보존 | 모든 본실험 |
| `tools/run_conditions_cap.py` | 11-16 | 현재 러너 복사본 | `--mem-cap-mib`, 진입점 `src_cap/cap_worker.py`, 결과 폴더 `<cid>__cap<MiB>` | GPU R_min 탐색 |
| `src_cap/cap_worker.py` | 11-16 | 워커를 `runpy`로 그대로 실행 | `torch.cuda.init()` 직후 `set_per_process_memory_fraction(cap/total)` 적용 → 상한 초과 시 실제 cudaMalloc 실패와 같은 재시도·OOM 경로 | 〃 |
| `tools/gpu_rmin_search.py` | 11-16 | 새 파일 | ok/OOM 경계를 32 MiB 해상도로 이진 탐색 | 〃 |
| `tools/run_conditions_env.py` | 11-17 | 현재 러너 복사본 | `--env-name`, 진입점 `src_env/outside_worker.py`, 세션에 래퍼·러너 sha와 드라이버 기록 | g4dn 환경 비교, 3090 스모크 |
| `src_env/outside_worker.py` | 11-17 (sha `a85d5213`) | 워커를 `runpy`로 그대로 실행 | `torch.cuda.init`, `max_memory_allocated`, `reset_peak_memory_stats`를 감싸서 워커가 직접 부른 호출에서만 동기화 후 torch + NVML 기록 → `measure.outside.json` | 〃 |

- 래퍼가 측정값을 바꾸지 않는다는 확인
  - cap: 상한 없이 돌리면 기존 값과 같다(탐색 상한 검증).
  - outside: 3090 스모크 3조건에서 peak allocated, peak reserved, NVML 피크가 기존 L256과 바이트 단위로 같았다.
  - 두 래퍼 모두 워커 파일을 그대로 실행하므로 `worker_sha256`가 바뀌지 않는다.

## 4. 분석 도구 (CPU 전용, 측정과 분리)

| 파일 | 날짜 | 역할 |
|---|---|---|
| `tools/xmem_replay.py` | 11-11 | GPU trace를 xMem에 그대로 재생 → 구간별 reserved 비교 (파일럿 12개 완전 일치) |
| `tools/reserved_anatomy.py` | 11-16 | 세그먼트를 만든 요청을 config 크기 공식으로 분류 (KV, logits, MLP 등) |
| `tools/xmem_replay_variants.py` | 11-16 | 한 가지만 바꾼 재생 (해제 시점, 작업공간 제거, warmup_0만, prefill 최대 요청 제거) |
| `measurement/local-gpu-vm/03-xmem-probes/reserved-anatomy-20261116/first_divergence.py` | 11-16 | xMem과 GPU가 처음 어긋나는 이벤트 찾기 |
| 〃 `replay_gpu_addresses.py` | 11-16 | xMem 세그먼트 주소에 GPU 실제 주소를 넣어 재생 → 바이트 일치 (주소 순서가 원인임을 확인) |
| 〃 `replay_addr_policy.py` | 11-16 | 주소 규칙(내림차순 등)으로 재생. **xMem 저장소 파일은 고치지 않고** 재생 스크립트 안에서 세그먼트 주소 함수를 바꿔 끼움 |
| 〃 `capacity_sweep.py` → `capacity_sweep_desc.py` | 11-16 | R_inf / R_min 계산. `_desc`는 원본 복사본에 내림차순 규칙을 넣은 것 |
| `tools/compare_envs.py` | 11-17 | 환경 A/C/B/REF 조건별 비교, 눈에 띈 조건 목록 (11-18: REF의 outside 누락 경고 제거) |
| `tools/select_envcmp_conditions.py` | 11-17 | 기존 실측에서 모델별 가벼움·중간·무거움 조건 자동 선택 (11-17: 동점이면 입력이 작은 조건 우선) |
| `tools/compare_gpu_runs.py` | 11-18 | 조건별 CSV 두 개 비교(버전·GPU 효과), outside 분포 |
| `envcmp/check_envcmp.py` | 11-18 | 환경 비교가 계획대로 끝났는지 점검 (드라이버·torch·이미지·커널·세 시점 기록·동시 실행) |

- xMem: 기준 커밋과 로컬 수정은 `measurement/local-gpu-vm/03-xmem-probes/xMem-local-changes/` (`BASE_COMMIT.txt`, `local_modifications.patch`)
  - 로컬 수정은 `exp/trainer/*`, `perf_estimator/estimator.py`이고 할당기 파일은 아니다.
  - INUOS는 xMem `1ee4bd08`을 수정 없이 쓴다.

## 5. Docker 이미지

| 태그 | ID (index / config) | 만든 날 | 내용 | 왜 | 쓰인 곳 |
|---|---|---|---|---|---|
| `cloudgpufit-bench:cu124-v1` | `457da255` / `e33dd4e4` | 10월 (11-11 고정 기록) | `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` + transformers 4.52.4, accelerate 1.4.0, nvidia-ml-py, Pillow. python 3.11.11, cuDNN 9.1.0 | 기준 이미지 | 3090 전 실험, T4 L256, envcmp A·B |
| `cloudgpufit-bench:cu128-v1` | `798b909f` / `f7efefc8` | 11-17 | base `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime` (digest `c16f4c74…`) + pip 9개를 cu124-v1과 같은 버전으로 고정. torch 2.7.1, CUDA 12.8, cuDNN 9.7.1, python 3.11.13 | Blackwell(sm_120) 지원. cu124-v1은 sm_90까지만 | g7, 3090 cu128 |
| `cloudgpufit-bench:pt251-cu124-v1` | `0b22fba1` / `f47e2cce` | 11-17 | **cu124-v1 위에 pip 레이어 1개**: torch 2.5.1, torchvision 0.20.1, torchaudio 2.5.1, triton 3.1.0. 나머지 동일 (`/opt/pt251_pip_changes.txt`) | PyTorch 배포판만 바꾼 비교 | envcmp C |

- 정의 파일: `measurement/local-gpu-vm/05-fp16-vram/aws/IMAGE_INFO.json`, `IMAGE_INFO_cu128.json`, `IMAGES_INFO_envcmp.json`, `Dockerfile.cu128`, `Dockerfile.pt251`
- `docker save` 결과는 결정적이다(두 번 저장해 sha256 동일 확인). 번들 안 이미지는 sha256로 검증한 뒤 로드한다.

## 6. AWS 실행 패키지와 스크립트

| 번들 (레포 밖, sha는 `EXCLUDED_LARGE_FILES.md`) | 날짜 | 내용 | 결과 |
|---|---|---|---|
| `cgf-fp16-aws-full-20261112.tar` | 11-12 | L1024 조건표, cu124 이미지 | (L4 실행 이력 있으나 결과 미수신) |
| `cgf-fp16-aws-full-20261115.tar` | 11-15 | L256 조건표로 변경 | g4dn.xlarge L256 395 완료 (E8), g7 실패 |
| `cgf-fp16-aws-cu128-20261117.tar` | 11-17 | cu128 이미지, `IMAGE_VARIANT=cu128` | g7 측정 완료, 결과 미수신 |
| `cgf-envcmp-g4dn-20261117.tar` | 11-17 | 39조건, 이미지 2개(cu124-v1 + pt251), 래퍼·러너·환경 비교 스크립트, 이전 g4dn 원본 39개(REF) | envcmp A/C/B 완료 (E11) |

- `aws/*.sh` 변경 이력
  - `*.v20261112`: L1024 시절
  - `*.v20261115`: L256 전환 직후
  - 현재판(11-17): 이미지 종류 선택(`CGF_IMAGE_VARIANT` 또는 `IMAGE_VARIANT` 파일), 라벨 `__cu128` 접미사, 러너에 `--image` 전달
  - 기본값은 cu124 그대로라 이전 실행 방식은 바뀌지 않는다.
- `envcmp/driver_install_b.sh`: AWS 실행 중 문제 두 가지를 반영해 수정했다(11-17). AWS에 보낸 판에는 이 수정이 없다.
  - 설치 전에 persistence 끄기 → `efa_nv_peermem`, `gdrdrv`를 먼저 내리고 nvidia 모듈 내리기

## 7. 조건표

| 파일 | sha | 개수 | 비고 |
|---|---|---|---|
| `config/conditions.json` | — | 680 | 공통표 (텍스트 {32,128,512,1024}² 등) |
| `config/conditions_main395.json` | `2634bda7` | 395 | L1024 본실험 |
| `config/conditions_main395_l256.json` | `d4a9b09c` | 395 | 1024 → 256. 245개는 L1024와 같은 객체 |
| `config/conditions_envcmp39.json` | `3420a89f` | 39 | envcmp. L256 조건 객체를 그대로 복사 (선택 기준표 `envcmp39_selection.*`) |
| `config/models.json` | — | 13모델 | revision 고정. `.bin` 가중치: opt-125m, vit-large (`use_safetensors=false`) |

## 8. INUOS(CPU 예측) 코드 요약 — 상세는 `prediction/local-cpu-vm/INUOS_CONTEXT_KO.md` §7, `prediction/local-cpu-vm/docs/03`

| 단계 | 코드 | 변경 이유 |
|---|---|---|
| v2 휴리스틱 (10-26) | `00-early-eval-recovery/` | FP32 prefill × 0.5 + 보정 상수. 정확도 부족으로 폐기 |
| native v3 → recovery → singlepass v4 (11-03 ~ 11-10) | `01-cpu-predictor-342/` | 실제 추론 trace + xMem 재생. host OOM과 시간 문제로 스트리밍 파서와 단일 패스로 개선 |
| CPU worker v1 → **v2** (11-11 ~) | `03-fp16-cpu-prediction/src/cgf680/cpu_worker.py` → `cpu_worker2.py` | GPU 워커(`d06f0f15`)와 같은 입력·생성·절차(`protocol2.py`). FP32 실행 + FP16 요청 크기 환산(`shadow.py`), 로딩 요청 합성(`load_emu.py`) |
| reserved 예측 v3 (11-16) | `src/cgf680/replay3.py`, `gemm_probe.py`, `gemm_shadow.py`, `tools/reserved_predict.py` | INUGPU 가이드 반영: 내림차순 주소, cuBLAS 작업공간 8,519,680 B를 첫 GEMM 직후 삽입(CPU 프로브로 위치 확인), 용량 = cuda_total − outside 312 MiB. **생산 worker(`cpu_worker2`)는 그대로 두고 새 파일로 추가** |
| CPU 실행 이미지 | `cloudgpufit-cpu-validation:20261108` (`875290b5`) | torch 2.6.0+cu124, transformers 4.52.4 (GPU 이미지와 같은 버전) |

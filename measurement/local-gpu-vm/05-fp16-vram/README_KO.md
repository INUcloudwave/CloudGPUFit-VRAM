# CloudGPUFit FP16 VRAM 실험 (CPU xMem 예측 vs GPU 실측)

- 범위: VRAM만. performance 모드·사전 전체-logits 검증 forward는 측정 프로세스에서 제거
- 위치: `/home/inugpu/cgf-fp16-vram-20261111` (기존 저장소·결과 미수정, git 미사용 별도 경로)

## 1. 구성

| 경로 | 내용 |
|---|---|
| `config/generate_conditions.py` | 680개 조건표·26개 파일럿 생성기 (결정적) |
| `config/models.json` | 13개 모델 revision·로더·safetensors 여부 + 공통 규칙 |
| `config/conditions.json`, `conditions.csv` | 공통 680개 조건 (GPU·CPU 공유) |
| `config/pilot.json` | RTX 3090 파일럿 26개 |
| `config/trace_pilot.json` | allocator history 수집 대상 5개 |
| `config/model_cache_manifest.json` | 13개 모델 스냅샷 파일·blob(sha256)·크기 (26.1GiB) |
| `src/vram_worker.py` | worker (measure / validate / trace) |
| `src/run_conditions.py` | 호스트 runner (조건마다 새 컨테이너·프로세스, 순차, 재개) |
| `tools/prepare_cache.py` | 새 호스트에서 고정 캐시 구성·검증 |
| `tools/xmem_replay.py` | GPU allocator trace → xMem allocator 재생 (CPU) |
| `tools/summarize_runs.py` | 결과 요약 CSV/표 |

## 2. 조건 생성 규칙

- 배치 1, 8, 16, 32, 128 × 모델별 후보 전부 교차, FP16, 총 680개
- 조건 ID: 텍스트 `<short>__b<B>__in<L>__new<N>`, 이미지 `<short>__b<B>__r<R>`
- 텍스트 입력: `torch.Generator().manual_seed(20261006)` → `randint(3, config.vocab_size, (B, L), int64)`
  - config의 pad/eos/bos/decoder_start 토큰 ID(정수 또는 리스트)는 3으로 치환
  - attention_mask = ones(B, L, int64). tokenizer 미사용 → BOS 추가 없음, truncation 없음
- 이미지 입력: 같은 seed → `randn(B, 3, R, R, float32).to(float16)`. 이미지 프로세서 미사용 → 리사이즈 없음
- 입력은 host CPU에 두고 매 호출 `.to("cuda")`
- 모델 로딩: `from_pretrained(revision 고정, torch_dtype=fp16, low_cpu_mem_usage, device_map cuda:0, local_files_only, attn_implementation="eager")` → `.to(cuda, fp16).eval()`
- 생성: `GenerationConfig.from_model_config(config)`로 hub generation_config.json 대체
  - greedy, beam 1, KV cache, `min_new_tokens = max_new_tokens = N`
  - 반복 패널티·sampling 등 hub 기본값 미적용
  - 실제 신규 생성 수 검사: decoder `out.shape[1]-L == N`, T5 `out.shape[1]-1 == N` (시작 토큰 제외)
- decoder 위치 한계 검사: `L+N > max_position_embeddings`면 invalid_input (GPT-2 512+512=1024는 허용)

## 3. GPU 측정 프로토콜 (`measure`)

- 조건마다 새 `docker run --rm` → 새 worker 프로세스, 모델·입력 1회 로드
- warmup 3회 → measure 5회, 같은 입력·호출, inference_mode, allocator 캐시 유지
- 각 measure:
  1. synchronize → 시작 allocated/reserved 기록
  2. `reset_peak_memory_stats()` → NVML 샘플러 시작 (같은 PID, 10ms)
  3. 추론 → synchronize → peak allocated/reserved, 종료값(출력·GPU 입력 참조 유지 상태) 기록
  4. NVML 샘플러 종료 → 출력 host 복사·sha → 출력/입력 해제 → synchronize → 해제 후 값 기록
- 요약: 1회차(CPU 비교용) 시작·피크, 5회 중앙값·최댓값(용량 판단용), NVML 최대
- 실패 분류: gpu_oom / cpu_oom / fp16_unsupported / invalid_input / generation_length_mismatch / nonfinite_output / compat_or_runtime_error
  - `failed_stage`에 model_load / warmup_i / measure_i 기록
  - worker가 결과 없이 죽으면 runner가 timeout / worker_killed_cpu_oom_suspected(137) / worker_exit_N 기록
- 출력 유효성: 별도 프로세스 `validate` — 모든 forward의 원시 logits 유한성(forward hook) + 생성 길이
- allocator history: 별도 프로세스 `trace` — load → warmup 3 → measure 1, 경계 카운터 + snapshot(gz)

## 4. 실행 명령

```bash
cd /home/inugpu/cgf-fp16-vram-20261111
# 조건표 재생성 (sha가 같아야 함)
python3 -I config/generate_conditions.py config
# 파일럿 (tmux 세션 cgfvram)
./run_pilot.sh 2>&1 | tee logs/pilot.txt
# 임의 조건 / 전체
python3 -I src/run_conditions.py --list pilot|all|<ids,콤마>|<file.json> --mode measure --gpu-label <label> [--instance-type <type>]
# 요약
python3 -I tools/summarize_runs.py runs/<label> config/pilot.json
# xMem 재생 (xmem-gpu 컨테이너, GPU 미사용)
docker exec -e CUDA_VISIBLE_DEVICES= xmem-gpu python /work/cgf-fp16-replay-20261111/xmem_replay.py /work/xMem <trace.json> <out.json>
```

## 5. 결과 스키마

- 경로: `runs/<gpu_label>/<condition_id>/<mode>.json` + `.launch.json`(docker 명령·rc·wall) + stdout/stderr
- `runs/<gpu_label>/progress.jsonl` (조건별 상태), `sessions.jsonl` (실행 세션: 이미지 id/digest, GPU, instance type)
- `measure.json` 주요 필드
  - `condition`, `status`, `failed_stage`, `error`
  - `environment`: GPU 이름·UUID·총용량, driver, torch/CUDA/cuDNN/transformers/python, CUDA_MODULE_LOADING(env·드라이버 조회), allocator backend·환경변수, TF32, cudnn.benchmark, image id, gpu_label, instance_type
  - `model`: id, revision, loader, class, attention 구현, 파라미터 dtype·bytes
  - `inputs`: 텐서별 shape·dtype·sha256, `generation_config`
  - `after_load`, `warmup[]`, `measures[]`(start/peak/end_with_output/released/nvml/new_tokens/output_sha256/elapsed), `summary`
- CPU·GPU 비교 스키마(행 = 조건 × GPU 타입)

| 필드 | 의미 |
|---|---|
| condition_id, gpu_type, instance_type | 추적 키 |
| cpu_pred_start_reserved / cpu_pred_peak_reserved / cpu_pred_peak_allocated | xMem: warmup 3 + measure 1 재생, measure 경계에서 peak 통계만 초기화 |
| gpu_m1_start_reserved / gpu_m1_peak_reserved / gpu_m1_peak_allocated | GPU 1회차 |
| gpu_max_peak_reserved, gpu_max_nvml_peak | 5회 최댓값 (용량 판단용) |
| diff_peak_reserved = cpu − gpu_m1, start_state_diff = cpu_start − gpu_start | 오차·시작 상태 차이 |
| cpu_status, gpu_status, notes | 실패·제약 사유 |

## 6. AWS 이관·재개

- 이미지: `cloudgpufit-bench:cu124-v1` (id `sha256:457da255…fb032`). `docker save | gzip` → `docker load` 후 id 일치 확인
- 모델 캐시: AWS에서 `tools/prepare_cache.py`로 고정 revision 직접 다운로드 + manifest 검증 (26.1GiB, 3090에서 업로드 불필요)
- 실행: 같은 `run_conditions.py`, `--gpu-label <계열>` (instance type은 IMDS 자동 기록 또는 `--instance-type`)
- 재개: 같은 명령 재실행 → `<mode>.json`이 있는 조건은 건너뜀. 실패 재시도는 `--retry-status <status>` (이전 파일은 attempts/로 이동, 삭제 없음)
- 계열 순서: g6 → g4dn, g5, g6e, gr6, g7, g7e. 각 계열 단일 GPU 1대
- 호환성 주의: g7·g7e(Blackwell)는 CUDA 12.4 / torch 2.6 cu124 휠이 지원하지 않을 가능성이 큼 → 새 공통 이미지가 필요하면 3090에서 파일럿 재검증 후 사용

## 7. RTX 3090 결과 (2026-11-11)

- 환경: RTX 3090 24GiB (cc 8.6), driver 570.211.01, 이미지 `cloudgpufit-bench:cu124-v1` (sha256:457da255…), python 3.11.11, torch 2.6.0+cu124, cuDNN 90100, transformers 4.52.4, CUDA_MODULE_LOADING LAZY(env·드라이버), allocator 설정 없음(native), TF32 off, cudnn.benchmark off, torch threads 4 (vCPU 4)
- 스모크: gpt2 b1/32/32, resnet-18 b1/r128, t5-small b1/32/32 — measure·validate·trace 모두 ok (14:30~14:32)
  - 스모크 중 발견·수정: validate가 logits processor 이후 점수(min_new_tokens의 EOS=-inf)를 검사 → 원시 logits forward hook으로 변경. 이전 판정은 `runs/rtx3090-smoke/.../attempts/`에 보존
- 파일럿 경과: measure 26개 21분 23초 (14:32:34→14:53:57), validate 26개 5분 12초, trace 5개 1분 20초
- measure 26/26 ok, 생성 수 전부 요청값과 일치, OOM 0
- validate 24/26 ok, Qwen2.5-1.5B-Instruct 2개 nonfinite_output (첫 forward부터). VRAM 측정 자체는 완료됨
- 표: `results/pilot_rtx3090_summary.md` / `.csv`
- xMem 재생(5개 trace): load 후·warmup·measure_0 경계의 start/end/peak reserved·allocated가 GPU와 모두 일치, segment 수 일치, 짝 없는 free 0 → `results/xmem_replay_rtx3090/`

## 8. Qwen2.5-1.5B-Instruct FP16 nonfinite 진단 (`results/diag_qwen/`)

- 측정(measure)은 정상 완료. 별도 validate에서 첫 forward부터 logits 비유한
- 변형 비교 (B8/L128 prefill 1회): FP16+eager만 실패. FP16+sdpa, BF16(eager/sdpa), FP32(eager/sdpa) 모두 유한. 입력 ID 범위(config vocab vs tokenizer vocab)는 무관
- 첫 비유한 지점: `model.layers.0.self_attn` (o_proj 출력)
- 원인 관측: transformers 4.52.4 eager는 `matmul(Q, K^T)` 후에 `* scaling` → 0번 레이어 unscaled |QK^T| 최대 127,295 (B8/L128), 82,706 (B1/L32) > FP16 최대 65,504 → inf → softmax NaN. scaling 후 값(11,251 / 7,310)은 FP16 범위 안
- B1/L32에서도 넘침 → Qwen FP16 eager 80개 조건 전부 같은 실패가 예상됨 (전 조건 검증은 아님)

## 9. attention 기본값 재실행 (2026-11-11)

- 파일럿 v1은 `attn_implementation="eager"` 명시 고정이었음 (transformers 기본값 아님)
- v2: 인자 생략(기본값) → `runs/rtx3090__attn-default/`, `run_pilot_attn_default.sh`, `--attn default`
- 통합 보고서: `results/REPORT_rtx3090_pilot_eager_vs_default_KO.md`, CSV: `results/pilot_rtx3090_eager_vs_default.csv`

## 10. 본실험 조건표 축소 + AWS 이관 준비 (2026-11-11)

- 새 조건표: `config/conditions_main395.json` / `.csv`
  - 생성 스크립트: `config/generate_main.py` (680표에서 부분집합을 추출)
  - 조건 수는 395개입니다. 요청 본문은 355개였지만 그룹별 개수 합(135+60+135+60+5)은 395이고, 규칙대로 만들었습니다.
- runner 변경 (`src/run_conditions.py`, 이전 버전은 `src/run_conditions.py.v2pilot`로 보존)
  - `--conditions` 옵션 추가
  - `--attn` 기본값을 default로 변경
  - dry-run 시 선택된 condition_id 목록을 출력
  - session에 조건 파일·코드 sha256 기록
- worker(`src/vram_worker.py`)는 v2 검증본 그대로이며 변경하지 않았습니다.
- 경로 확인: `runs/rtx3090__main395-pathcheck` (3개 조건). 결과는 `logs/main395_pathcheck_compare.txt`
- AWS 관련 파일
  - 런북: `AWS_RUNBOOK_KO.md`
  - 스크립트: `aws/`
  - 판정 도구: `tools/check_pilot.py`
  - 캐시 해시 검증: `tools/prepare_cache.py --hash` (이전 버전은 `tools/prepare_cache.py.v1`로 보존)

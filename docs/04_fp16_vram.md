# 04. FP16 VRAM 실측 (RTX 3090 / AWS 공용 패키지)

> 레포 경로 대응: VM `/home/inugpu/cgf-fp16-vram-20261111/` → 레포 `gpu-vm/05-fp16-vram/` (진행 중인 `runs/rtx3090__main395_l256/`은 종료 후 추가). 대용량 제외 파일은 `EXCLUDED_LARGE_FILES.md` 참고.


- 루트: `/home/inugpu/cgf-fp16-vram-20261111` (아래 경로는 모두 이 루트 기준 상대경로)
- 근거: 코드, `runs/*/sessions.jsonl`·`progress.jsonl`, `logs/`, `results/`, 기존 문서(`README_KO.md`, `AWS_RUNBOOK_KO.md`, `results/REPORT_rtx3090_pilot_eager_vs_default_KO.md`, INUOS 전달 문서 2종)
- 작성 시점: 2026-11-15 18:13 KST (L256 본실험 진행 중)

---

## 1. 개요

### 1.1 목적
- CPU에서 xMem으로 예측한 FP16 추론 VRAM과 비교할 **GPU 실측 정답값** 확보
- 범위는 VRAM만. performance 측정과 사전 전체-logits 검증 forward는 측정 프로세스에서 제외
- 1차 비교 기준값: `measure.json`의 `summary.first_measure_peak_reserved` (byte)
  - 보조: `first_measure_peak_allocated`, `after_load.reserved_bytes.all.current`, `first_measure_start_reserved`

### 1.2 기간
- 2026-11-11 14:30: 스모크 시작
- 2026-11-12 23:13: main395(L1024) 종료
- 2026-11-15 17:00: main395_l256 시작, 작성 시점 기준 진행 중

### 1.3 실행 환경 (`measure.json`의 `environment`, `aws/IMAGE_INFO.json`)

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, 24576 MiB, cc 8.6, `cuda_total_memory` 25,306,529,792 B |
| 드라이버 | 570.211.01 |
| Docker 이미지 | `cloudgpufit-bench:cu124-v1`, ID `sha256:457da255…fb032` (OCI index digest), config digest `sha256:e33dd4e4…` |
| 이미지 내용 | python 3.11.11, torch 2.6.0+cu124 (CUDA 12.4), cuDNN 90100, transformers 4.52.4 (base `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`) |
| allocator | native, `PYTORCH_CUDA_ALLOC_CONF` 미설정, expandable_segments off |
| 설정 | `CUDA_MODULE_LOADING=LAZY`(환경변수와 드라이버 조회 모두), TF32 off, `cudnn.benchmark=False`, torch threads 4 (vCPU 4) |
| 시드 | 20261006 |
| 모델 캐시 | 13개 모델, 75개 파일, 27,989,184,806 B (`config/model_cache_manifest.json`) |

### 1.4 모델 (13개, `config/models.json`, revision 고정)
- decoder_llm: opt-125m, TinyLlama-1.1B-Chat-v1.0, Qwen2.5-1.5B-Instruct, gpt2 / gpt2-medium / gpt2-xl
- text_encdec: t5-small / t5-base / t5-3b
- vision_cls: resnet-18, resnet-50, mobilenet_v2_1.0_224, vit-large-patch16-224

### 1.5 worker 코드 버전 (sha256 앞 8자리)

| sha | 내용 | 사용처 |
|---|---|---|
| `afdddc88` | 최초판 | 스모크 measure·trace, 스모크 첫 validate |
| `1e86ee7e` | validate를 원시 logits forward hook으로 수정 | 스모크 validate 재실행, 파일럿 v1 measure·validate·trace(5개) |
| `d06f0f15` | `--attn` 옵션 추가 (v2 검증본) | 파일럿 v2, v1 라벨의 gpt2 b16 trace, pathcheck, main395, main395_l256 |

- runner: `eb0196ea`(v1) → `27fae8d2`(v2 파일럿, `src/run_conditions.py.v2pilot`로 보존) → `77e3f4a9`(현재, main395 이후)

---

## 2. 실험 목록 (시간순)

### 2.1 스모크 (2026-11-11 14:30~14:32, 15:35)
- 라벨: `runs/rtx3090-smoke`, `runs/rtx3090-smoke__attn-default`
- 조건: gpt2 b1/in32/new32, resnet-18 b1/r128, t5-small b1/in32/new32 (680표 조건)
- 실행: `run_conditions.py --list <id,…> --mode measure|validate|trace --gpu-label rtx3090-smoke`
- 상태
  - measure 3/3 ok, trace 1/1 ok (gpt2)
  - validate: 처음 gpt2가 `nonfinite_output` (first step 0)
    - 원인: logits processor 이후 점수(min_new_tokens가 EOS를 -inf로 만듦)를 검사했음
    - 수정: 원시 logits forward hook으로 변경 → 재실행 gpt2·t5-small ok
    - 이전 판정은 `runs/rtx3090-smoke/gpt2__b1__in32__new32/attempts/20261111T143209__validate.*`에 보존
  - attn-default 스모크: gpt2 b1/32/32 measure ok (sdpa)
- 값: gpt2 b1/32/32 m1_peak_reserved 478 MiB (eager·sdpa 같음), resnet-18 b1/r128 50 MiB, t5-small b1/32/32 272 MiB
- xMem 재생: `results/xmem_replay_rtx3090/smoke/smoke_gpt2.json`
  - 경계 10/10, 짝 없는 free 0, segment 12/12, 로드 후 reserved 일치

### 2.2 파일럿 v1: eager 고정 (2026-11-11 14:32~15:00)
- 라벨: `runs/rtx3090`, 스크립트 `run_pilot.sh` (로그 `logs/pilot.txt`)
- 설계: 26개 조건 (`config/pilot.json`)
  - 텍스트 9종 × {b8/128/128, b16/512/512}
  - CNN 3종 × {b8/r224, b16/r512}
  - ViT × {b8/r224, b16/r224}
- 로딩: `attn_implementation="eager"` 명시 (transformers 기본값 아님)
- 순서: measure 26 → validate 26 → trace 5 (`config/trace_pilot.json`)
- 시각
  - measure 14:32:34→14:53:57 (21분 23초)
  - validate ~14:59:11 (5분 12초)
  - trace ~15:00:31
- 상태
  - measure 26/26 ok, 생성 토큰 수 전부 요청값과 일치, OOM 0
  - validate 24/26 ok. Qwen2.5-1.5B 2개는 `nonfinite_output`(첫 forward부터)
  - trace 5/5 ok
- 표: `results/pilot_rtx3090_summary.{md,csv}`, `logs/pilot_summary.md`

### 2.3 파일럿 v2: transformers 기본 attention (2026-11-11 15:36~16:05)
- 라벨: `runs/rtx3090__attn-default`, 스크립트 `run_pilot_attn_default.sh` (로그 `logs/pilot_v2_attn_default.txt`)
- 차이: `--attn default` → `attn_implementation` 인자 생략
  - 실제 결과: OPT·TinyLlama·Qwen·GPT-2 3종·ViT는 sdpa (14조건)
  - T5 3종(sdpa 미지원)과 CNN(attention 없음)은 eager (12조건)
- 시각
  - measure 15:36:13→15:57:15 (21분 2초)
  - validate →16:02:28
  - trace(default 6개) 16:02:30→16:04:15
- 상태: measure 26/26 ok, validate 26/26 ok, trace 6/6 ok
- 주요 결과 (통합 보고서 §3)
  - sdpa 14조건: peak reserved 감소 10, 동일 3, 증가 1 (gpt2-xl b8/128 +28 MiB)
  - 최대 차이: TinyLlama b16/512/512 reserved −1060 MiB, allocated −832 MiB
  - eager로 같은 12조건은 두 실행 값이 모두 같음 (재현 확인)
  - 52개 measure 전부 5회 peak reserved 동일, 1회차 시작 reserved = 1회차 peak reserved
- 통합 산출물: `results/REPORT_rtx3090_pilot_eager_vs_default_KO.md`, `results/pilot_rtx3090_eager_vs_default.csv` (`tools/build_report.py`)
  - 사본: `reference/pilot_rtx3090_eager_vs_default.csv` (`check_pilot.py`가 입력 해시 기준으로 사용)

### 2.4 allocator trace (2026-11-11)
- 모드: `--mode trace` (load → warmup 3 → measure 1, `_record_memory_history`)
- v1(eager): `config/trace_pilot.json` 5개, 14:59~15:00, worker `1e86ee7e`
- v2
  - `config/trace_pilot_v2.json` 6개 = v1 5개 + gpt2 b16/512/512
  - default: 6개 모두 실행
  - eager: `runs/rtx3090` 라벨로 실행. 기존 5개는 skip, gpt2 b16만 새로 실행 (16:04, worker `d06f0f15`)
- 12개 trace(eager 6 + default 6) 공통 확인
  - `start_empty=True`
  - `alloc_events_equal_counter=True` (잘림 없음)
  - 경계 10개

| 조건 | eager alloc / segment | default alloc / segment |
|---|---|---|
| gpt2 b8/128/128 | 244,162 / 11 | 204,734 / 11 |
| opt-125m b8/128/128 | 213,879 / 4 | 186,739 / 4 |
| qwen2.5-1.5b b8/128/128 | 753,908 / 62 | 694,004 / 61 |
| t5-small b8/128/128 | 273,124 / 36 | 273,124 / 36 |
| resnet-50 b8/r224 | 1,757 / 12 | 1,757 / 12 |
| gpt2 b16/512/512 | 975,298 / 26 | 817,598 / 24 |

### 2.5 xMem 재생 (2026-11-11, 결과 파일 mtime 16:06~16:09)
- 도구: `tools/xmem_replay.py`. xmem-gpu 컨테이너에서 `CUDA_VISIBLE_DEVICES=`로 GPU 없이 실행 (README §4)
- 방법
  - GPU의 `alloc`(원요청 크기)과 `free_completed` 이벤트를 xMem `CachingAllocator`에 순서대로 재생
  - segment 이벤트는 재생하지 않고 개수만 비교
- 결과 (`results/xmem_replay_rtx3090/{eager,default}/*.json`, 표 `replay_table.md`)
  - 12개 전부 경계 10/10
  - 짝 없는 free 0
  - segment 수 일치
  - 로드 후·각 구간 시작/끝 reserved, allocated와 measure_0 peak reserved/allocated가 GPU와 모두 일치
  - 예: gpt2 b16/512/512 m1 peak reserved: eager 1982.0, sdpa 1838.0 MiB (GPU=xMem)
- 재생 소요: 0.1~54.0 s/조건

### 2.6 Qwen2.5-1.5B FP16 eager nonfinite 진단 (2026-11-11 15:22~15:23, 파일 mtime)
- 도구: `tools/diag_qwen_nonfinite.py`, `tools/diag_qwen_qk.py`
  - 실행 명령은 파일에 기록되지 않음
- 변형 비교 (B8/L128 prefill 1회, `results/diag_qwen/diag_qwen_nonfinite.json`)
  - FP16+eager만 비유한 (입력 ID 범위 config vocab 151936 / tokenizer 151665 둘 다)
  - FP16+sdpa, BF16(eager/sdpa), FP32(eager/sdpa)는 모두 유한
  - 첫 비유한 모듈: `model.layers.0.self_attn.o_proj`
- QK 크기 (`diag_qwen_qk.json`, FP32에서 측정, layer 0)
  - 스케일 전 |QKᵀ| 최대: B8/L128 127,295, B1/L32 82,706 → FP16 최대 65,504 초과
  - 스케일 후: 11,251 / 7,310
  - 넘침 레이어는 두 크기 모두 [0]
- 해석(문서 기재): transformers 4.52.4 eager가 `matmul(Q,Kᵀ)` 후에 `* scaling`을 하기 때문에 넘침이 생김

### 2.7 attention 커널 진단 (2026-11-11 ~16:05)
- 도구: `tools/diag_attention_kernels.py`
  - `run_pilot_attn_default.sh` 13행에서 docker로 실행
  - 출력: `results/attention_kernels_rtx3090_default.{json,log}`
- 대상: v2 파일럿 중 sdpa 14조건
  - generate 1토큰(prefill)과 2토큰의 profiler 커널 차이로 decode 커널을 분리
  - ViT는 forward 1회
- 결과
  - 14조건 모두 prefill·decode에서 `pytorch_flash::flash_fwd_kernel`만 관측
  - prefill 호출 수 = 레이어 수 (opt/gpt2 12, TinyLlama 22, gpt2-medium·ViT 24, Qwen 28, gpt2-xl 48)
  - sdpa backend 플래그(flash/mem_efficient/math/cudnn)는 모두 True

### 2.8 pathcheck (2026-11-11 17:43~17:44)
- 목적: runner 변경(`--conditions`, `--attn` 기본값 default, session 해시 기록) 후 경로 확인
- 스크립트: `logs/run_main395_pathcheck.sh`, 라벨 `runs/rtx3090__main395-pathcheck`
- 조건 3개: gpt2 b8/128/128, resnet-18 b8/r224, opt-125m b1/128/128 (`conditions_main395.json`)
- 상태: measure 3/3 ok, validate 3/3 ok
- 비교 (`logs/main395_pathcheck_compare.txt`)
  - gpt2·resnet-18은 v2 파일럿과 m1 peak, 출력 sha, 입력 sha가 동일 (476.0 / 80.0 MiB)
  - opt-125m b1은 파일럿에 없어 비교 대상 없음 (330.0 MiB)
  - 비교표의 `peaks_equal_5` 열은 잘못된 키를 써서 무효라고 파일에 적혀 있음. 아래에 회차별 peak가 따로 있고 5회 모두 같음

### 2.9 main395 L1024 (2026-11-12 09:51~23:13)
- 조건표: `config/conditions_main395.json` (sha `2634bda7…`, 395개, 680표의 부분집합)
  - OPT/TinyLlama/Qwen, T5 3종: L×N ∈ {128,512,1024}²
  - GPT-2 3종: {128,512}²
  - CNN: r ∈ {128,224,256,512}
  - ViT: 224
  - 배치: 1/8/16/32/128
  - 요청 본문은 355개였지만 그룹 합(135+60+135+60+5)이 395라서 규칙대로 395개로 만들었다고 기록됨
- 스크립트: `run_main395_vm.sh` (measure 전체 → validate 전체 → `summarize_runs.py`), 로그 `logs/main395_rtx3090.txt`
- 시각
  - measure 09:52:11→21:07:57 (wall 합 11.26 h)
  - validate 21:08:05→23:13:03 (2.08 h)
- 상태
  - measure 381 ok / 14 gpu_oom (모두 `failed_stage=warmup_0`)
  - validate 371 ok / 24 gpu_oom
    - measure OOM 14개는 validate에서도 OOM
    - measure는 ok인데 validate만 OOM인 조건 10개: opt-125m b128 6, gpt2 b128 2, gpt2-medium b128 2 (모두 `measure_0`)
  - validate ok 371개 모두 `all_finite=True`
- ok 381개 전부: m1 시작 reserved = m1 peak reserved, max_peak = m1_peak
- 조건별 상세: `results/gpu_rtx3090_main395_L1024_per_condition.csv` (byte, `tools/export_for_inuos.py`)
- 요약: `runs/rtx3090__main395/summary.csv`, `logs/main395_rtx3090__main395_summary.md`

### 2.10 main395_l256 (2026-11-15 17:00 ~ 11-16 00:45, 완료 — 결과 요약은 ../AI_CONTEXT_KO.md §4.5)
- 조건표: `config/conditions_main395_l256.json` (sha `d4a9b09c…`, 395개)
  - 바뀐 점: 1024 → 256 (OPT/TinyLlama/Qwen, T5 3종의 L×N ∈ {128,256,512}²). 1024 조건의 실행 시간이 너무 길어서 바꿈
  - 나머지(GPT-2, CNN, ViT, 배치, 모델·revision, worker·runner 코드)는 같음
  - 680표에 있는 245개는 원본 객체와 같음 (assert로 확인)
  - 256이 들어간 150개는 같은 형식으로 새로 만듦
  - 파일럿 26개 포함
- 스크립트: `run_main395_l256_vm.sh` (tmux), 로그 `logs/main395_l256_rtx3090.txt`
- 시각: `MAIN_START 2026-11-15T17:00:58`, session 시작 17:01:00
- 진행 (`progress.jsonl`, 18:13:33 기준)
  - measure 87/395 완료, 전부 ok (opt-125m 45, TinyLlama 42)
  - validate는 아직 시작 전
  - 실행 순서는 models.json 순서: opt → TinyLlama → Qwen → gpt2×3 → CNN×3 → ViT → T5×3
- 종료 후 생성 예정: `runs/rtx3090__main395_l256/summary.csv`, `logs/main395_rtx3090__main395_l256_summary.md`

### 2.11 AWS 이관 패키지 (11-18 갱신: T4 L256·envcmp 결과, cu128 번들은 `docs/06`, 계보는 `docs/07`)
- 런북: `AWS_RUNBOOK_KO.md`. 스크립트: `aws/`. 계열 순서: g6 → g4dn, g5, g6e, gr6, g7, g7e
- 2026-11-15 기본값
  - 조건표: `conditions_main395_l256.json`
  - 라벨: `aws-<type>__l256`
  - 이전 1024 조건표는 `CGF_CONDITIONS=conditions_main395.json` (라벨 `aws-<type>`)
- INUGPU 자가 점검 (2026-11-11 17:45~17:49)
  - 이미지 스트림 sha와 diff_id: `IMAGE_OK true` (`logs/aws_scripts_selftest_inugpu.txt`)
  - 캐시 75파일 내용 해시 불일치 0 (`logs/cache_hash_verify_inugpu.txt`)
- `check_pilot.py` 자가 점검(런북 기재): RTX 3090 v2는 26/26 PASS, v1 eager는 0/26 FAIL (음성 대조)
- AWS 실행 결과는 이 VM에 없음 (`runs/aws-*` 없음)

---

## 3. 코드 설명

### 3.1 `src/vram_worker.py` (424줄, sha `d06f0f15`)
- 역할: 프로세스 1개가 (조건, 모드) 1개를 처리. 컨테이너 안에서 실행
- CLI: `--models`, `--conditions`, `--condition`, `--mode {measure,validate,trace}`, `--out`, `--warmup 3`, `--measure 5`, `--nvml-interval 0.01`, `--threads min(8,cpu)`, `--attn {eager,default}`
  - 주의: worker 자체의 `--attn` 기본값은 `eager`(152행). runner가 항상 `--attn`을 명시해서 넘김
- 모드별 반복 수(165행): measure = warmup 3 / measure 5, trace = 3 / 1, validate = 0 / 1
- 주요 흐름 (줄번호)

| 줄 | 단계 |
|---|---|
| 27-29 | `CUDA_MODULE_LOADING` 미설정이면 LAZY |
| 59-61 | `stats()`: `memory_stats()` 중 15개 키 (allocated/reserved current·peak, segment, alloc 카운터, num_ooms 등) |
| 64-78 | `classify()`: gpu_oom / cpu_oom / fp16_unsupported / invalid_input / generation_length_mismatch / nonfinite_output / compat_or_runtime_error |
| 93-138 | `NvmlSampler`: 같은 PID의 NVML 프로세스 사용량을 10 ms마다 샘플 (`--pid=host` 필요) |
| 156-158 | 출력 파일이 이미 있으면 덮어쓰기를 거부 (rc 3) |
| 167-173 | 결과 dict 초기화 (worker/models/conditions sha, protocol 설명) |
| 177-182 | `save()`: `.tmp`에 쓴 뒤 replace (원자적 저장) |
| 186-191 | 스레드, `manual_seed`, TF32 off, benchmark off, GPU 1개 assert |
| 192-195 | trace 모드: `torch.cuda.init()` 전에 `_record_memory_history(enabled="all", stacks="python", max_entries=5e7)` |
| 200-226 | environment 기록. 다른 GPU 프로세스가 있으면 중단 |
| 227 | `start_state` |
| 229-232 | `boundary()`: trace 모드에서 synchronize 후 카운터 기록 |
| 239 | 로드 직전 `reset_peak_memory_stats()` |
| 241-247 | `from_pretrained(revision, fp16, low_cpu_mem_usage, device_map cuda:0, use_safetensors, local_files_only, [attn eager])` → `.to(cuda,fp16).eval()`, output_attentions/hidden_states False |
| 251-264 | `after_load` 통계, `model` 정보 (class, 실제 `_attn_implementation`, sdpa backend 플래그, param/buffer bytes) |
| 268-274 | 비전 입력: `randn(B,3,R,R, fp32).to(fp16)` |
| 276-284 | 텍스트 입력: `randint(3, vocab, (B,L))`, 특수 토큰은 3으로 치환, attention_mask 1 |
| 285-289 | decoder에서 L+N > max_position_embeddings면 invalid_input |
| 290-297 | `GenerationConfig.from_model_config` + greedy, beam 1, cache, min=max=N |
| 298 | 입력 텐서별 shape, dtype, sha256 기록 |
| 302-306 | validate: 모든 forward의 원시 `output.logits`에 `isfinite().all()` hook |
| 308-319 | `invoke()`: 매 회차 `.to("cuda")` 복사, generate 또는 forward, 생성 길이 확인 (decoder `shape[1]-L`, T5 `shape[1]-1`) |
| 322-338 | warmup 루프: 회차별 elapsed, new_tokens, end_with_output, released |
| 343-368 | measure 루프 |
| 346 | 시작 allocated/reserved 기록 |
| 348 | `reset_peak_memory_stats()` (peak만 초기화, 캐시 유지, `empty_cache` 호출 없음) |
| 356-359 | peak, end_with_output, stats_end 기록 |
| 361-364 | NVML 결과, 출력을 host로 복사해 sha 계산, del |
| 369-374 | validation 요약 (steps_checked, all_finite, first_nonfinite_step) |
| 375-385 | `summary` (first_measure_*, median/max peak, NVML 최대/중앙값) |
| 389-397 | 예외: status 분류, failed_stage, error, traceback, `stats_at_failure` |
| 399-418 | trace: `_snapshot()` → `trace.snapshot.pickle.gz`, action_counts, `alloc_events_equal_counter`, `start_empty` |
| 419-420 | 저장. 종료코드는 ok면 0, 아니면 1 |

- 출력 schema
  - 공통: `schema, mode, condition, status, stage, failed_stage?, error?, worker_sha256, models_json_sha256, conditions_json_sha256, argv, protocol, environment, start_state, after_load, model, generation_config, inputs, special_tokens_replaced_by_3, warmup[], measures[], timings_s, worker_wall_s`
  - `measures[i]`: `i, ok, new_tokens, elapsed_s, start{allocated,reserved}, peak{…}, end_with_output{…}, stats_end, released{…}, output_sha256`, measure 모드에서는 `nvml_process_peak_bytes, nvml_samples*`
  - measure.json: `summary{first_measure_peak_reserved/allocated, first_measure_start_reserved/allocated, median/max_peak_reserved, median/max_peak_allocated, max/median_nvml_process_peak}`
  - validate.json: `validation{steps_checked, all_finite, first_nonfinite_step}`. 메모리 값은 측정값으로 쓰지 않음 (hook 추가 할당, warmup 없음)
  - trace.json: `history{boundaries[](이름+stats), action_counts, end, alloc_events_equal_counter, start_empty, snapshot_file}`
    - 경계 이름: `model_load_begin/end, warmup_{0..2}_begin/end, measure_0_begin/end`

### 3.2 `src/run_conditions.py` (167줄, sha `77e3f4a9`)
- 역할: 호스트 runner. 조건마다 새 `docker run --rm`, 순차 실행, 재개 가능. 표준 라이브러리만 사용
- CLI: `--list pilot|all|<file.json>|id,id`, `--mode`, `--gpu-label`(필수), `--instance-type`, `--image cloudgpufit-bench:cu124-v1`, `--hf-cache`, `--root`, `--timeout 3600`, `--retry-status`(반복 가능), `--dry-run`, `--conditions conditions.json`, `--attn default`

| 줄 | 단계 |
|---|---|
| 38-48 | AWS IMDSv2로 instance type 조회 (실패하면 None) |
| 69-72 | `--conditions`는 `<root>/config` 안의 파일만 허용 |
| 75-85 | `--list` 해석, 모르는 id가 있으면 중단 |
| 87-93 | image id/digest, `nvidia-smi` GPU가 정확히 1개인지 확인, instance type |
| 97-110 | session(시작 시각, argv, 이미지, GPU, 조건 파일 sha, worker·runner sha)을 `sessions.jsonl`에 추가. dry-run이면 id 목록만 출력 |
| 116-125 | `<mode>.json`이 있으면 skip. `--retry-status`에 해당하면 기존 파일을 `attempts/<stamp>__*`로 이동 |
| 126-128 | 다른 GPU 프로세스가 있으면 중단 |
| 130-138 | docker 명령: `--network none --pid=host --gpus device=<UUID> --user uid:gid`, HF offline, src/config/hf-cache는 ro 마운트, 조건 디렉터리를 `/out`에 마운트 |
| 143-149 | timeout이면 `docker stop` 후 rc를 `"timeout"`으로 기록 |
| 150-151 | `<mode>.launch.json` (cmd, start, end, wall_s, returncode) |
| 152-157 | worker 결과가 없으면 runner가 `timeout` / `worker_killed_cpu_oom_suspected`(137) / `worker_exit_<rc>`로 기록 |
| 159-162 | `progress.jsonl`에 `{t, condition_id, mode, status, rc, wall_s}` 추가 |

- 출력: `runs/<label>/<cid>/<mode>.{json,launch.json,stdout.log,stderr.log}`, `progress.jsonl`, `sessions.jsonl`
- `src/run_conditions.py.v2pilot`: 이전판. `--conditions` 옵션 없음(conditions.json 고정), `--attn` 기본값 eager

### 3.3 `config/*.py`
- `generate_conditions.py`: 680개 공통표 생성 (결정적)
  - 텍스트 4종(opt/TinyLlama/Qwen/T5×3)은 {32,128,512,1024}², GPT-2 3종은 {32,128,512}², CNN은 4해상도, ViT는 224
  - 배치 5종
  - 파일럿 26개도 생성. 출력: `models.json, conditions.json/.csv, pilot.json` + sha 출력
  - `models.json.common.attn_implementation: "eager"`는 생성 당시의 설명 문구. worker는 이 값을 읽지 않음
- `generate_main.py`: 680표에서 395개 부분집합 추출 → `conditions_main395.json/.csv`
  - assert: 개수, 모델별·그룹별 개수, 독립 재열거, 파일럿 포함, 원본 객체와 동일
- `generate_main_l256.py`: 1024를 256으로 바꾼 395개 → `conditions_main395_l256.json/.csv`
  - assert: 1024 없음, 680표와 겹치는 245개는 키 순서까지 동일, 새 150개에는 모두 256 포함, 파일럿 포함
- 조건 객체: 텍스트 `{condition_id, model, workload_type, batch_size, input_len, max_new_tokens}`, 비전 `{…, resolution}`

### 3.4 `tools/`

| 파일 | 역할 / 사용법 | 입력 → 출력 |
|---|---|---|
| `summarize_runs.py` | `summarize_runs.py runs/<label> [list.json] > summary.md` | measure/validate/launch json → `runs/<label>/summary.csv`(MiB) + md 표 |
| `export_for_inuos.py` | `export_for_inuos.py runs/<label> <conditions.json> <out.csv>` | 조건별 1행(byte, 입력 sha 전체 64자, `fail_*`, `validate_*`, 조건표에 있지만 결과가 없으면 `missing`) |
| `check_pilot.py` | `check_pilot.py <root> <label> [list] --conditions <f>` | PASS 조건: measure ok·5회 ok·토큰 수 일치, validate ok·all_finite, `attn_requested=default`, worker sha `d06f0f15…`, 입력 sha 앞 16자 = `reference/…csv`. VRAM은 참고로만 출력. 종료코드 0/1 |
| `build_report.py` | `build_report.py <root>` | v1/v2 파일럿 → `results/pilot_rtx3090_eager_vs_default.csv` + md 표 |
| `xmem_replay.py` | `xmem_replay.py <xmem_repo> <trace.json> <out.json>` (GPU 없이) | trace + snapshot → 구간별 GPU/xMem start·peak·end reserved/allocated, segment 수, missing_frees |
| `replay_table.py` | `replay_table.py <root>` | 재생 json 12개 → `replay_table.md` |
| `diag_attention_kernels.py` | 컨테이너에서 `… /config <out.json> id…` | profiler 커널 이름(prefill/decode, ViT forward) → json |
| `diag_qwen_nonfinite.py` | `… <out.json>` | dtype × attention × vocab 범위 변형별 첫 비유한 모듈, 레이어 최대값 |
| `diag_qwen_qk.py` | `… <out.json>` | eager attention을 패치해 레이어별 스케일 전/후 max\|QKᵀ\| 기록 (FP32) |
| `prepare_cache.py` | 컨테이너에서 `… /exp/config [--verify-only] [--hash]` | manifest 기준 고정 revision 다운로드 + 크기·blob 이름(+내용 해시) 검증, 불일치면 rc 1 |
| `model_cache_filelist.py` | `… <config_dir> <hf_home> > model_cache_filelist.txt` | snapshot symlink, blob, refs의 상대경로 목록 (tar -T 스트리밍용) |
| `download_new_models.py` | 공유 캐시에 없던 4개 모델(opt-125m, TinyLlama, Qwen2.5-1.5B, resnet-18)을 고정 revision으로 다운로드 | HF 캐시 (`logs/download.log`) |

- `tools/prepare_cache.py.v1`: `--hash` 추가 이전판

### 3.5 `run_*.sh` (루트)
- `run_pilot.sh`: 라벨 rtx3090. 파일럿 measure → validate → trace(`trace_pilot.json`) (v1: 당시 runner·worker에 `--attn` 옵션이 없었고 eager로 고정)
- `run_pilot_attn_default.sh`: 라벨 `rtx3090__attn-default`
  - `--attn default`로 measure → validate → trace(v2 6개)
  - 이어서 eager trace 6개(라벨 rtx3090)
  - 마지막에 docker로 `diag_attention_kernels.py` (14조건)
- `run_main395_vm.sh`: `conditions_main395.json`, 라벨 `rtx3090__main395`, measure all → validate all → summarize. runner 실패 시 `RUNNER_ABORT`
- `run_main395_l256_vm.sh`: 위와 같은 구조. `conditions_main395_l256.json`, 라벨 `rtx3090__main395_l256`
- `logs/run_main395_pathcheck.sh`: 3개 조건 measure·validate (2.8절)

### 3.6 `aws/*.sh`
- `RUN_ON_AWS.sh`: 단일 진입점. 다시 실행하면 이어서 진행
  - 순서: SHA256SUMS 확인 → preflight → 이미지 확인(이미 같은 이미지면 skip, 아니면 load) → 캐시 verify(실패 시 download) → 파일럿 → `PILOT_PASSED`가 있으면 본실험
  - 로그: `logs/RUN_ON_AWS_<label>.txt`
- `aws_00_preflight.sh <type>`: IMDS instance type이 기대값과 같은지, GPU 1개인지 확인. docker·디스크·메모리 출력
- `aws_01_load_image.sh [file]`: docker save 스트림을 load하면서 sha256·바이트 수·image id·diff_id를 `IMAGE_INFO.json`과 대조 → `IMAGE_OK`
- `aws_02_cache.sh download|verify <cache>`: 컨테이너에서 `prepare_cache.py` 실행. verify는 `--network none --hash`
- `aws_03_pilot.sh <label> <cache>`: 파일럿 measure → validate → `check_pilot.py`. rc 0이면 `runs/<label>/PILOT_PASSED`
- `aws_04_main.sh <label> <cache>`: `PILOT_PASSED`가 없으면 거부. measure → validate(`CGF_MAIN_VALIDATE=0`이면 생략) → summarize
- `*.v20261112`: 1024 조건표 고정판 (라벨 `aws-<type>`)
- `aws/IMAGE_INFO.json`: 태그, OCI index/config digest, diff_id 7개, save 스트림 sha256(`ac217e13…`)·바이트 수(3,358,278,144)

### 3.7 `summary.csv` 열 (`summarize_runs.py`, MiB = byte / 2^20)
- `condition_id, status, failed_stage, error(160자)`
- `load_reserved_MiB` (= `after_load.reserved_bytes.all.current`)
- `m1_start_reserved_MiB, m1_peak_reserved_MiB, m1_peak_allocated_MiB, median_peak_reserved_MiB, max_peak_reserved_MiB, max_nvml_peak_MiB`
- `new_tokens, measure_s_median, model_load_s, wall_s`
- `validate_status, validate_all_finite, validate_steps, validate_first_nonfinite_step`

---

## 4. 결과 요약

### 4.1 파일럿 (MiB, m1_peak_reserved, eager → default)
- 26개 전체 표: `results/REPORT_rtx3090_pilot_eager_vs_default_KO.md` §3
- 주요 값

| 조건 | eager | default |
|---|---|---|
| opt-125m b16/512/512 | 1940 | 1524 |
| tinyllama b16/512/512 | 3986 | 2926 |
| qwen2.5 b16/512/512 | 4200 | 3868 |
| gpt2-xl b16/512/512 | 9374 | 9174 |
| t5-3b b16/512/512 | 12996 | 12996 |
| vit-large b16/r224 | 788 | 738 |

### 4.2 main395 L1024 모델별 (measure, `gpu_rtx3090_main395_L1024_per_condition.csv`)

| 모델 | n | ok | OOM | validate 실패 | 실제 attn | 로드 후 reserved | m1_peak_reserved 최소 (조건) | 최대 (조건) | measure 시간 |
|---|---|---|---|---|---|---|---|---|---|
| opt-125m | 45 | 45 | 0 | 6 | sdpa | 316 | 330 (b1/128/128) | 20570 (b128/1024/128) | 43.3분 |
| tinyllama-1.1b | 45 | 45 | 0 | 0 | sdpa | 2234 | 2262 (b1/128/128) | 20730 (b128/128/512) | 147.7분 |
| qwen2.5-1.5b | 45 | 45 | 0 | 0 | sdpa | 3448 | 3450 (b8/512/128) | 20562 (b128/128/1024) | 164.9분 |
| gpt2 | 20 | 20 | 0 | 2 | sdpa | 474 | 476 (b8/128/128) | 21026 (b128/128/512) | 11.2분 |
| gpt2-medium | 20 | 20 | 0 | 2 | sdpa | 1002 | 1004 (b8/128/128) | 22528 (b128/512/128) | 22.2분 |
| gpt2-xl | 20 | 17 | 3 | 3 | sdpa | 3486 | 3488 (b1/512/128) | 17828 (b16/128/512) | 28.6분 |
| resnet-18 | 20 | 20 | 0 | 0 | eager | 48 | 50 (b1/r128) | 2548 (b128/r512) | 1.9분 |
| resnet-50 | 20 | 20 | 0 | 0 | eager | 82 | 84 (b1/r128) | 6168 (b128/r512) | 2.0분 |
| mobilenet-v2 | 20 | 20 | 0 | 0 | eager | 26 | 28 (b1/r128) | 6580 (b128/r512) | 2.0분 |
| vit-large | 5 | 5 | 0 | 0 | sdpa | 620 | 624 (b1) | 1594 (b128) | 0.7분 |
| t5-small | 45 | 45 | 0 | 0 | eager | 272 | 276 (b1/128/128) | 17570 (b128/1024/128) | 46.0분 |
| t5-base | 45 | 45 | 0 | 0 | eager | 678 | 688 (b1/128/128) | 23556 (b128/1024/128) | 85.8분 |
| t5-3b | 45 | 34 | 11 | 11 | eager | 7166 | 7218 (b16/128/128) | 23524 (b32/512/512) | 119.5분 |

- 단위: MiB. 시간은 `progress.jsonl`의 wall_s 합
- measure gpu_oom 14개 (모두 warmup_0)
  - gpt2-xl b128: 128/512, 512/128, 512/512
  - t5-3b b128: 128/512, 128/1024, 512/128, 512/512, 512/1024, 1024/128, 1024/512, 1024/1024
  - t5-3b b32: 512/1024, 1024/512, 1024/1024
- `stats_at_failure` reserved peak: 21,762~23,814 MiB
  - t5-3b b128/1024/*는 8.00 GiB 할당 시도에서 실패

### 4.3 main395_l256 진행 (`progress.jsonl`, 2026-11-15 18:13:33 기준)
- measure 87/395 완료, 87 ok, 실패 0. validate 0/395
- 완료: opt-125m 45/45, tinyllama 42/45
- wall 합 1.21 h. 가장 긴 조건: tinyllama b128/256/512 273.9 s
- 남은 순서: tinyllama 3 → Qwen 45 → gpt2×3 60 → CNN 60 → ViT 5 → T5×3 135 → validate 395
- 값 비교(겹치는 245개가 L1024와 같은지)는 이 문서에서 하지 않음. 진행 중인 run 디렉터리는 `progress.jsonl`만 읽었음

---

## 5. 결과·로그 파일 위치

| 내용 | 경로 |
|---|---|
| 실험 설명 / AWS 런북 | `README_KO.md`, `AWS_RUNBOOK_KO.md` |
| 조건표·설정 | `config/conditions*.json/.csv`, `config/models.json`, `config/pilot.json`, `config/trace_pilot*.json` |
| 캐시 manifest | `config/model_cache_manifest.json`, `config/model_cache_filelist.txt` |
| 스모크 | `runs/rtx3090-smoke/`, `runs/rtx3090-smoke__attn-default/` |
| 파일럿 v1 (eager) + eager trace | `runs/rtx3090/` (`summary.csv` 포함), `logs/pilot.txt`, `logs/pilot_summary.md` |
| 파일럿 v2 (default) + trace | `runs/rtx3090__attn-default/`, `logs/pilot_v2_attn_default.txt` |
| 파일럿 통합 보고서 / CSV | `results/REPORT_rtx3090_pilot_eager_vs_default_KO.md`, `results/pilot_rtx3090_eager_vs_default{.csv,_tables.md}`, `results/pilot_rtx3090_summary.{csv,md}` |
| 파일럿 기준 CSV (check_pilot용) | `reference/pilot_rtx3090_eager_vs_default.csv` |
| trace 원본 | `runs/<label>/<cid>/trace.json`, `trace.snapshot.pickle.gz` |
| xMem 재생 | `results/xmem_replay_rtx3090/{eager,default,smoke}/`, `replay_table.md` |
| Qwen 진단 | `results/diag_qwen/diag_qwen_{nonfinite,qk}.{json,log}` |
| attention 커널 진단 | `results/attention_kernels_rtx3090_default.{json,log}` |
| pathcheck | `runs/rtx3090__main395-pathcheck/`, `logs/main395_pathcheck{,_compare}.txt` |
| main395 L1024 | `runs/rtx3090__main395/` (395×{measure,validate}, `summary.csv`), `logs/main395_rtx3090.txt`, `logs/main395_rtx3090__main395_summary.md` |
| main395 L1024 조건별 CSV (byte) | `results/gpu_rtx3090_main395_L1024_per_condition.csv` |
| main395_l256 (진행 중) | `runs/rtx3090__main395_l256/`, `logs/main395_l256_rtx3090.txt` |
| 코드 해시 기록 | `logs/code_sha256_{pilot,pilot_v2,main395_bundle}.txt`, `logs/SHA256SUMS_code_config_results.txt` |
| AWS 자가 점검 | `logs/aws_scripts_selftest_inugpu.txt`, `logs/cache_hash_verify_inugpu.txt`, `logs/cache_verify_*.txt` |
| GPU 매니페스트·교차검토 (읽기 전용 감사, 2026-11-12) | `handoff_codex/{build_manifest.py,cross_review.py,gpu_manifest.json,validation.json}` |
| 이관 번들 (루트 밖) | `/home/inugpu/cgf-fp16-aws-bundle-20261111/`, `…-20261115/`, `/home/inugpu/cgf-fp16-inuos-handoff-2026111{2,5}/` |

---

## 6. 한계 / 미확인

- **validate 메모리**: hook 추가 할당과 warmup 0 때문에 측정값으로 쓰지 않음
  - measure ok인데 validate만 OOM인 10개(L1024)는 VRAM 값은 유효하지만 출력 유효성은 확인하지 못함
  - hook의 정확한 할당 내역은 trace를 뜨지 않아 미확인
- **trace 범위**: allocator 요청 기록은 파일럿 6개 조건(×2 설정)뿐
  - 본실험 조건의 요청 단위 분포는 없음
  - cuBLAS/cuBLASLt/cuDNN workspace 요청의 위치·빈도·수명은 미확인
- **Qwen FP16 eager**: 80개 조건 전부 같은 실패가 예상된다고 문서에 적혀 있지만, 전 조건 검증은 아님
  - v2 이후 본실험은 sdpa라 해당 없음
- **커널 진단**: 파일럿 sdpa 14조건만. 본실험의 큰 배치·긴 길이에서도 FlashAttention만 쓰는지는 미확인
- **NVML 값**: CUDA context 등 allocator 밖 메모리를 포함하므로 reserved와 직접 비교 불가
- **GPU 의존성**: CNN cuDNN workspace는 GPU와 cuDNN 버전에 따라 다를 수 있음
  - AWS 각 계열(특히 g7/g7e Blackwell)에서 cu124 이미지가 호환되는지는 미확인 (README §6)
- **AWS 결과**: 이 VM에 없음. 라벨, 인스턴스, 해시, 회수 경로 미확인
- **L256**: 진행 중 (작성 시점 87/395 measure)
  - OOM 후보(gpt2-xl b128 3개, t5-3b b128의 128/512 조합, 256이 들어간 t5-3b 일부)는 INUOS 문서의 예상일 뿐 실측 전
  - 겹치는 245개가 L1024와 같은 값인지는 아직 확인하지 않음
- **조건 수**: 요청 본문은 355개였지만 그룹 규칙의 합이 395라서 395개로 진행 (문서화됨)
- **실행 명령 기록 없음**: Qwen 진단과 xMem 재생은 스크립트에 기록되지 않음
  - 시각은 결과 파일 mtime 기준
  - xMem 재생 명령 형식은 README §4에만 있음
- **pathcheck 비교표**: `peaks_equal_5` 열은 잘못된 키를 써서 무효 (파일에 명시). 회차별 peak 목록으로 대신 확인

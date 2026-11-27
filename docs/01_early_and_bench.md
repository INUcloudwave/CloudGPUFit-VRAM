# 01. 초기 실험 및 cloudgpufit-bench (INUGPU, RTX 3090)

> 레포 경로 대응: VM `/home/inugpu/<파일>` → 레포 `gpu-vm/00-early-scripts/` (루트 스크립트), `cloudgpufit-bench/` → `gpu-vm/01-cloudgpufit-bench/`. 대용량 제외 파일은 `EXCLUDED_LARGE_FILES.md` 참고.


> 작성 근거: `/home/inugpu` 아래 파일 내용·mtime·결과 JSON/CSV, 그리고 셸 히스토리에 남은 일부 명령. 경로는 모두 `/home/inugpu` 기준 상대경로.
> 시각: 결과 폴더 이름은 UTC(`...Z`), 파일 mtime은 VM 로컬 시각(UTC+9, 폴더명과 9시간 차이로 확인).
> "(추정)"은 파일로 직접 확인되지 않은 해석, "미확인"은 확인 불가 항목.

---

## 1. 개요

- **목적**
  - 초기(2026-03): RTX 3090에서 ResNet-50 / YOLOv5s의 FLOPs·VRAM·지연·처리량 측정. 스크립트 출력에 "CloudSim GPUCloudlet 파라미터 요약"이 있어, 당시에는 CloudSim 시뮬레이션 입력값을 만드는 용도였음
  - 2026-09: 24GB GPU에 14B LLM(Qwen2.5-14B-Instruct, FP16)이 그대로 올라가는지(strict), CPU offload로 돌아가는지 확인
  - 2026-10: `cloudgpufit-bench`로 HF 모델 9종 × 342 조건의 추론 지연·메모리(GPU 실측값, ground truth)를 측정하고, 실행 시간을 줄인 러너(v2→v3)가 기존 결과를 재현하는지 검증
- **기간 (파일 근거)**

| 시기 | 실험 | 근거 |
|---|---|---|
| 2026-03-31 | ResNet-50 / YOLOv5s 벤치마크 | `benchmark_*.py`, `*_benchmark.json` mtime |
| 2026-09-14 | 14B strict/offload OOM 테스트 | `oom_test.py`, `strict_14b.json`, `offload_14b.json` mtime |
| 2026-10-06 ~ 10-07 | bench v1: 환경 구축, smoke, 342조건 전체 실행 | `Dockerfile*`, `results/20261006T*` |
| 2026-10-18 | v2/v3 러너 검증(baseline9, 대조 실험) | `results/20261018T*`, zip/tar mtime |
| 2026-10-26 | v3 러너로 342조건 전체 재실행(all342) | `results/20261026T*`, `inugpu-all342-results.tar.gz` |
| 2026-10-27 | recheck6: 차이가 컸던 6조건을 원래 코드로 재측정 | `recheck6_*`, `results/20261027T*` |
| 2026-11-01 / 11-03 | 모델 revision 목록 내보내기 / outside 실험용 UUID 패치 | `cloudgpufit-model-revisions.json`, `fix_uuid_check.py` |

- **답하려던 질문**
  - 워크로드 유형(decoder LLM / vision 분류 / text enc-dec), 모델 크기, batch, 길이, dtype에 따라 RTX 3090의 실측 VRAM(NVML 프로세스 사용량, torch reserved/allocated)과 지연이 어떻게 달라지는가
  - 메모리 구성(가중치 storage, self/cross KV cache, reserved 바깥 메모리 = NVML − reserved)을 분리해서 기록할 수 있는가
  - 측정 프로토콜을 바꿔(모델 재사용, 반복 수 축소) 실행 시간을 줄여도 메모리 측정값이 유지되는가
- **확인된 사실 / 범위 밖**
  - 이 폴더의 결과는 전부 `instance_label=inugpu-rtx3090`. AWS(T4/A10G/L4/L40S)에서 실행한 결과는 이 범위에 없음
  - 계획 파일에 `aws_core` 프로파일(repeats 3, warmup 10, iterations 100)과 bf16 미지원 GPU 처리(`unsupported_dtype`) 코드는 있음. AWS 실행 여부는 **미확인**

---

## 2. 실험 목록

### 2.1 초기 ResNet-50 / YOLOv5s 벤치마크 (2026-03-31)
- 설계: torchvision ResNet-50(`weights=None`, 무작위 가중치), ultralytics YOLOv5s(640×640)
- 조건: FP32(기본값), ResNet batch 1/8/16/32/64, YOLO batch 1/4/8/16/32. 추가로 학습 스텝 1개 측정(ResNet batch 32, YOLO batch 8)
- 측정: CUDA Event 타이밍(warmup 3, repeat 10), `max_memory_allocated`, torch.profiler FLOPs
- 환경(결과 JSON 기준): RTX 3090, total_memory 23.57GB, CUDA 11.8. `venv`에 torch 2.7.1+cu118, ultralytics 8.4.32가 설치돼 있어 이 venv에서 실행한 것으로 보임(추정)
- 실행 명령: 히스토리에 없음(**미확인**). 스크립트는 현재 디렉터리에 JSON을 저장하는 구조

### 2.2 14B LLM strict / offload OOM 테스트 (2026-09-14)
- 모델: `Qwen/Qwen2.5-14B-Instruct`, FP16
- 모드
  - `strict`: `device_map={"":0}`, GPU에 전부 올림
  - `offload`: `device_map="auto"`, `max_memory={GPU 22GiB, CPU 38GiB}`, `offload_folder=./_offload`
- 절차: 로드 → 짧은 프롬프트로 warmup(8 tok) → `--new-tokens`(기본 64) greedy 생성
- 판정값: `LOAD_OOM` / `RUNTIME_OOM` / `FIT_OK` / `OFFLOAD_OK`
- 실행 명령: 히스토리에 없음. 인자 구조로 보면 `python oom_test.py --model Qwen/Qwen2.5-14B-Instruct --mode {strict|offload} --json {strict|offload}_14b.json` 형태(추정)

### 2.3 cloudgpufit-bench 공통 설계 (342조건)
- 환경: Docker 이미지 `cloudgpufit-bench:cu124-v1`
  - 베이스: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`(digest 고정), transformers 4.52.4, accelerate 1.4.0, nvidia-ml-py 12.570.86
  - 결과 JSON 기준: torch 2.6.0+cu124, Python 3.11.11, 드라이버 570.211.01, CPU 스레드 4
- 모델 9종 (HF revision 고정, `hf-cache/benchmark-revisions.json`)

| workload_type | small | medium | large |
|---|---|---|---|
| decoder_llm | gpt2 | gpt2-medium | gpt2-xl |
| vision_cls | mobilenet_v2_1.0_224 | resnet-50 | vit-large-patch16-224 |
| text_encdec | t5-small | t5-base | t5-3b |

- shape 생성 규칙(`make_benchmark_plan.py`): 기본 조건 → 축 하나씩 변경 → 복합 조건 추가, 중복 제거

| 유형 | 기본 | 변경 축 | 복합(extras) | shape 수 |
|---|---|---|---|---|
| decoder_llm (batch, seq_len, max_new_tokens) | (1,256,64) | batch 1/2/4/8, seq 128/256/512/768, new 16/64/128/256 | (4,512,128) (4,768,128) (8,128,16) (1,768,256) | 14 |
| vision_cls (batch, resolution) | (1,224) | batch 1/2/4/8/16, res 128/224/384/512 | (4,384) (4,512) (8,128) | 11 (ViT-L은 224만, 5개) |
| text_encdec (batch, enc_len, dec_len, new) | (1,256,1,64) | batch 1/2/4, enc 128/256/512/1024, dec 1/32/128, new 16/64/128/256 | (2,512,1,128) (2,1024,1,128) (4,128,1,16) (1,512,128,128) | 15 |

- dtype: fp16 / fp32 / bf16 → core 342개 (decoder 126, vision 81, enc-dec 135), smoke 9개(모델별 fp16 1개)
- 측정 모드 5종
  - `performance`: 전체 generate(또는 vision forward) 지연, p90, 처리량
  - `memory`: 50ms 간격 NVML 프로세스 메모리 샘플링 + torch peak reserved/allocated
  - `first_logits`: forward 1회(첫 토큰 logits) 지연. vision은 해당 없음
  - `tokens`: streamer로 TTFT / TPOT / ITL 측정. vision은 해당 없음
  - `components`: forward hook으로 생성 스텝마다 self/cross KV storage, 바깥 메모리 스냅샷(1회)
- 고정 설정: seed 20261006, eager attention, TF32 끔, greedy(num_beams 1), `min_new_tokens = max_new_tokens`, use_cache, 컨테이너 네트워크 차단 + HF 오프라인
- 메모리 가드: reserved 또는 NVML peak가 VRAM의 85%를 넘으면 같은 (모델, dtype, 비-batch shape)에서 더 큰 batch는 건너뜀(`skipped_memory_guard`)

### 2.4 실행 이력 (`cloudgpufit-bench/results/`)

| 결과 폴더 | 시각(로컬) | 러너 / 격리 | 조건 | 반복(warmup / 본측정) | 상태 | 소요 |
|---|---|---|---|---|---|---|
| `20261006T070324Z_bd56be04` | 10-06 16:03~16:30 | v1 | smoke 9 | 1 / 3 | ok 39, n/a 6 | - |
| `20261006T074129Z_59c3aaef` | 10-06 16:41~16:48 | v1 | smoke 9 (재실행) | 1 / 3 | ok 39, n/a 6 | - |
| `20261006T083514Z_e0e9d733` | 10-06 17:35 ~ 10-07 11:25 | v1, 모드마다 새 컨테이너, 시작 온도 ≤36°C 대기 | core 342 | 3 / 20 | ok 1548, n/a 162 | 64,205 s (17.83 h) |
| `20261018T071920Z_v2_f88a5b92` | 10-18 16:19~16:31 | v2 shared-case | baseline9 | 3 / 20 | ok 39, n/a 6 | 728 s |
| `20261018T094213Z_v2_a048565d` | 10-18 18:42~18:44 | v2 shared-case | 2조건 × 3모드 | 3 / 20 | ok 6 | 110.5 s |
| `20261018T094608Z_v2_2932ef9a` | 10-18 18:46~18:48 | v2 separate-mode | 같은 2조건 × 3모드 | 3 / 20 | ok 6 | 132.6 s |
| `20261018T115334Z_v3_b82c33c4` | 10-18 20:53~21:06 | v3 hybrid | baseline9 | 3 / 20 | ok 39, n/a 6 | 762 s |
| `20261018T121515Z_v3_00e0fb68` | 10-18 21:15~21:19 | v3 hybrid, 반복 축소 | baseline9 | 1 / perf 5, mem 3, fl 5, tok 3 | ok 39, n/a 6 | 271 s |
| `20261026T073104Z_v3_70368c77` | 10-26 16:31~16:51 | v3 hybrid, 반복 축소 | core 342 (**중단**) | 1 / 5·3·5·3 | ok 195 (39조건까지) | 미완료 |
| `20261026T082509Z_v3_e8c4d64f` | 10-26 17:25~21:06 | v3 hybrid, 반복 축소 | core 342 | 1 / 5·3·5·3 | ok 1548, n/a 162 | 13,301 s (3.69 h) |
| `20261027T045157Z_9801141c` | 10-27 13:51~14:02 | v1(원래 코드) | recheck6 6조건 | 3 / 20 | ok 24, n/a 6 | - |

- 모든 실행: threads 4, 이미지 ID 동일, worker SHA256은 버전마다 일정(v1 `d0a1ae29…`, v2 `2a6b79dc…`, v3 `4018cd8e…`)
- n/a(`not_applicable`) = vision 조건의 first_logits / tokens 모드
- 상태 집계에 `gpu_oom`, `skipped_memory_guard`, `runtime_error`, `worker_failed`, `nonfinite_output`는 한 번도 없음

### 2.5 실행 명령 (스크립트·문서·히스토리 근거)
- 이미지 빌드(히스토리): `sudo docker build -f Dockerfile.benchmark --build-arg BASE_IMAGE="$BASE_IMAGE" -t cloudgpufit-bench:cu124-v1 .`
- GPT-2 smoke(히스토리): `sudo docker run --rm --gpus all ... cloudgpufit-bench:cu124-v1 python /work/check_gpt2.py` → `results/gpt2-smoke.json`
- 계획 생성: `python3 make_benchmark_plan.py` → `benchmark_plan.json`
- v1 smoke(히스토리): `python3 run_bench.py --profile local_smoke`
- v1 core 342: 명령은 히스토리에 없음. manifest(local_core, 5모드, threads 4, max_start_temp 36)로 보면 `python3 run_bench.py --profile local_core --threads 4 --max-start-temp 36` 형태(추정)
- v2 baseline9(`LOCAL_README.md`): `python3 run_bench_v2.py --profile local_core --case-set baseline9 --threads 4 --no-temperature-wait --reference results/20261006T083514Z_e0e9d733`
- v3 baseline9(`START_HERE.txt`): `python3 run_bench_v3.py --profile local_core --case-set baseline9 --threads 4 --no-temperature-wait --reference results/20261006T083514Z_e0e9d733`
- all342(히스토리): `python3 run_bench_v3.py --profile local_core --case-set all --threads 4 --no-temperature-wait --warmup 1 --performance-iterations 5 --memory-iterations 3 --first-logits-iterations 5 --token-iterations 3 --reference results/20261006T083514Z_e0e9d733`
  - 히스토리에서 이 명령 바로 뒤에 tmux 설치와 `tmux new -s gpu-bench`가 있음. 첫 시도(`70368c77`)가 세션 종료로 끊겨 tmux 안에서 다시 실행한 것으로 보임(추정, 중단 원인은 **미확인**)
- recheck6(히스토리)
  - 원래 코드 SHA256(`run_bench.py`, `bench_worker.py`)이 최초 실행과 같은지 먼저 검증
  - `results/20261006T083514Z_e0e9d733/selected-plan.json`에서 6조건만 뽑아 `recheck6-original-plan.json` 생성(warmup 3, 본측정 20)
  - `sudo -n nohup python3 -u run_bench.py --profile local_core --plan recheck6-original-plan.json --threads 4 --max-start-temp 36 --modes components memory performance first_logits tokens > recheck6_<시각>.log`
  - 결과 폴더 소유자가 root인 것은 sudo로 실행했기 때문

---

## 3. 코드 설명

### 3.1 `benchmark_resnet50.py`
- 역할: ResNet-50 추론(batch별) + 학습 1스텝의 시간·메모리·FLOPs 측정
- 흐름: `get_gpu_info` → batch마다 `measure_flops`(profiler, no_grad) → `measure_time`(CUDA Event) → `measure_memory`(empty_cache 후 peak) → 학습 스텝(SGD, batch 32)
- 출력: `resnet50_benchmark.json`
- 참고: 시간·메모리 측정의 `model(x)`는 `no_grad` 없이 실행돼 autograd 그래프가 생김(FLOPs 측정만 no_grad). 가중치는 무작위(`weights=None`)

### 3.2 `benchmark_yolov5.py`
- 역할: YOLOv5s 추론(batch 1~32, 640×640) + 학습 스텝(batch 8) 측정. 구조는 ResNet 스크립트와 같음
- 모델 로드: ultralytics `YOLO("yolov5s.pt")` 우선, 없으면 torch.hub. 홈에 있는 `yolov5su.pt`(03-31 19:32)는 이때 받은 가중치로 보임(추정)
- 출력: `yolov5_benchmark.json`
- 학습 스텝은 더미 손실(`p.mean()` 합) 사용. 실제로는 오류로 실패(결과 요약 참고)

### 3.3 `oom_test.py`
- 역할: FP16 Causal LM이 24GB에 들어가는지 판정
- 입력: `--model`, `--mode strict|offload`, `--new-tokens`, `--json`
- 흐름: 로드(OOM이면 `LOAD_OOM` 기록 후 종료) → `hf_device_map`으로 GPU/CPU 배치 집계 → warmup 생성 → 본 생성(OOM이면 `RUNTIME_OOM`) → tok/s·peak 기록
- 출력: 판정 JSON(`strict_14b.json`, `offload_14b.json`), `_offload/` 폴더(현재 비어 있음)

### 3.4 `fix_uuid_check.py` (bench와 별개, outside 실험 보조)
- 역할: `cloudgpufit-outside/outside-lab/probe.py`의 GPU UUID 비교 로직 패치. PyTorch UUID와 NVML UUID의 `GPU-` 접두사 차이를 정규화해서 비교
- 안전장치: 원본 SHA256 확인 → 문자열 치환 3곳 → 패치 결과 SHA256 확인 → 백업(`probe.py.before-uuid-fix`) 후 기록 → 정규화 함수 검증
- 상태: 히스토리에 실행 기록이 있고, 현재 `probe.py` SHA256이 스크립트의 `NEW_SHA`와 같고 백업 파일도 있음 → 패치 적용 완료

### 3.5 `cloudgpufit-bench/Dockerfile`, `Dockerfile.benchmark`
- 내용은 같음(줄바꿈만 다름). `BASE_IMAGE` 인자 위에 transformers/accelerate/nvidia-ml-py/Pillow 버전 고정, `pip freeze`를 `/opt/environment.lock.txt`로 저장
- `image-id.txt`(`sha256:67245a…`)는 첫 빌드 ID. 실제 실행 manifest의 이미지 ID는 모두 `sha256:457da255…`(`Dockerfile.benchmark`로 다시 빌드한 이미지로 추정)

### 3.6 `check_gpt2.py`
- 역할: 컨테이너 GPU 동작 확인. GPT-2 fp16, batch 1, 입력 128, 16토큰 생성 1회
- 출력: `results/gpt2-smoke.json`(용도 `smoke_check_only`)

### 3.7 `make_benchmark_plan.py`
- 역할: 9개 모델 × shape × dtype으로 smoke 9 / core 342 조건 생성
- 검증(assert): 조건 수, 유형별 수, case_id 중복 없음, decoder `seq_len + max_new_tokens ≤ 1024`, ViT-L은 224만
- 출력: `benchmark_plan.json`(이미 있으면 덮어쓰지 않음, `open("x")`)
- 참고: 계획 파일의 `local_core.modes`는 `[performance, memory]`이지만, `run_bench.py`의 `--modes` 기본값(5모드 전체)이 실행 시 이를 덮어씀. `selected-plan.json`에 5모드가 기록된 이유
- `benchmark_plan_v2.json`은 `benchmark_plan.json`과 바이트 단위로 같음(이름만 v2)

### 3.8 `bench_worker.py` = `bench_worker_legacy.py` (v1 워커)
- 두 파일은 같음(`cmp` 일치). 컨테이너 안에서 (조건 1개, 모드 1개)를 실행
- `--prepare`: 계획의 모델마다 HF revision을 고정하고 snapshot 다운로드(safetensors 우선) → `revisions.json`
- `main` 흐름
  1. 입력 필드 검증 → bf16 미지원 GPU면 `unsupported_dtype`, vision의 tokens/first_logits는 `not_applicable`
  2. 다른 GPU 프로세스가 있으면 `gpu_busy`
  3. 모델 로드(eager, `device_map cuda:0`). memory 모드면 로드 중에도 NVML 샘플링(startup peak)
  4. `memory_baseline`(파라미터/버퍼 storage, resident allocated/reserved), environment 기록
  5. CPU에서 무작위 입력 생성(특수 토큰 치환, enc-dec는 decoder prefix 첫 토큰 = start token)
  6. forward 1회로 logits 유한성 확인(아니면 `nonfinite_output`)
  7. warmup → `reset_peak_memory_stats` → 본측정(components는 1회)
  8. 모드별 지표 계산: latency 요약(n/mean/median/p90/max), 처리량, TTFT/TPOT/ITL, NVML 샘플(+`outside_reserved_bytes` = NVML − reserved), `memory_guard_exceeded`
- 측정 구간: "CPU 텐서 준비 완료 → H2D → 추론 → D2H". 모델 로드와 전처리는 제외
- OOM은 예외 문자열로 판별해서 `gpu_oom`으로 기록

### 3.9 `run_bench.py` (v1 러너, 호스트)
- 인자: `--profile`, `--plan`, `--image`, `--instance-label`, `--modes`, `--model`, `--limit`, `--threads`, `--max-start-temp`, `--idle-timeout`
- 흐름: 계획 필터링 → 결과 폴더(`results/<UTC>_<uuid8>`) 생성 → `selected-plan.json` → 이미지 ID 확인 → prepare 컨테이너 → `environment.lock.txt`, `run-manifest.json`
- 조건 × repeat × 모드마다
  - GPU idle 대기: 메모리 ≤ 기준+64MiB, 온도 ≤ max_start_temp(기본: 시작 온도+10, 최대 70), 사용률 ≤5%가 3초 연속
  - 매번 새 `docker run --rm --network none` 컨테이너로 워커 실행 → 조건별 JSON
  - OOM/가드가 나면 같은 shape의 더 큰 batch는 건너뜀
  - `summary.csv`, `index.json`에 누적
- 한 모드마다 모델을 새로 로드 → 342조건에서 워커 1,548회 실행

### 3.10 `bench_worker_v2.py` / `run_bench_v2.py` (v2)
- 목적: 같은 조건·반복 수에서 실행 시간 단축
- 워커 변경점(v1 대비)
  - `run_shared`: 한 컨테이너에서 모드를 `performance → memory → first_logits → tokens → components` 순서로 실행하고 모델은 조건 안에서만 재사용(`load_or_reuse`: config, generation_config, CPU/CUDA RNG 복원. 다른 조건으로 넘어가면 예외)
  - 모드 사이 `cleanup_between_modes`: sync + `gc.collect` + `empty_cache`
  - `wait_mode_idle`, `gpu_health`(온도, SM 클럭, thermal slowdown 비트를 측정 전후 경계에서 기록)
  - `input_sha256`으로 모드 간 입력이 같은지 확인. components 단계에서 nonfinite가 나오면 앞선 ok 결과를 `invalidated_by_component_check`로 바꿈
  - 단계별 `timings_seconds`, `BENCH_EVENT` 진행 이벤트 출력
  - 측정 함수 본체(invoke 등)는 v1과 같음
- 러너 변경점
  - `--isolation shared-case|separate-mode`(separate는 legacy 워커로 모드마다 새 컨테이너)
  - `--case-set all|baseline9|baseline27`(baseline9 = 모델별 첫 fp16 조건, `__01`)
  - `--no-temperature-wait`, 모드별 반복 수 오버라이드, `--dry-run`, 진행률·ETA heartbeat
  - 추가 출력: `events.jsonl`, `host-timings.json`, `case-timings.json`. summary 컬럼에 `model_reused`, `worker_main_seconds` 추가

### 3.11 `bench_worker_v3.py` / `run_bench_v3.py` (v3)
- 워커: v2와 차이는 `--keep-allocator-cache` 한 가지. 켜면 모드 사이에 `empty_cache`를 하지 않음
- 러너: `--isolation hybrid`(기본값) 추가
  - 그룹 1: `performance, memory, tokens, components`를 한 컨테이너에서 실행(모델 재사용, allocator 캐시 유지)
  - 그룹 2: `first_logits`는 legacy 워커로 따로 실행(새 프로세스)
  - manifest에 `runner_version: 3`, `mode_groups`, `allocator_cache_policy: retain_within_case` 기록
- 바꾼 이유(`START_HERE.txt`): v2 shared-case에서 first_logits가 느려지고 GPT-2 memory peak가 기존보다 낮게 나옴 → first_logits만 분리하고 캐시를 유지해 기존 피크를 재현하려는 의도

### 3.12 `compare_results.py`
- 역할: reference 폴더(r0)와 candidate 폴더를 (조건, 모드) 단위로 비교
- 허용 범위: latency/처리량 ±5%, TTFT/TPOT ±10%, allocated ±1%(최소 1MiB), reserved/NVML ±5%(최소 32MiB), storage·KV 바이트는 정확히 일치해야 함
- 함께 표시하는 issue: environment 키 불일치, `iteration_count_changed`, scope/shape 변경, NVML 부분 수집, thermal slowdown, run 내부 CV > 5%
- 출력: candidate 폴더의 `comparison.csv`, `comparison-summary.json`. 스스로 "회귀 점검용이며 동등성 증명 아님"이라고 명시

### 3.13 `test_protocol.py`
- CPU 전용 unittest. 342조건 유지, pilot이 모든 모델·dtype을 포함하는지, 측정 함수 AST가 원본과 같은지, 재사용 시 RNG/config 복원·조건 간 재사용 금지, 모드 순서와 factory 1회 호출, 온도 대기 비활성화, hybrid의 first_logits 분리, hybrid 캐시 유지, 비교기 regression 검출
- `LOCAL_CHECKS.txt`: v2 시점 7개 통과, "GPU validation: NOT RUN". `START_HERE.txt`: v3 시점 9개 통과

---

## 4. 결과 요약

### 4.1 초기 벤치마크 (`resnet50_benchmark.json`, `yolov5_benchmark.json`)

| 모델 | batch | GFLOPs | 평균 ms | peak MB | img/s |
|---|---|---|---|---|---|
| ResNet-50 | 1 | 8.18 | 7.098 | 197.35 | 140.9 |
| ResNet-50 | 32 | 261.71 | 21.179 | 2744.29 | 1510.9 |
| ResNet-50 | 64 | 523.42 | 39.809 | 5374.66 | 1607.7 |
| YOLOv5s | 1 | 24.0 | 8.994 | 64.84 | 111.2 |
| YOLOv5s | 8 | 191.98 | 15.243 | 272.66 | 524.8 |
| YOLOv5s | 32 | 767.93 | 55.216 | 985.16 | 579.5 |

- ResNet-50 학습(batch 32): 65.758 ms/step, peak 2824.57 MB
- YOLOv5s 학습(batch 8): 실패. `"'dict' object has no attribute 'mean'"`(더미 손실이 dict 출력을 처리하지 못함)
- OOM 없음

### 4.2 14B OOM 테스트
- `strict_14b.json`: **LOAD_OOM**. 136 MiB를 더 할당하다 실패. 프로세스가 23.43 GiB 사용 중, PyTorch 할당 23.17 GiB
- `offload_14b.json`: **OFFLOAD_OK**
  - weights 27.51 GiB, 로드 16.06 s, peak 21.9 GiB
  - 생성 속도 **0.57 tok/s**
  - `hf_device_map` 항목 GPU 38 / CPU 14
- 결론(파일 기준): FP16 14B는 24GB에 그대로 들어가지 않고, offload로는 돌아가지만 매우 느림

### 4.3 v1 core 342 (`cloudgpufit-bench/analysis_summary.json`, 대상 `20261006T083514Z_e0e9d733`)
- 행 1,710 = ok 1,548 + not_applicable 162. 소요 17.83 h
- 순수 측정 구간은 전체 시간의 47.6%(30,546 s). 나머지(로드, warmup, 컨테이너 등)는 분리 불가
- 모델별(조건별 performance p90 지연과 NVML peak의 중앙값 / 최댓값)

| 모델 | 조건 | p90 지연 중앙값 ms | p90 최대 ms | NVML peak min / 중앙값 / max MiB |
|---|---|---|---|---|
| gpt2 | 42 | 656.9 | 2564.2 | 798 / 990 / 3182 |
| gpt2-medium | 42 | 1213.7 | 4803.4 | 1322 / 1568 / 4880 |
| gpt2-xl | 42 | 2398.4 | 9052.6 | 3800 / 4214 / 11930 |
| mobilenet_v2 | 33 | 8.7 | 10.2 | 342 / 380 / 746 |
| resnet-50 | 33 | 7.4 | 19.8 | 398 / 454 / 682 |
| vit-large | 15 | 22.9 | 134.2 | 938 / 1016 / 1768 |
| t5-small | 45 | 682.0 | 2669.9 | 562 / 594 / 874 |
| t5-base | 45 | 1232.6 | 4966.7 | 890 / 1012 / 1662 |
| t5-3b | 45 | 2416.0 | 9613.3 | 5950 / 7534 / 13060 |

- 품질: NVML 수집 342/342 ok, 음수 outside 샘플 0, 음수 unclassified 스냅샷 0
- performance CV 중앙값 0.44%, 최대 14.1%(`core__vision_cls__medium__fp32__06`). 상위 10개가 모두 vision
- 가장 큰 NVML peak: 13,060 MiB(`core__text_encdec__large__fp32__13`) → 24GB의 85% 가드에 걸린 조건 없음
- 측정 종료 시 GPU 온도 29~65°C

### 4.4 v2 / v3 검증 (baseline9, `START_HERE.txt` + 결과 summary.csv)
- v2 대조(2조건 × 3모드): 모델 공유 110.48 s, 모드 분리 132.59 s → **약 16.7% 단축**
- 아래 값은 각 결과 폴더의 `summary.csv`에서 직접 읽어 확인

| 조건(fp16 `__01`) | memory NVML peak MiB: v1 / v2 / v3 / v3 축소 | first_logits p90 ms: v1 / v2 / v3 / v3 축소 |
|---|---|---|
| gpt2 | 804 / **698** / 804 / 804 | 9.9 / 11.9 / 10.5 / 9.8 |
| gpt2-medium | 1352 / **1196** / 1352 / 1352 | 18.7 / 22.5 / 18.7 / 19.5 |
| gpt2-xl | 3900 / **3660** / 3900 / 3900 | 36.8 / 37.8 / 36.4 / 37.8 |
| t5-small | 588 / 588 / 588 / 588 | 17.4 / 20.4 / 16.7 / 17.1 |
| t5-base | 1002 / 1002 / 1002 / 1002 | 32.3 / 38.1 / 30.8 / 31.1 |
| t5-3b | 7508 / 7508 / 7508 / 7508 | 63.3 / 64.6 / 62.6 / 64.1 |
| mobilenet / resnet / vit-l | 344 / 400 / 938(v2만 vit 942) | 해당 없음 |

- 해석
  - v2 shared-case: GPT-2 계열 memory peak가 기존보다 낮아짐. first_logits는 느려짐
  - v3 hybrid: memory peak가 v1과 같아지고 first_logits 차이도 작아짐. 반복 수를 줄인 run(`00e0fb68`)도 memory peak가 같음
- baseline9 소요: v2 728 s, v3 762 s, v3 축소 271 s
- v2에서 first_logits가 느려진 원인은 `START_HERE.txt`에 "아직 확정되지 않음"으로 적혀 있음

### 4.5 all342 (v3 hybrid + 반복 축소, `inugpu-run-manifest.json`, `inugpu-summary.csv`)
- `status: complete`, 342조건, ok 1,548 / not_applicable 162. 소요 **13,301 s(3.69 h)**, v1(17.83 h)의 약 21%
- 컨테이너: 조건마다 2개(공유 그룹 + first_logits). vision의 first_logits는 컨테이너를 띄우지 않음(0 s)
- v1(`e0e9d733`)과 비교. 두 `summary.csv`를 이 문서 작성 중 직접 비교한 결과:

| 지표 | 비교 수 | v1과 완전히 같음 | 절대 차이 중앙값 | 최대 | >5% |
|---|---|---|---|---|---|
| memory NVML peak | 342 | 326 | 0% | 5.51% | 4 |
| memory torch reserved peak | 342 | 326 | 0% | 7.35% | 8 |
| weight storage | 342 | 342 | 0% | 0% | 0 |
| self KV snapshot peak | 342 | 342 | 0% | 0% | 0 |
| performance p90 지연 | 342 | 0 | 1.67% | 39.7% | 27 |
| first_logits p90 | 261 | 0 | 1.42% | 32.8% | 41 |
| TTFT p90 | 261 | 0 | 1.25% | 29.5% | 40 |
| TPOT p90 | 261 | 0 | 1.79% | 26.9% | 21 |

- NVML peak가 2% 넘게 다른 8조건은 모두 GPT-2 small/medium의 fp16·bf16이고 전부 all342 쪽이 더 큼
  - 예: `decoder_llm__small__fp16__03` 992 → 1042, `decoder_llm__medium__fp16__12` 2686 → 2834 MiB
- thermal slowdown(측정 전후 경계): 확인 가능한 1,287건 모두 False. first_logits 261건은 legacy 워커라 기록 없음

### 4.6 recheck6 (`results/20261027T045157Z_9801141c`, `recheck6_20261027_135157.log`)
- 대상: all342에서 performance p90 증가율이 가장 컸던 6조건(상위 1~6위와 같음. 선정 기준을 적은 문서는 없음)
- 원래 v1 코드와 원래 프로토콜(warmup 3, 본측정 20, 36°C 대기)로 재측정. 로그상 30행 모두 ok / not_applicable

| 조건 | performance p90 ms: v1 / all342 / recheck6 | NVML peak (3회 모두) |
|---|---|---|
| `text_encdec__small__fp16__14` | 186.3 / 260.2 / 191.2 | 594 |
| `decoder_llm__small__fp32__12` | 1344.0 / 1799.9 / 1397.6 | 3182 |
| `vision_cls__medium__fp32__11` | 6.47 / 7.26 / 6.32 | 474 |
| `vision_cls__medium__bf16__02` | 7.49 / 8.33 / 7.25 | 400 |
| `decoder_llm__large__fp16__05` | 2203.6 / 2446.8 / 2283.4 | 3868 |
| `vision_cls__medium__bf16__04` | 7.74 / 8.54 / 8.10 | 446 |

- 원래 프로토콜로 재측정하면 지연이 v1 값에 가까워짐. memory peak는 세 실행 모두 같음
- 이 차이가 반복 수 축소, warmup 1회, 온도 대기 해제, 모델 공유 중 어느 것 때문인지는 분리되지 않음(**미확인**)

### 4.7 실패·특이사항
- bench 전체 실행에서 OOM·런타임 오류 0건(최대 13.06 GiB)
- `20261026T073104Z_v3_70368c77`: 39번째 조건(`decoder_llm__small__bf16__12`)의 components 모드 직전 `between_modes_gpu_wait` 이벤트에서 기록이 끊김
- 실패 기록: YOLOv5 학습 스텝 오류, Qwen 14B strict LOAD_OOM

---

## 5. 결과 파일 위치

| 경로 | 내용 |
|---|---|
| `resnet50_benchmark.json`, `yolov5_benchmark.json` | 초기 batch별 FLOPs/시간/메모리, 학습 스텝 |
| `yolov5su.pt` | YOLOv5 가중치(ultralytics 다운로드로 추정) |
| `strict_14b.json`, `offload_14b.json` | Qwen2.5-14B 판정 결과 |
| `_offload/` | offload 폴더(비어 있음) |
| `cloudgpufit-model-revisions.json` | 9개 모델 HF revision. `cloudgpufit-bench/hf-cache/benchmark-revisions.json`과 같음(11-01, 같은 날 만든 `cloudgpufit-image.tar.gz`·`cloudgpufit-cache.tar.gz`와 함께 내보낸 것으로 추정) |
| `cloudgpufit-bench/benchmark_plan.json` (= `benchmark_plan_v2.json`) | smoke 9 + core 342 조건 정의 |
| `cloudgpufit-bench/results/<run>/` | 실행별 폴더: `<case>__<mode>__r0.json`(원시 샘플 포함), `summary.csv`, `index.json`, `run-manifest.json`, `selected-plan.json`, `revisions.json`, `environment.lock.txt`. v2/v3는 추가로 `events.jsonl`, `host-timings.json`, `case-timings.json` |
| `cloudgpufit-bench/results/gpt2-smoke.json` | 컨테이너 GPU 동작 확인(GPT-2 fp16, 192.2 ms, peak reserved 314 MiB) |
| `cloudgpufit-bench/analysis_summary.json` | v1 core 342 실행 분석(시간 분해, 모델별 통계, CV, 온도) |
| `cloudgpufit-bench/inugpu-summary.csv`, `inugpu-run-manifest.json` | all342(`20261026T082509Z_v3_e8c4d64f`)의 summary/manifest를 꺼낸 사본(원본과 같음) |
| `cloudgpufit-bench/recheck6-original-plan.json`, `recheck6_20261027_135157.log`, `LATEST_RECHECK6_LOG.txt` | recheck6 계획, 실행 로그, 로그 경로 포인터 |
| `cloudgpufit-bench/LOCAL_README.md`, `LOCAL_CHECKS.txt`, `LOCAL_SHA256.json` | v2 배포 안내, CPU 테스트 결과, 파일 해시 |
| `cloudgpufit-bench/START_HERE.txt`, `SHA256.json` | v3 배포 안내 + v2 대조 결과 요약, 파일 해시 |
| `cloudgpufit-bench/cloudgpufit-bench-local.zip` / `-v3.zip` | v2 / v3 배포 패키지(코드 + 문서) |
| `benchmark-review-20261006.tar.gz` | v1 core 342 결과 + `run_bench.py`, `bench_worker.py`, `benchmark_plan.json` |
| `benchmark-local-results.tar.gz` | v2 baseline9(`f88a5b92`) + v2 코드 |
| `benchmark-comparison.tar.gz` | v2 shared vs separate 대조(`a048565d`, `2932ef9a`) |
| `benchmark-v3.tar.gz` / `benchmark-v3-2.tar.gz` | v3 baseline9(`b82c33c4`) / v3 반복 축소(`00e0fb68`) |
| `inugpu-all342-results.tar.gz` | all342 결과 폴더(경로에 `results/` 접두사 없음) |
| `recheck6-20261027-results.tar.gz` | recheck6 결과 + 계획 + v1 코드 |

(위 tar.gz는 모두 `cloudgpufit-bench/` 아래에 있음)

---

## 6. 한계 / 미확인

- **미확인**: AWS(T4/A10G/L4/L40S)에서의 342조건 실행. 이 디렉터리에는 RTX 3090 결과만 있음. `aws_core` 프로파일은 정의만 있음
- **미확인**: `compare_results.py`의 출력(`comparison.csv`, `comparison-summary.json`)이 어느 결과 폴더에도 없음. 비교를 다른 곳에서 했거나 결과를 남기지 않은 것. 4.4~4.6의 비교 수치는 summary.csv를 직접 읽어 확인한 값
- **미확인**: v1 core 342와 v2 baseline9 실행 명령 원문(히스토리에 없음. manifest와 README로 추정)
- **미확인**: 초기 ResNet/YOLO, 14B 테스트의 실행 명령과 환경. CUDA 11.8 venv 사용은 추정
- **미확인**: 첫 all342 시도(`70368c77`)의 중단 원인
- **미확인**: recheck6 대상 선정 기준 문서. 데이터상으로는 all342 performance 증가율 상위 6개와 같음
- **미확인**: v2 shared-case에서 first_logits 지연 증가와 memory peak 감소가 생긴 정확한 원인(`START_HERE.txt`도 미확정이라고 적음)
- 측정 한계(코드 기준)
  - NVML은 50 ms 간격 샘플링이라 순간 피크를 놓칠 수 있음
  - 온도·클럭은 측정 전후 경계에서만 기록
  - local_core는 repeat 1회, 독립 반복 없음
  - all342는 반복 수를 줄였으므로(performance 5, memory 3, tokens 3) 지연 통계의 표본이 v1(20)보다 적음
- 초기 벤치마크는 무작위 가중치(ResNet), autograd가 켜진 상태의 추론 측정, `max_memory_allocated`만 기록(NVML 없음)이라 bench 결과와 직접 비교할 수 없음
- `image-id.txt`(`67245a…`)와 실제 실행 이미지 ID(`457da255…`)가 다름. 재빌드 경위는 히스토리 순서로 추정

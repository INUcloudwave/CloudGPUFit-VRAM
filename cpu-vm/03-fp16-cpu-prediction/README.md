# CloudGPUFit FP16 VRAM 재실험 — CPU 요청 추적 + xMem 예측 (inuos)

- 목적: RTX 3090·AWS GPU 실측(warmup 3 → 측정 5)과 같은 조건에서 CPU 텐서 요청을 추적해 xMem으로 GPU allocated/reserved 예측
- 비교 대상: GPU **첫 번째 측정 회차**. GPU 5회 최댓값은 용량 참고값으로 별도 보관
- 기존 코드·결과 미수정. 이 저장소만 사용 (git 브랜치 `exp/fp16-680-cpu-20261111`)

## 2026-11-15 L256 조건표 전환 (현재)
- GPU(INUGPU) 조건표가 `conditions_main395_l256.json`(d4a9b09c…)으로 바뀜. 측정 코드는 그대로
  - 정렬표: `handoff_codex/gpu_cpu_alignment.md`
- 이전 표 실행 `runs/main395`는 사용자 요청으로 중단·보존 (`archive/main395_cpu_L1024_stopped_20261115.tar.gz`)
- 새 실행 `runs/main395_l256`
  - 겹치는 244개: `tools/link_overlap.py`로 기존 결과를 심볼릭 링크로 재사용 (`reuse_manifest.json`)
  - 새로 돌리는 151개: 새 150개 + 이전에 실행하지 않은 1개
  - CPU worker(`cpu_worker2.py`)는 수정하지 않음
- 실행기 `tools/run_l256.py`
  - P단계: 3개 병렬. cpuset 분리(0-6 / 7-13 / 14-20, 7스레드), 조건마다 새 컨테이너
  - 메모리 감시: 호스트 가용 메모리가 5GiB 미만이고 2개 이상 실행 중이면 가장 늦게 시작한 작업을 종료(`preempted_low_mem`)
  - S단계: P단계가 모두 끝난 뒤, 메모리 실패·시간 초과 작업을 순차 실행(0-19, 20스레드, 제한 없음). S단계 결과로 확정
- 스레드 영향 확인: `docs/run_thread_check.sh` → `results/thread_check/compare.jsonl`
- 용량 재생(후처리): `tools/replay_capacity.py`
  - GPU 메모리 크기 설정: `manifest/gpu_profiles.json`
  - 결과: `results/<run>/capacity_<gpu>.jsonl`
  - 1TiB로 재생하면 worker 결과와 일치함을 확인 (5/5)
- 비교
  - `tools/collect395.py --run-name main395_l256`
  - `tools/compare_gpu_l256.py --gpu-csv <INUGPU 조건별 CSV> --capacity rtx3090`
  - `tools/check_gpu_overlap.py` (GPU L1024 ↔ L256의 겹치는 245개 동일성)
```bash
cd /home/inuos/cgf-fp16-680-20261111
python3 src/cgf680/cases395.py --conditions conditions_main395_l256.json --out cases395_l256.jsonl --expect-sha d4a9b09caff3064c00c04358bf2e462996bced23de93da98652bd1aba04b7289
python3 tools/link_overlap.py --src main395 --src-cases manifest/cases395.jsonl --dst main395_l256 --dst-cases manifest/cases395_l256.jsonl
tmux new-session -d -s cgf-l256 "bash docs/run_main395_l256.sh"      # 재실행 = 재개
docs/run_in_image.sh tools/replay_capacity.py --run main395_l256 --gpu rtx3090 --jobs 4
```

## 구조
| 경로 | 내용 |
|---|---|
| `manifest/cases.jsonl`, `.csv` | 공통 680개 조건표 (case_id = GPU 조인 키, `pilot` 26개 표시). `src/cgf680/cases.py`로 생성, sha256 `3f94c510…` |
| `manifest/model_pins.json` | 13개 모델 pretrained revision·가중치 파일 (기존 9개는 aws342 pin과 동일) |
| `src/cgf680/protocol.py` | **장치 공통 프로토콜**: 모델 로딩, 입력 생성, generate 인자, 추론 1회. GPU runner도 이 함수 사용 권장 |
| `src/cgf680/cpu_worker.py` | 케이스 1개 = 새 프로세스. profiler 세션 1개로 warmup 3 + 측정 1 추적 → 정규화 → xMem 순서 재생 |
| `tools/run_suite.py` | 호스트 실행기: 케이스별 새 컨테이너, RAM 한도·swap 없음·제한 시간, 재개, 실패 분류 |
| `tools/collect.py` | 케이스 결과 → `results/<run>/cpu_results.{csv,jsonl}`, `summary.json` |
| `tools/compare_gpu.py` | GPU 결과 조인(case_id·revision·입력 해시·프로토콜 대조) → bytes 차이·상대오차 |
| `tools/compare_events.py` | 두 실행의 요청 trace 완전 일치 검사 |
| `runs/<run>/cases/<case_id>/` (git 제외) | `result.json`, `events.npz`(정규화 요청), `mem.jsonl`(RSS), `container.log`, `runner.json`, `progress.json` |

## 재현·재개 명령 (tmux, 순차 실행)
```bash
cd /home/inuos/cgf-fp16-680-20261111
PYTHONPATH=src python3 -m cgf680.cases --out manifest/cases.jsonl          # 조건표 재생성(동일 sha 확인)
docs/download.sh                                                             # pinned 가중치 (네트워크 필요)
tmux new-session -d -s cgf680-pilot "bash docs/run_pilot.sh"                 # 파일럿 26개 (재실행 = 재개)
tmux new-session -d -s cgf680-main  "bash docs/run_main.sh"                  # 본실험 680개 (재실행 = 재개)
python3 tools/collect.py --run-name main
python3 tools/compare_gpu.py --cpu results/main/cpu_results.jsonl --gpu <GPU결과> --gpu-label rtx3090 --out results/main/compare_rtx3090.csv
```
- 재개: 같은 `--run-name`으로 다시 실행하면 최종 상태(ok, nonfinite_output, fp16_unsupported, input_mismatch, unsupported_shape) 케이스는 건너뜀. cpu_ram_oom·timeout·error는 `--retry-failed` 때만 재시도(이전 시도 폴더는 `.attempt-<t>`로 보존)
- 단일 케이스: `python3 tools/run_suite.py --run-name X --select ids --ids <case_id> --conv1d-colmajor`

## CPU 프로토콜 (케이스당)
1. 새 컨테이너·새 프로세스, `--network none`, `--memory 42g --memory-swap 42g`, `--cpus 20`, torch threads 20
2. `from_pretrained(revision, torch_dtype=fp16, low_cpu_mem_usage, attn_implementation="eager")` → `eval()`
3. 입력: host Generator seed 20261006. 텍스트 `randint(3, vocab, (B,L))` int64, pad/eos/bos id → 3, BOS 추가·truncation 없음, mask int64 1. 이미지 `randn(B,3,R,R)` float32 → fp16 (이미지 프로세서 미사용 → 224로 리사이즈 없음). 장치에 1회 배치 후 모든 회차 동일 텐서 사용
4. generate: `do_sample=False, num_beams=1, use_cache=True, min_new_tokens=max_new_tokens=N, pad_token_id=pad or eos`. 그 외는 pinned `generation_config.json` 기본값(결과에 해석값 기록). T5 신규 생성 수 = 출력 길이 − 1(decoder start 제외)
5. profiler 세션 1개: warmup 0·1·2, measure 0 각각 `record_function("cgf680::<j>::<phase>::<i>")` 구간. 회차 사이 `del output`, 자동 GC 비활성(세션 전 `gc.collect`)
6. 경계 표식: 구간 밖 고유 크기 uint8 할당(M0~M5)으로 이벤트 순서상 정확한 위치 확보
7. 재생: 상주(파라미터·버퍼 고유 storage → 입력) malloc 후 요청을 Ev Idx 순서 그대로 xMem `CachingAllocator`(lock 커밋 1ee4bd08, 무수정)에 적용. warmup이 만든 캐시 유지, M3부터 피크 계산
8. 저장: 시작(M3)·피크(M3~M5)·종료(M4: 출력 참조 유지) + 출력 해제 후(M5) allocated/reserved 정수 bytes

## 검증된 CPU 전용 최적화 (요청 trace 불변)
| 항목 | 내용 | 근거 |
|---|---|---|
| profiler RecordFunction 범위 `USER_SCOPE` | aten op 이벤트 미기록, `[memory]` 이벤트·구간 표식만 기록 | opt-125m·gpt2 파일럿 조건에서 전체 범위 대비 이벤트 42만·54만 건 순서·크기·짝·예측값 완전 일치. RSS 16→1.4 GiB, 20→1.9 GiB |
| GPT-2 계열 Conv1D 가중치 column-major stride | shape·bytes 동일, stride만 변경 → FP16 addmm 빠른 경로 | gpt2 smoke 2조건 완전 일치, 12배 단축 (파일럿 크기 검증은 `runs/verify-layout`) |

## 알려진 차이·한계 (결과 해석 시 필수)
- 이 VM CPU(Cascade Lake)는 FP16 연산 하드웨어 미지원: FP16 bmm ~1.8 GFLOPS(스레드 확장 없음), conv ~10 GFLOPS, 일부 matmul 0.3 GFLOPS → 큰 조건은 실행 시간이 매우 김
- GPU 시작 상태는 `from_pretrained→cuda` 로딩 이력(임시 버퍼·캐시)을 포함, CPU는 빈 allocator에 상주 storage를 순서대로 malloc → segment 배치 차이 가능 (결과 `resident.difference_vs_gpu_start`)
- 출력 finite 검사: 매 forward logits를 numpy view로 검사(c10 allocator 미경유 → 추적 요청 추가 없음). GPU 쪽 검사 방식은 별도 확인 필요
- GPU 전용 요청(cuBLAS workspace 8.125 MiB/handle·stream, cuBLASLt 1 MiB 임시, cuDNN workspace)은 CPU trace에 없음. CPU 전용 요청(스칼라 등)은 GPU에 없음
- xMem 용량 1 TiB(압박 없음)로 재생. GPU OOM은 별도 분류
- CPU 예측값으로 NVML 프로세스 사용량·Outside 값을 채우지 않음

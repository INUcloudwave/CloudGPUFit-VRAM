# CloudGPUFit reserved 오차 원인 분리 실험 (inuos)

- 목적: CPU 예측(xMem) reserved와 GPU 실측 reserved 차이를 (a) 실행 이력/초기 캐시, (b) allocator 정책 모사, (c) GPU 전용 요청으로 분리
- 운영 예측 코드·기존 결과는 수정하지 않음. 이 디렉터리만 사용 (git 브랜치 `diag/reserved-error-20261110`)

## 구조
| 경로 | 내용 |
|---|---|
| `provenance/ENV_AND_PATH_REPORT.md` | 실제 실행 경로, 버전, 이전 설명과의 일치/불일치 |
| `provenance/pytorch_v2.6.0_ref/` | 대조에 쓴 PyTorch v2.6.0 원문 (SHA256SUMS) |
| `manifest/common_manifest.json` | 3 case + 실행 구성(R0/R1/R2/D1) 공통 정의 (`src/cgf_diag/build_manifest.py`로 생성) |
| `src/cgf_diag/events.py` | 정규화 이벤트 스키마, CPU chrome trace / GPU snapshot importer (안전 unpickler) |
| `src/cgf_diag/replay.py` | 순서 보존 xMem 재생기, 운영 방식(시간 그룹) 재현, 10MiB 경계 진단 정책 |
| `src/cgf_diag/cpu_history_trace.py` | GPU worker와 같은 호출 순서의 CPU 요청 추적 (호출 단위 profiler 창) |
| `src/cgf_diag/analyze_cpu.py` | 기준선 재현(A1~A3), 이력 prefix 재생 |
| `src/cgf_diag/gpu_replay.py` | GPU 요청 재생·결정 비교·최초 불일치·CPU 요청 대조·hybrid 재생 |
| `src/cgf_diag/summarize.py` | (1) 기준선 (2) 이력 CPU (3) GPU 재생/실측 나란히 요약 |
| `tests/test_replayer_synthetic.py` | 합성 검증 8종 (결과: `results/synthetic/`) |
| `handoff/` | inugpu 인계: 요청 문서, 참조 수집기, manifest, lock |
| `runs/` (git 제외) | 원본 trace·로그 (`baseline/`, `cpu_history/`) |
| `results/` | 분석 JSON, 요약 |

## 재현 명령 (모두 tmux 안에서, jobs=1)
```bash
cd /home/inuos/cgf-reserved-diag-20261110
docs/run_in_image.sh tests/test_replayer_synthetic.py results/synthetic/synthetic_validation.json   # 합성 검증
tmux new-session -d -s cgf-diag "bash docs/run_cpu_stage.sh"     # CPU 단계 전체 (DONE 표식 있으면 건너뜀)
docs/run_in_image.sh -m cgf_diag.summarize --out /exp/results/SUMMARY_cpu.md
```
- 실행 이미지 `cloudgpufit-cpu-validation:20261108`, xMem `/home/inuos/xmem-work/xMem`(lock 검증), predictor `/home/inuos/CloudGPUFit-cpu-singlepass-20261108` 읽기 전용 mount

## GPU 자료 도착 후
```bash
tmux new-session -d -s cgf-gpu "bash docs/run_gpu_import.sh /path/to/return_root"
# 결과: results/gpu/<case>/<run_id>/import_report.json, results/SUMMARY_after_gpu.md
```
- 해석 순서
  1. `import.initial_state == empty`, `trace_complete == true` 확인 (아니면 전체 이력 비교라고 부르지 않음)
  2. `policies.xmem_pinned.comparison`: 같은 GPU 요청에서 xMem 결정이 실제와 갈라지는 최초 요청과 전후 상태
  3. `request_diff_vs_cpu`: phase별 GPU 전용/CPU 전용 요청 크기 (workspace 후보는 frames 확인)
  4. `hybrid_gpu_prefix_then_cpu_requests`: GPU 로딩 이력 + CPU 요청 → 로딩 이력 기여분
- 용량 압박/OOM 검증은 별도: `--capacity-bytes <GPU 총용량 − 비allocator 점유>`로 재실행, 산정 근거를 함께 기록

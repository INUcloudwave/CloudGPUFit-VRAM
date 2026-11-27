# prediction/local-cpu-vm/ — INUOS(CPU 예측) 코드·결과·문서

- 보고서 작성용 통합 문서(INUGPU·INUOS·AWS): `../REPORT_CONTEXT_KO.md` (먼저 읽기)
- 세부: `../AI_CONTEXT_KO.md` (INUGPU 쪽, 11-17까지) → `INUOS_CONTEXT_KO.md` (INUOS 최신 상태) → `../aws/README_KO.md` (AWS 결과)
- 실험별 상세 문서: `docs/`

| 경로 | 내용 | 원본 위치 (INUOS VM) | 상세 |
|---|---|---|---|
| `INUOS_CONTEXT_KO.md` | INUOS 쪽 맥락·현재 상태·열린 문제 | — | — |
| `docs/01_cpu_predictor_342_KO.md` | 10-26 ~ 11-10: 휴리스틱 → native v3 → recovery → singlepass v4, 342조건 CPU 예측 | — | — |
| `docs/02_reserved_diag_KO.md` | 11-10 ~ 11-11: reserved 오차 원인 분리 (실행 이력) | — | — |
| `docs/03_fp16_cpu_prediction_KO.md` | 11-11 ~ : FP16 CPU 예측 (680/395, RTX 3090 대조, L256 전환) | — | — |
| `00-early-eval-recovery/cloudgpufit-eval/` | baseline9 (10-26, v2 휴리스틱) 실행 결과 | `/home/inuos/cloudgpufit-eval` | docs/01 §3.1 |
| `00-early-eval-recovery/CloudGPUFit_CPU_recovery_20261108/` | 11-08 복구 패키지 (diagnosis, fix.patch) | `/home/inuos/CloudGPUFit_CPU_recovery_20261108` | docs/01 §3.5 |
| `00-early-eval-recovery/CloudGPUFit_Codex_handoff_20261108.md` | 11-08 인수인계 문서 | `/home/inuos/` | docs/01 |
| `01-cpu-predictor-342/predictor-singlepass-20261108/` | 342 실행에 쓴 예측기 VRAM 관련 파일 + `uncommitted_vs_7f822d7.diff` (기준 커밋 7f822d7, 원격 `codex/vram-native-inference-20261103`) | `/home/inuos/CloudGPUFit-cpu-singlepass-20261108` | docs/01 §3.7, §4 |
| `01-cpu-predictor-342/CloudGPUFit_CPU_predictions_20261110/` | 342조건 CPU 예측 결과 패키지 (325 ok / 17 실패) | `/home/inuos/CloudGPUFit_CPU_predictions_20261110` | docs/01 §3.10 |
| `01-cpu-predictor-342/cloudgpufit-predictions-runs/` | 예측 실행 원본 (native-v3, recovery, singlepass, parallel, all342, oom-unlimited) | `/home/inuos/cloudgpufit-predictions` | docs/01 §3.3~3.11 |
| `02-reserved-diag/` | reserved 오차 진단 저장소 전체 (코드, `runs/` 원시 trace, `inputs/` GPU 반환·L4 결과, `results/`) | `/home/inuos/cgf-reserved-diag-20261110` | docs/02 |
| `03-fp16-cpu-prediction/` | FP16 CPU 예측 저장소 전체 (코드, `runs/` 조건별 결과·요청 trace npz·로그, `results/`, `handoff_codex/`) | `/home/inuos/cgf-fp16-680-20261111` | docs/03 |
| `03-fp16-cpu-prediction/review-20261112/` | 11-12 INUOS→INUGPU 교차 검토 | `/home/inuos/cloudgpufit-review` | docs/03 §2.15 |
| `03-fp16-cpu-prediction/inputs_from_inugpu/` | INUGPU가 보낸 handoff 2종 (11-12, 11-15) | `/home/inuos/cgf-fp16-inuos-handoff-2026111{2,5}` | docs/03 §2.9, §2.15 |
| `00-early-eval-recovery/cloudgpufit-cpu-diag-*/` | 11-05/11-08 진단 수집 (kernel.log·docker inspect 제외) | `/home/inuos/cloudgpufit-cpu-diag-*` | docs/01 §3.3 |
| `00-early-eval-recovery/predictor-recovery-20261108/` | recovery worktree 변경·추가 파일 + diff | `/home/inuos/CloudGPUFit-cpu-recovery-20261108` | docs/01 §3.4 |
| `01-cpu-predictor-342/cloudgpufit-comparison/` | AWS 실측 tarball 6개 (T4, A10G, L4×3, L40S), 비교용 사본 | `/home/inuos/cloudgpufit-comparison` | docs/01 §3.12 |
| `EXCLUDED_LARGE_FILES.md` | 레포에서 뺀 파일 목록 (50MB 초과는 sha256 포함) | — | — |

# 환경 비교 요약 (3개 조건)

| env | label | driver | torch | cuDNN | ok | gpu_oom | 기타 |
|---|---|---|---|---|---|---|---|
| A | runs/rtx3090__envcmp_smoke_A | 570.211.01 | 2.6.0+cu124 | 90100 | 3 | 0 | 0 |
| C | runs/rtx3090__envcmp_smoke_C | 570.211.01 | 2.5.1+cu124 | 90100 | 2 | 0 | 1 |
| REF | runs/rtx3090__main395_l256 | 570.211.01 | 2.6.0+cu124 | 90100 | 3 | 0 | 0 |

## C − A (MiB, 양쪽 값이 있는 조건만)

| 지표 | n | 0인 조건 | 최소 | 중앙값 | 최대 |
|---|---|---|---|---|---|
| first_peak_allocated | 2 | 2 | +0.0 | +0.0 | +0.0 |
| first_peak_reserved | 2 | 2 | +0.0 | +0.0 | +0.0 |
| max_nvml_process_peak | 2 | 2 | +0.0 | +0.0 | +0.0 |
| cuda_init_outside | 3 | 3 | +0.0 | +0.0 | +0.0 |
| after_load_outside | 2 | 2 | +0.0 | +0.0 | +0.0 |
| after_warmup_outside | 2 | 2 | +0.0 | +0.0 | +0.0 |
| after_warmup_reserved | 2 | 2 | +0.0 | +0.0 | +0.0 |
| after_warmup_nvml_process_used | 2 | 2 | +0.0 | +0.0 | +0.0 |
| num_alloc_retries | 3 | 3 | +0.0 | +0.0 | +0.0 |

눈에 띈 조건 (1개, `flagged_C_vs_A.json`):

- `opt-125m__b1__in128__new128`: status ok->compat_or_runtime_error

## REF − A (MiB, 양쪽 값이 있는 조건만)

| 지표 | n | 0인 조건 | 최소 | 중앙값 | 최대 |
|---|---|---|---|---|---|
| first_peak_allocated | 3 | 3 | +0.0 | +0.0 | +0.0 |
| first_peak_reserved | 3 | 3 | +0.0 | +0.0 | +0.0 |
| max_nvml_process_peak | 3 | 3 | +0.0 | +0.0 | +0.0 |
| cuda_init_outside | 0 | | | | |
| after_load_outside | 0 | | | | |
| after_warmup_outside | 0 | | | | |
| after_warmup_reserved | 0 | | | | |
| after_warmup_nvml_process_used | 0 | | | | |
| num_alloc_retries | 3 | 3 | +0.0 | +0.0 | +0.0 |

눈에 띈 조건 (0개, `flagged_REF_vs_A.json`):

- 없음

## 값 누락 (둘 다 ok인데 지표가 없음)

- REF opt-125m__b1__in128__new128 after_warmup_outside
- REF resnet-50__b16__r512 after_warmup_outside
- REF tinyllama-1.1b-chat__b1__in128__new128 after_warmup_outside

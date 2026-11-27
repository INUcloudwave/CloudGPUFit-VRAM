# 환경 비교 요약 (39개 조건)

| env | label | driver | torch | cuDNN | ok | gpu_oom | 기타 |
|---|---|---|---|---|---|---|---|
| A | runs/aws-g4dn.xlarge__envcmp_A | 595.112.07 | 2.6.0+cu124 | 90100 | 39 | 0 | 0 |
| C | runs/aws-g4dn.xlarge__envcmp_C | 595.112.07 | 2.5.1+cu124 | 90100 | 33 | 0 | 6 |
| B | runs/aws-g4dn.xlarge__envcmp_B | 580.196.07 | 2.6.0+cu124 | 90100 | 39 | 0 | 0 |
| REF | runs/aws-g4dn.xlarge__l256 | 595.112.07 | 2.6.0+cu124 | 90100 | 39 | 0 | 0 |

## C − A (MiB, 양쪽 값이 있는 조건만)

| 지표 | n | 0인 조건 | 최소 | 중앙값 | 최대 |
|---|---|---|---|---|---|
| first_peak_allocated | 33 | 33 | +0.0 | +0.0 | +0.0 |
| first_peak_reserved | 33 | 33 | +0.0 | +0.0 | +0.0 |
| max_nvml_process_peak | 33 | 32 | -2.0 | +0.0 | +0.0 |
| cuda_init_outside | 39 | 39 | +0.0 | +0.0 | +0.0 |
| after_load_outside | 33 | 33 | +0.0 | +0.0 | +0.0 |
| after_warmup_outside | 33 | 32 | -2.0 | +0.0 | +0.0 |
| after_warmup_reserved | 33 | 33 | +0.0 | +0.0 | +0.0 |
| after_warmup_nvml_process_used | 33 | 32 | -2.0 | +0.0 | +0.0 |
| num_alloc_retries | 39 | 39 | +0.0 | +0.0 | +0.0 |

눈에 띈 조건 (6개, `flagged_C_vs_A.json`):

- `opt-125m__b1__in128__new128`: status ok->compat_or_runtime_error
- `opt-125m__b32__in256__new128`: status ok->compat_or_runtime_error
- `opt-125m__b128__in512__new256`: status ok->compat_or_runtime_error
- `vit-large-patch16-224__b1__r224`: status ok->compat_or_runtime_error
- `vit-large-patch16-224__b32__r224`: status ok->compat_or_runtime_error
- `vit-large-patch16-224__b128__r224`: status ok->compat_or_runtime_error

## B − A (MiB, 양쪽 값이 있는 조건만)

| 지표 | n | 0인 조건 | 최소 | 중앙값 | 최대 |
|---|---|---|---|---|---|
| first_peak_allocated | 39 | 39 | +0.0 | +0.0 | +0.0 |
| first_peak_reserved | 39 | 39 | +0.0 | +0.0 | +0.0 |
| max_nvml_process_peak | 39 | 36 | -2.0 | +0.0 | +0.0 |
| cuda_init_outside | 39 | 39 | +0.0 | +0.0 | +0.0 |
| after_load_outside | 39 | 39 | +0.0 | +0.0 | +0.0 |
| after_warmup_outside | 39 | 36 | -2.0 | +0.0 | +0.0 |
| after_warmup_reserved | 39 | 39 | +0.0 | +0.0 | +0.0 |
| after_warmup_nvml_process_used | 39 | 36 | -2.0 | +0.0 | +0.0 |
| num_alloc_retries | 39 | 39 | +0.0 | +0.0 | +0.0 |

눈에 띈 조건 (0개, `flagged_B_vs_A.json`):

- 없음

## REF − A (MiB, 양쪽 값이 있는 조건만)

| 지표 | n | 0인 조건 | 최소 | 중앙값 | 최대 |
|---|---|---|---|---|---|
| first_peak_allocated | 39 | 39 | +0.0 | +0.0 | +0.0 |
| first_peak_reserved | 39 | 39 | +0.0 | +0.0 | +0.0 |
| max_nvml_process_peak | 39 | 39 | +0.0 | +0.0 | +0.0 |
| cuda_init_outside | 0 | | | | |
| after_load_outside | 0 | | | | |
| after_warmup_outside | 0 | | | | |
| after_warmup_reserved | 0 | | | | |
| after_warmup_nvml_process_used | 0 | | | | |
| num_alloc_retries | 39 | 39 | +0.0 | +0.0 | +0.0 |

눈에 띈 조건 (0개, `flagged_REF_vs_A.json`):

- 없음

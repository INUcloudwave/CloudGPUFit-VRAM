# reserved 오차 진단 요약 (자동 생성)

- 단위 MiB(2^20). PRED=시뮬레이터 예측, OBS=실측. 용량 80GiB(압박 없음) 고정

## core__decoder_llm__small__fp16__01 (basic comparison)

| 항목 | reserved | allocated |
|---|---:|---:|
| 이전 L4 memory 모드 피크 reserved / allocated (OBS, 참고치) | 492.0 | 298.7 |
| 이전 L4 performance 모드: 로딩 직후 reserved / allocated (OBS) | 474.0 | 250.5 |
| (1) CPU 운영 기준선 reserved / allocated (PRED, 저장값) | 330.0 | 296.5 |
|   A1 운영 방식 재현(시간 그룹) — 저장값 일치=True | 330.0 | 296.5 |
|   A2 같은 trace 순차 재생 | 330.0 | 296.5 |
|   A3 A2 + PyTorch2.6 10MiB 경계(진단) | 330.0 | 296.5 |
| (2) CPU 이력 trace: generate 1회 단독 (GPU식 입력 복제) | 330.0 | 296.5 |
|   누적 이력: 검증 forward 1회까지 | 370.0 | 356.6 |
|   누적 이력: +warmup 1회 | 384.0 | 356.6 |
|   누적 이력: R1 통제(검증1+warmup1+측정3) | 384.0 | 356.6 |
|   누적 이력: AWS performance 모드 끝 | 384.0 | 356.6 |
|   AWS 구조 전체(performance→memory) 피크 | 398.0 | 356.6 |
|   memory 모드 측정 구간 피크 (L4 memory 피크와 대응) | 398.0 | 296.5 |
|   AWS 구조 전체, 10MiB 경계 진단 정책 | 398.0 | 356.6 |
| (3) GPU D1_cold_single_empty_cache: 실제 reserved 피크 (OBS, trace) | 474.0 | — |
|     같은 GPU 요청 → xMem (PRED); 최초 결정 불일치 요청 #5248, 불일치 6건 | 474.0 | 386.5 |
|     같은 GPU 요청 → 10MiB 경계 진단 정책; 불일치 6건 | 474.0 | 386.5 |
|     GPU 로딩 구간 + CPU 요청 (hybrid PRED) | 474.0 | 386.5 |
| (3) GPU R0_aws_replica: 실제 reserved 피크 (OBS, trace) | 492.0 | — |
|     같은 GPU 요청 → xMem (PRED); 최초 결정 불일치 요청 #5669, 불일치 6건 | 492.0 | 386.5 |
|     같은 GPU 요청 → 10MiB 경계 진단 정책; 불일치 6건 | 492.0 | 386.5 |
|     GPU 로딩 구간 + CPU 요청 (hybrid PRED) | 490.0 | 386.5 |
| (3) GPU R1_controlled: 실제 reserved 피크 (OBS, trace) | 492.0 | — |
|     같은 GPU 요청 → xMem (PRED); 최초 결정 불일치 요청 #5669, 불일치 6건 | 492.0 | 386.5 |
|     같은 GPU 요청 → 10MiB 경계 진단 정책; 불일치 6건 | 492.0 | 386.5 |
|     GPU 로딩 구간 + CPU 요청 (hybrid PRED) | 490.0 | 386.5 |
| (3) GPU R2_cold_single: 실제 reserved 피크 (OBS, trace) | 492.0 | — |
|     같은 GPU 요청 → xMem (PRED); 최초 결정 불일치 요청 #5248, 불일치 6건 | 492.0 | 386.5 |
|     같은 GPU 요청 → 10MiB 경계 진단 정책; 불일치 6건 | 492.0 | 386.5 |
|     GPU 로딩 구간 + CPU 요청 (hybrid PRED) | 490.0 | 386.5 |

## core__decoder_llm__small__fp16__12 (large reserved error)

| 항목 | reserved | allocated |
|---|---:|---:|
| 이전 L4 memory 모드 피크 reserved / allocated (OBS, 참고치) | 1678.0 | 719.7 |
| 이전 L4 performance 모드: 로딩 직후 reserved / allocated (OBS) | 474.0 | 250.5 |
| (1) CPU 운영 기준선 reserved / allocated (PRED, 저장값) | 928.0 | 717.1 |
|   A1 운영 방식 재현(시간 그룹) — 저장값 일치=True | 928.0 | 717.1 |
|   A2 같은 trace 순차 재생 | 928.0 | 717.1 |
|   A3 A2 + PyTorch2.6 10MiB 경계(진단) | 928.0 | 717.1 |
| (2) CPU 이력 trace: generate 1회 단독 (GPU식 입력 복제) | 928.0 | 717.1 |
|   누적 이력: 검증 forward 1회까지 | 1664.0 | 1450.9 |
|   누적 이력: +warmup 1회 | 1668.0 | 1450.9 |
|   누적 이력: R1 통제(검증1+warmup1+측정3) | 1668.0 | 1450.9 |
|   누적 이력: AWS performance 모드 끝 | 1668.0 | 1450.9 |
|   AWS 구조 전체(performance→memory) 피크 | 1668.0 | 1450.9 |
|   memory 모드 측정 구간 피크 (L4 memory 피크와 대응) | 1668.0 | 716.6 |
|   AWS 구조 전체, 10MiB 경계 진단 정책 | 1668.0 | 1450.9 |
| (3) GPU D1_cold_single_empty_cache: 실제 reserved 피크 (OBS, trace) | 934.0 | — |
|     같은 GPU 요청 → xMem (PRED); 최초 결정 불일치 요청 #1327, 불일치 1건 | 932.0 | 719.7 |
|     같은 GPU 요청 → 10MiB 경계 진단 정책; 불일치 1건 | 932.0 | 719.7 |
| (3) GPU R0_aws_replica: 실제 reserved 피크 (OBS, trace) | 1678.0 | — |
|     같은 GPU 요청 → xMem (PRED); 최초 결정 불일치 요청 #1784, 불일치 1건 | 1676.0 | 1453.6 |
|     같은 GPU 요청 → 10MiB 경계 진단 정책; 불일치 1건 | 1676.0 | 1453.6 |
| (3) GPU R1_controlled: 실제 reserved 피크 (OBS, trace) | 1678.0 | — |
|     같은 GPU 요청 → xMem (PRED); 최초 결정 불일치 요청 #1784, 불일치 1건 | 1676.0 | 1453.6 |
|     같은 GPU 요청 → 10MiB 경계 진단 정책; 불일치 1건 | 1676.0 | 1453.6 |
| (3) GPU R2_cold_single: 실제 reserved 피크 (OBS, trace) | 920.0 | — |
|     같은 GPU 요청 → xMem (PRED); 최초 결정 불일치 요청 #1327, 불일치 1건 | 918.0 | 719.7 |
|     같은 GPU 요청 → 10MiB 경계 진단 정책; 불일치 1건 | 918.0 | 719.7 |

## core__decoder_llm__medium__fp32__12 (workspace and large error candidate)

| 항목 | reserved | allocated |
|---|---:|---:|
| 이전 L4 memory 모드 피크 reserved / allocated (OBS, 참고치) | 4574.0 | 2850.7 |
| 이전 L4 performance 모드: 로딩 직후 reserved / allocated (OBS) | 1380.0 | 1377.6 |
| (1) CPU 운영 기준선 reserved / allocated (PRED, 저장값) | 3552.0 | 2842.6 |
|   A1 운영 방식 재현(시간 그룹) — 저장값 일치=True | 3552.0 | 2842.6 |
|   A2 같은 trace 순차 재생 | 3552.0 | 2842.6 |
|   A3 A2 + PyTorch2.6 10MiB 경계(진단) | 3552.0 | 2842.6 |
| (2) CPU 이력 trace: generate 1회 단독 (GPU식 입력 복제) | 3552.0 | 2842.6 |
|   누적 이력: 검증 forward 1회까지 | 4568.0 | 3863.5 |
|   누적 이력: +warmup 1회 | 4574.0 | 3863.5 |
|   누적 이력: R1 통제(검증1+warmup1+측정3) | 4574.0 | 3863.5 |
|   누적 이력: AWS performance 모드 끝 | 4574.0 | 3863.5 |
|   AWS 구조 전체(performance→memory) 피크 | 4574.0 | 3863.5 |
|   memory 모드 측정 구간 피크 (L4 memory 피크와 대응) | 4574.0 | 2842.6 |
|   AWS 구조 전체, 10MiB 경계 진단 정책 | 4574.0 | 3863.5 |
| (3) GPU D1_cold_single_empty_cache: 실제 reserved 피크 (OBS, trace) | 3572.0 | — |
|     같은 GPU 요청 → xMem (PRED); 최초 결정 불일치 요청 #None, 불일치 0건 | 3572.0 | 2850.7 |
|     같은 GPU 요청 → 10MiB 경계 진단 정책; 불일치 0건 | 3572.0 | 2850.7 |
| (3) GPU R0_aws_replica: 실제 reserved 피크 (OBS, trace) | 4574.0 | — |
|     같은 GPU 요청 → xMem (PRED); 최초 결정 불일치 요청 #None, 불일치 0건 | 4574.0 | 3871.6 |
|     같은 GPU 요청 → 10MiB 경계 진단 정책; 불일치 0건 | 4574.0 | 3871.6 |
| (3) GPU R1_controlled: 실제 reserved 피크 (OBS, trace) | 4574.0 | — |
|     같은 GPU 요청 → xMem (PRED); 최초 결정 불일치 요청 #None, 불일치 0건 | 4574.0 | 3871.6 |
|     같은 GPU 요청 → 10MiB 경계 진단 정책; 불일치 0건 | 4574.0 | 3871.6 |
| (3) GPU R1_controlled__stacks_all_dladdr: 실제 reserved 피크 (OBS, trace) | 4574.0 | — |
|     같은 GPU 요청 → xMem (PRED); 최초 결정 불일치 요청 #None, 불일치 0건 | 4574.0 | 3871.6 |
|     같은 GPU 요청 → 10MiB 경계 진단 정책; 불일치 0건 | 4574.0 | 3871.6 |
| (3) GPU R2_cold_single: 실제 reserved 피크 (OBS, trace) | 3572.0 | — |
|     같은 GPU 요청 → xMem (PRED); 최초 결정 불일치 요청 #None, 불일치 0건 | 3572.0 | 2850.7 |
|     같은 GPU 요청 → 10MiB 경계 진단 정책; 불일치 0건 | 3572.0 | 2850.7 |

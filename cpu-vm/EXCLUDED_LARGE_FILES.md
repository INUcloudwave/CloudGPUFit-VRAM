# cpu-vm/ — 저장소에서 제외한 파일 (CPU VM)

> **원본 없음**: 아래 파일은 GitHub 업로드 제약 등으로 저장소에 넣지 않았고, 실험에 쓴 VM이 삭제되어 원본은 더 이상 남아 있지 않습니다. 경로·크기·sha256은 기록용으로만 남깁니다. 분석 결과와 요약은 저장소에 포함된 결과 파일과 docs/ 문서에서 확인할 수 있습니다.

- 기준: GitHub 업로드 제약(50MB 초과), 모델 가중치, 호스트·접속 정보, 같은 레포에 이미 있는 중복 코드
- 그 밖의 실험 코드·입력·결과·로그는 모두 포함 (요청 trace npz, 원시 trace, GPU 반환 snapshot 중 50MB 이하 포함)

## 50MB 초과 (sha256으로 원본 확인 가능)

| INUOS VM 경로 (/home/inuos 기준) | 크기(B) | sha256 |
|---|---|---|
| `cgf-fp16-680-20261111/archive/main395_cpu_L1024_stopped_20261115.tar.gz` | 397728991 | `d8bd86db813d243d850a746c3602b062bd506ea2f7a4d124a7cd43887dfa5a62` |
| `cgf-fp16-680-20261111/runs/smoke/opt/trace.json` | 59193854 | `3481ee34814e7d47b88a7fb45177f897e1f1236d288a8d99ad410bb33679e8ac` |
| `cgf-fp16-680-20261111/runs/fp32-calib/cases/fp32__gpt2-xl__b16__in512__new512/events.npz` | 56555180 | `38bc9ae9d04cf1b10dda45b3ba9fadd7a0516a7fe590e90e2f68c9c29a369a8d` |
| `cgf-reserved-diag-20261110/inputs/gpu_return_20261111/return/core__decoder_llm__medium__fp32__12/R0_aws_replica/memory_snapshot.pickle` | 323947640 | `13e91ae2e16966e877bc24e1f2667d1eebf536e11386bf2652d5b045917fcb83` |
| `cgf-reserved-diag-20261110/inputs/gpu_return_20261111/return/core__decoder_llm__medium__fp32__12/R1_controlled/memory_snapshot.pickle` | 129665241 | `31dcd3287ffe2ae35aca10267567364108fa41ae7625eb093705422f11db21dc` |
| `cgf-reserved-diag-20261110/inputs/gpu_return_20261111/return/core__decoder_llm__medium__fp32__12/R1_controlled__stacks_all_dladdr/memory_snapshot.pickle` | 548129191 | `7e4a8a4c57da4a27b2da578dab5d1a74093b91ad7c4fa5649a8dee3e977470c2` |
| `cgf-reserved-diag-20261110/inputs/gpu_return_20261111/return/core__decoder_llm__small__fp16__01/R0_aws_replica/memory_snapshot.pickle` | 102008264 | `bdb41028cb0ab902fc60208189fc85e12298ba01d68e12521270e7aa7daa27ed` |
| `cgf-reserved-diag-20261110/inputs/gpu_return_20261111/return/core__decoder_llm__small__fp16__12/R0_aws_replica/memory_snapshot.pickle` | 209522340 | `755d350aa8e5cd49d8961d696715da1fc85913bbef95718976bb3226a817ab57` |
| `cgf-reserved-diag-20261110/inputs/gpu_return_20261111/return/core__decoder_llm__small__fp16__12/R1_controlled/memory_snapshot.pickle` | 83884255 | `2693d7b471500ed8c5e6e9a47d924a66c7ce2ac48ae7b7330302dd7eb5829d04` |
| `cgf-reserved-diag-20261110/inputs/gpu_return_raw_20261111/cgf-reserved-diag-20261110-return.tar.gz` | 89819792 | `030d347566e02549fb35c95e453b4c08ab6b67118a8fc7fa3755214233d132ec` |
| `cgf-reserved-diag-20261110/results/gpu/core__decoder_llm__medium__fp32__12/R0_aws_replica/gpu_events.normalized.jsonl.gz` | 62471593 | `e82cc68230ae79c59a8e205174d3201623a881ac8e47236ae034dca88d0ad8a8` |
| `cgf-reserved-diag-20261110/runs/cpu_history/core__decoder_llm__medium__fp32__12/from_config_per_step/events.jsonl.gz` | 62308315 | `51a975ee771b3d45a61c479cac6ea24dd34c77f3be8a16f399377f7a92969201` |

## 그 밖에 뺀 것

| 대상 | 사유 |
|---|---|
| `cgf-fp16-680-20261111/hf_cache/` (27G), `cgf-reserved-diag-20261110/hf_cache/` | 모델 가중치. revision은 `03-fp16-cpu-prediction/manifest/models.json`에 고정 |
| `cloudgpufit-cpu-diag-*/kernel.log`, `container.txt` | 호스트 커널 로그, docker inspect(호스트·환경 정보). 내용 요약은 docs/01 §3.3 |
| `CloudGPUFit-cpu-singlepass-20261108/`, `CloudGPUFit-cpu-recovery-20261108/`의 변경 없는 앱 파일 (각 약 310MB) | 같은 레포의 커밋 7f822d7(브랜치 `codex/vram-native-inference-20261103`)과 동일. 변경·추가 파일과 `uncommitted_vs_7f822d7.diff`만 포함 |
| `/home/inuos/*.zip` | 포함한 폴더들의 zip 사본 |
| 각 저장소의 `.git/` | 커밋 이력은 docs에 정리. 원격 없는 로컬 저장소 |

## 수정한 것

- `03-fp16-cpu-prediction/inputs_from_inugpu/cgf-fp16-inuos-handoff-20261115/INUOS_GPU_EXPERIMENT_SETUP_KO.md`: 사설 IP를 `[사설 IP 비공개]`로 바꿈 → 같은 폴더 `SHA256SUMS`와 이 파일만 불일치
- 하위 폴더의 `.gitignore` → `gitignore.original.txt` (결과 파일이 git에서 무시되지 않도록)

## 2026-11-20 추가

| 대상 | 사유 |
|---|---|
| AWS CPU 실제 FP16 결과의 `invocation_fp16_requests.npz` 396개 (433 MB) | 크기. 원본 INUOS `~/aws-result/aws_fp16_l256_results.tar.gz` (sha256 `114d14404d26e594400d4a300d2e2811707b5578c71e3405473db1501afe5d2b`). `../aws/README_KO.md` §4 |
| `03-fp16-cpu-prediction/archive/main395_cpu_L1024_stopped_20261115.tar.gz` (380 MB) | 중단된 L1024 CPU 실행 압축본. 같은 내용의 결과는 `runs/main395/`에 포함 |
| AWS GPU 로그의 EC2 인스턴스 ID, 파일명의 내부 호스트명 | 접속 정보. `i-REDACTED`, `host-redacted`로 바꿈 (측정값은 그대로) |

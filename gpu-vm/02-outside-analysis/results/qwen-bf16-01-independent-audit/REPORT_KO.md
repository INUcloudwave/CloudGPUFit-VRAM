# Qwen BF16 outside 분석 보고서

작성일: 2026-11-08 (한국 표준시). 실제 실행 환경: inugpu@gpuvm-1, 프로젝트 `/home/inugpu/cloudgpufit-outside`. VS Code 연결 상태 자체는 확인하지 않았으며, 이 환경에서 서버 파일을 직접 읽었습니다. 적용 위치의 AGENTS.md는 발견되지 않았습니다.

## 결과

본 실험은 12/12 정상 종료(모든 exit code 0, 하위 manifest ok)했습니다. GPT-2와 Qwen2.5-7B는 BF16에서 동일한 outside 단계 경로 254 → 296 → 312MiB를 보였습니다. 첫 forward 이후 모든 안정 단계에서 312MiB이며, 50개 조건·단계 반복 그룹의 프로세스 중앙값 범위는 모두 0MiB입니다. 단일 GPU 환경의 탐색적 프로세스 반복 결과이며 독립 GPU 환경 표본은 1개입니다.

## 정의와 감사

outside는 각 조회 구간의 해당 PID NVML 메모리 − 실제 PyTorch reserved로 원시 샘플마다 재계산했습니다. allocated 또는 xMem 예측치를 빼지 않았고, 독립 최대값끼리 차감하지 않았습니다. TNT 순서, 조회 시각, torch 전후 reserved 변화, PID 일치와 paired 산술을 검증했습니다. 기존 analyse_probe.py의 240개 process-stage 행에 대해 clean 개수·outside 중앙값이 독립 구현과 일치했습니다. 이 분석은 GPU를 사용하지 않았습니다.

원시 샘플 7,802개 = stable 3,654개 + running 4,148개입니다. CUDA 초기화 전 두 단계의 예상 결측은 504개(12×2×21)이며 전부 initialized=false, own PID absent였습니다. 나머지 stable 3,150개는 150개 process-stage에 각 21개씩 clean입니다. 모두 마지막 10개 span 0MiB로 기준 8MiB 이내입니다. running clean은 4,146개입니다. 음수 outside, 예상 밖 결측, 다른 GPU PID, 주 조회/API 오류, device_memory_info 오류는 발견되지 않았습니다. MIG NotSupported는 장치 메타데이터의 지원 여부이며 측정 API 실패와 구별했습니다.

품질 제외 2개는 모두 running입니다. GPT-2 performance_first r0 repeated_inference_block0의 조회 6.032467ms는 5ms 초과, Qwen performance_first r2 first_forward_finite_probe의 332MiB 관측은 reserved 전후 변화 때문에 제외했습니다. 값을 0으로 바꾸거나 원본을 삭제하지 않았습니다. 전체 조회 지연 중앙값 1.530847ms, 경험적 p95 약 2.604114ms, 최대 6.032467ms입니다. 50ms 간격과 21개 stable sample, 최소 clean 10개, 의미 있는 차이 및 반복 범위 16MiB 등 원래 criteria를 유지했습니다.

## 단계별 paired 메모리

단위 MiB. 아래는 동일 조건의 clean stable process 중앙값입니다. 두 실행 이력과 각 3개 반복에서 값이 같았습니다. allocated만 소수 셋째 자리로 반올림했습니다. NVML·reserved 최대는 다음 절에 별도 기술했습니다.

|모델|단계|NVML|allocated|reserved|outside|
|---|---|---:|---:|---:|---:|
|openai-community/gpt2|after_cuda_init|254|0|0|254|
|openai-community/gpt2|after_model_load|768|250.487|472|296|
|openai-community/gpt2|after_first_forward|788|259.487|476|312|
|openai-community/gpt2|after_first_inference|802|259.487|490|312|
|openai-community/gpt2|after_warmup3|802|259.487|490|312|
|openai-community/gpt2|final_idle|802|9|490|312|
|Qwen/Qwen2.5-7B|after_cuda_init|254|0|0|254|
|Qwen/Qwen2.5-7B|after_model_load|14842|14525.636|14546|296|
|Qwen/Qwen2.5-7B|after_first_forward|15180|14533.761|14868|312|
|Qwen/Qwen2.5-7B|after_first_inference|15190|14533.761|14878|312|
|Qwen/Qwen2.5-7B|after_warmup3|15190|14533.761|14878|312|
|Qwen/Qwen2.5-7B|final_idle|15190|8.125|14878|312|

after_device_metadata_and_seed, after_first_gpu_allocation, after_transformers_import는 모두 254MiB입니다. after_repeat_block0/1/2와 performance_first에만 있는 after_performance5도 모두 312MiB입니다. 모델 로딩 이후 첫 forward checkpoint의 outside 증가분은 +16MiB로 사전 meaningful threshold와 같습니다. CUDA 초기화→모델 로딩은 +42MiB입니다. 첫 forward는 입력 전송·forward·logits 유한성 확인을 함께 수행하므로 특정 라이브러리 하나의 효과로 귀속할 수 없습니다. final_idle에서는 모델 해제로 allocated가 줄었지만 allocator cache가 유지되어 reserved 및 outside는 유지되었습니다.

## 최대값과 비교

|모델|관측 reserved 최대|관측 해당 PID NVML 최대|
|---|---:|---:|
|GPT-2 BF16|490|802|
|Qwen BF16|14878|15190|

각 열은 모든 원시 샘플에 대한 독립적인 기술 최대값입니다. 두 열의 차이를 outside 추정으로 사용하지 않았습니다. 단계별 최대와 paired 중앙값은 paired_process_stage.csv에서 분리되어 있습니다. 로딩 중 연속 샘플이 없고 짧은 연산의 transient peak는 놓칠 수 있으므로 실제 최대 보장은 아닙니다.

GPT-2 BF16와 Qwen BF16의 공통 안정 단계 outside 차이는 0MiB입니다. reserved 규모가 크게 달라도 이 조건에서는 outside가 같았습니다. 모델 규모와 아키텍처가 함께 바뀌었으므로 순수 파라미터 수 인과효과를 검증한 것은 아닙니다.

direct와 performance_first는 동일 모델·반복·공통 안정 단계 72쌍에서 outside 차이 0MiB였습니다. 두 경로 모두 첫 forward, generate, warmup을 수행하며 performance_first가 generate 5회만 추가합니다. cold/warm 비교나 원 AWS 실행 이력의 정확한 재현은 아닙니다.

2026-11-04의 large-01 GPT-2 FP16와 2026-11-08의 qwen-bf16-01 GPT-2 BF16를 별도 suite로 비교했습니다. 공통 stable 75쌍에서 outside 차이 0MiB입니다. 이미지·probe·GPU·드라이버·torch/runtime·revision·입력 hash를 대조한 기록은 run_provenance.csv에 있습니다. 이전 FP16 suite의 CUDA_MODULE_LOADING 환경 변수는 미설정, 이번 BF16은 명시적 LAZY였으나 late CUDA API의 실제 모드는 둘 다 LAZY였습니다. 날짜가 다른 별도 suite이며 무작위 정밀도 교차 실험은 아니므로 일반적인 정밀도 무효과를 입증하지 않습니다.

## 환경 및 재현 근거

본 실험은 GPT-2/Qwen × direct/performance_first × 새 프로세스 3회이며 PID 12개와 canonical UUID 1개를 확인했습니다. smoke는 이 수에 포함하지 않았습니다. 케이스 dtype은 case["dtype"]의 bf16입니다. CLI의 기본 fp16은 모델 dtype을 나타내지 않습니다. B1/T256/N64, eager attention, TF32=false, native allocator, CPU threads=4, timeout=900초, CUDA_MODULE_LOADING=LAZY이며 late driver API에서도 실제 LAZY를 확인했습니다. CUDA_MODULE_DATA_LOADING은 미설정입니다.

GPU: RTX 3090 24GiB, UUID GPU-17eb7299-8b39-914d-7c5b-26908502904f, driver 570.211.01. Python 3.11.11, torch 2.6.0+cu124, Transformers 4.52.4, CUDA build/runtime 12.4입니다. driver API 12.8 지원 표시와 구별했습니다.

이미지 ID: `sha256:457da255e6e2de7ee05ff7814674c90abcf6d1ed04e73df3cc4eebcdb31fb032`

GPT-2 revision: `607a30d783dfa663caf39e06633721c8d4cfcd7e`; Qwen revision: `272aa6d30271028cdf113345b1e82a660437edaa`. 고정 plan/revisions와 모든 하위 case를 대조했습니다. 모델별 input_ids hash는 6개 반복/이력에서 동일하며, 모델 간 입력 token 값은 다릅니다. 전체 입력 hash는 run_provenance.csv에 보존했습니다.

|파일|실행 당시 및 현재 SHA256|
|---|---|
|outside-lab/pilot.py|`3856bedccb997ae288af112f3f5e38818b3598c8f7df0b851c52adb432d030bb`|
|outside-lab/analyse_probe.py|`c4abc775416779bc2422edec4a4c74cd5c5ac99246ca9214e7027124d6475997`|
|outside-lab/probe.py|`7df9a02ce9ca938b9702ee2c9fd7d0414d0a4cfc0cfe5ab87c97d7a069243d88`|
|outside-lab/qwen-bf16-plan.json|`81a5b18a0fdc539334d68a538f530a1571e4383f835a6ad66627d6fefd398f59`|
|outside-lab/qwen-revisions.json|`27c449dc9780a68ffa713cfe318cfd6743c07158ceaa46867c2e1a98dd9232c4`|
|outside-lab/criteria.json|`e3d3c92373a0ffbcc79f973156b83e6a404e40e5044649bd828d361f38a8455b`|

실행 당시 기록된 13개 Python source와 3개 입력 파일의 hash가 현재 파일과 모두 일치했습니다. 새 감사·C단계 준비 스크립트는 별도 추가 파일입니다. 각 원시 manifest와 samples의 SHA256, suite manifest, schedule, container-inspect 및 smoke manifest hash는 evidence_sha256.csv에 있습니다. 이는 현재 증거의 식별값이며 과거 저장 이후 불변성을 별도 증명하는 것은 아닙니다.

원시 경로: `results/qwen-bf16-01/session/suite-manifest.json`, 하위 `manifest.json`/`samples.jsonl`/로그; 이미지 기록 `results/qwen-bf16-01/container-inspect.json`; 계획 `qwen-bf16-schedule.json`; 로그 `qwen-bf16-01.log`. 기존 분석은 `results/qwen-bf16-01-analysis`, 독립 감사는 `results/qwen-bf16-01-independent-audit`에 있습니다. 기존 분석 디렉터리가 없음을 확인한 뒤 생성했으며 과거 결과는 덮어쓰지 않았습니다.

## FP16 실패와 자료 범위

`results/qwen-smoke-01/smoke/manifest.json`의 traceback은 first_forward_finite_probe에서 RuntimeError: nonfinite logits입니다. before/after torch import, CUDA 초기화, metadata/seed, 첫 GPU allocation, Transformers import, 모델 로딩까지 stable 21개씩이며 첫 forward running 15개 후 실패했습니다. after_first_forward 이후 안정 관측은 없습니다. 실패 전 자료는 partial로 유지하며 성공 추론 결과로 사용하지 않았습니다. BF16 smoke의 status ok와 본 실험 12개 성공은 별도로 확인했습니다.

qwen-weight-check-01.log의 CPU weight 검사(339개 BF16 tensor, 비유한/FP16 범위 초과 없음, 최대 절댓값 414, cuda_initialized=false)는 중간 연산 overflow를 배제하거나 실패 원인을 확정하지 못합니다. FP16 실패 원본·캐시·image·probe·criteria는 변경하지 않았습니다. 프로젝트 내 AWS 원시 JSON은 발견되지 않아 인계 기록의 AWS 수치를 이번에 재검증하지 않았습니다. AWS의 다른 driver와 local 결과 차이를 하드웨어 효과로 귀속하지 않았습니다. inuos 작업에는 접근하거나 변경하지 않았습니다.

## 해석 및 다음 실험

현재 근거는 동일 환경에서 outside가 연산 이력에 따라 단계적으로 증가한 뒤 안정된다는 가설과, 조사한 모델 범위에서는 모델 reserved 규모에 민감하지 않다는 관찰을 지지합니다. outside는 순수 CUDA context 크기라고 단정할 수 없습니다. 기존 환경에 대한 조회형 312MiB 기준값 후보는 가능하지만, 미측정 GPU 예측이나 새 아키텍처 정확도는 검증하지 못했습니다. 학습을 시작할 근거는 아직 부족하며, C단계의 통제된 환경 비교와 이후 독립 GPU 환경 표본이 우선입니다.

다음 준비물은 `outside-lab/LOADING_SENSITIVITY_DESIGN_KO.md`를 참조하십시오. GPT-2 BF16 하나로 동일 이미지 LAZY/EAGER × 이력 2종 × 블록 3개, 총 12 fresh worker의 비교를 준비했습니다. 기존 LAZY 자료를 새 EAGER 대조군 대신 사용하지 않고 contemporaneous LAZY도 실행하도록 설계했습니다. 계획과 실행기 구문 및 CPU dry-run만 검증했으며 GPU 실행은 아직 시작하지 않았습니다. 향후 held-out GPU MAE32/p95 64/최대 과소추정128MiB, 경계 ±256MiB, 학습 후보 20% MAE 개선은 목표일 뿐 달성 결과가 아닙니다. 시간 샘플 기반 CI나 p값은 만들지 않았습니다.

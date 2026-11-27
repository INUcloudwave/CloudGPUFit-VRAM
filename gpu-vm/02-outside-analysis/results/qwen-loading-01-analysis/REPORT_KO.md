# Qwen 7B BF16 LAZY/EAGER outside 분석

작성일: 2026-11-08 (한국 표준시). 서버: inugpu@gpuvm-1.

## 결과

**6개 worker 모두 정상 종료했습니다. Qwen의 첫 forward 이후 outside는 LAZY 312MiB, EAGER 1,066MiB였고, 세 대응 블록 모두 +754MiB 차이가 재현됐습니다.** GPT-2 확인 실험의 EAGER 1,040MiB보다 26MiB 높았고, LAZY는 두 모델 모두 312MiB였습니다. GPT-2와 Qwen은 별도 suite이므로 이 26MiB를 모델 변경만의 인과효과로 확정하지 않습니다.

|주요 단계|Qwen LAZY|Qwen EAGER|블록별 EAGER−LAZY 범위|
|---|---:|---:|---:|
|CUDA 초기화 후|254MiB|824MiB|+570~+570MiB|
|모델 로딩 후|296MiB|824MiB|+528~+528MiB|
|첫 forward 후|312MiB|1066MiB|+754~+754MiB|
|최종 idle|312MiB|1066MiB|+754~+754MiB|

첫 generate, warmup, 반복 추론 후에도 같은 outside로 유지됐습니다. mode/stage 24개 그룹에서 각3개 프로세스 중앙값 범위는 모두0MiB였습니다. 3반복은 탐색적이며 독립 GPU 환경 표본은1개입니다.

## 설계 및 실행 확인

Qwen/Qwen2.5-7B, revision `272aa6d30271028cdf113345b1e82a660437edaa`, case dtype BF16, B1/T256/N64, direct, eager attention, TF32=false, cudnn_benchmark=false, native allocator, CPU threads4, timeout900초입니다. direct에도 첫 forward/generate/warmup/반복 추론이 포함되며 cold 실험이 아닙니다.

seed20261110으로 정한 블록 순서는 LAZY→EAGER, LAZY→EAGER, EAGER→LAZY입니다. 홀수3블록이라 순서는 완전 균형적이지 않습니다. 글로벌 independent_repeat는0~2이며, 각 worker를 별도 컨테이너에서 순차 실행했습니다.

실제 worker 관측 구간: 2026-11-08T17:32:59.005301+09:00 ~ 2026-11-08T17:38:15.785626+09:00. tmux 지속 세션에서 실행했고 전체 종료 코드0, orchestration complete, 하위 suite6개 complete/각exit0, child manifest ok6, 고유PID6, canonical GPU UUID1을 확인했습니다. 실제 late CUDA API 모드도 각 worker의 LAZY/EAGER 설정과 일치했습니다.

실행 전 다른 GPU compute 프로세스 부재와 이미지/probe hash, 고정 revision 캐시4개 shard의 파일 존재 검사를 확인했습니다. 읽기 전용 cache를 사용했고 패키지/driver/image/원본 probe/criteria를 변경하지 않았습니다. 과거 FP16 실패를 재시도하거나 삭제하지 않았고, inuos CPU 작업에 접근·중단·환경 변경하지 않았습니다. 유료 AWS는 실행하지 않았습니다.

## paired 메모리와 관측 최대

단위 MiB. 같은 clean stable 집합의 process 중앙값이며 모든 반복에서 같았습니다. allocated만 소수셋째자리로 반올림했습니다. outside는 각 raw에서 해당 PID NVML−실제 reserved로 먼저 계산했습니다.

|모드|단계|NVML|allocated|reserved|outside|
|---|---|---:|---:|---:|---:|
|LAZY|after_cuda_init|254|0|0|254|
|LAZY|after_model_load|14842|14525.636|14546|296|
|LAZY|after_first_forward|15180|14533.761|14868|312|
|LAZY|after_first_inference|15190|14533.761|14878|312|
|LAZY|final_idle|15190|8.125|14878|312|
|EAGER|after_cuda_init|824|0|0|824|
|EAGER|after_model_load|15370|14525.636|14546|824|
|EAGER|after_first_forward|15934|14533.761|14868|1066|
|EAGER|after_first_inference|15944|14533.761|14878|1066|
|EAGER|final_idle|15944|8.125|14878|1066|

|모드|관측 reserved 최대|관측 해당 PID NVML 최대|
|---|---:|---:|
|LAZY|14878MiB|15190MiB|
|EAGER|14878MiB|15944MiB|

독립 최대값은 기술 열로만 두고 outside 산출에 차감하지 않았습니다. allocated를 빼거나 xMem 예측 오차를 흡수하지 않았습니다. 로딩 중 연속 관측이 없고 transient peak를 놓칠 수 있어 실제 최대 보장은 아닙니다. 대응 stable 단계의 reserved와 allocated는 모드 간 같았습니다.

## 품질 및 독립 감사

원시4,310개 = stable1,764 + running2,546개입니다. 예상 초기화 전 결측252개(6×2×21)는 initialized=false 및 own PID absent였으며 실패와 구별했습니다. 추가 제외는 블록2 EAGER first_inference running의 reserved bracket 변화1개입니다. 그 샘플의 outside1,066MiB도 기준에 따라 제외했으며 원본은 유지했습니다.

음수 outside, 예상 밖 결측, query>5ms, 다른 GPU PID, 주 조회/API 오류 및 device_memory_info 오류는 발견되지 않았습니다. 조회 지연 중앙값1.639558ms, 경험적p95 약2.806228ms, 최대3.861794ms입니다.

초기화 후 stable process-stage72개는 각21개 clean, 총1,512개입니다. running clean은2,545개입니다. 마지막10개 span은 모든 안정 행에서0MiB입니다. 블록 내 모드 대응36쌍과 mode/stage24그룹(각3개 프로세스)을 사용했습니다. 기존 analyse_probe.py의114개 행 clean count/outside 중앙값과 독립 raw 재계산이 모두 일치했습니다. 기존 analyser의 child별 repeat group72개는 각1프로세스이므로3반복 판정으로 잘못 사용하지 않았습니다.

원래 criteria의 stable21/50ms, 최소clean10, last10span8MiB, query5ms, 반복범위/의미있는 차이16MiB를 유지했습니다. 시간 샘플에서 CI/p값을 계산하지 않았으며 음수·결측을0으로 바꾸지 않았습니다. 주요4단계의 모드 차이는 모든3블록에서 같은 방향으로16MiB를 넘었습니다.

## GPT-2 및 이전 Qwen과 비교

|첫 forward 이후 outside|GPT-2 BF16 direct|Qwen BF16 direct|
|---|---:|---:|
|LAZY|312MiB (10회)|312MiB (3회)|
|EAGER|1040MiB (10회)|1066MiB (3회)|
|모드 차이|728MiB|754MiB|

GPT-2 확인 suite와 recorded environment, software/image/GPU/driver/probe/criteria/allocator/TF32를 대조하여 같음을 확인했습니다. Qwen의 case/revision/input hash도 이전 qwen-bf16-01 Qwen 실행들과 일치했습니다. 네 시점의 recorded environment에서 의도한 차이는 CUDA_MODULE_LOADING 하나였으며 CUDA_MODULE_DATA_LOADING은 미설정입니다. 해당 내부 기본 동작까지 따로 분해한 것은 아닙니다.

모델 간26MiB 차이는 사전 의미효과 기준16MiB보다 큽니다. 다만 별도 시점/suite 비교이고 모델 구조와 규모, 토큰 입력, 실행 연산 경로가 함께 다르므로 파라미터 수나 특정 라이브러리의 효과로 귀속할 수 없습니다. 두 모델에서 LAZY가 같았다는 기존 결과를 모든 로딩 정책으로 확대할 수 없다는 관찰입니다. 순수모델 인과효과를 확정하려면 같은 시기 블록 안에서 모델×모드 2×2를 무작위 교차해야 합니다.

## 환경·재현 증거

RTX3090 24GiB, UUID `GPU-17eb7299-8b39-914d-7c5b-26908502904f`, driver570.211.01, Python3.11.11, torch2.6.0+cu124, Transformers4.52.4, CUDA build/runtime12.4. driver의12.8 지원 표시와 구별했습니다.

image ID: `sha256:457da255e6e2de7ee05ff7814674c90abcf6d1ed04e73df3cc4eebcdb31fb032`

probe SHA256: `7df9a02ce9ca938b9702ee2c9fd7d0414d0a4cfc0cfe5ab87c97d7a069243d88`

source/input hash는 실행 전과 각 worker 전에 검사했고 최종 감사에서도 모두 일치했습니다. 전체 입력 hash와 환경 정보는 run_provenance.csv, 원시 manifest에 보존했습니다.

- 사전설계: outside-lab/QWEN_LOADING_DESIGN_KO.md
- 계획/실행/분석: outside-lab/qwen-loading-plan.json, run_qwen_loading.py, analyse_qwen_loading.py (raw 감사는 audit_qwen_bf16.py 재사용)
- 원시: results/qwen-loading-01/orchestration.json 및 child/session의 suite-manifest.json, run/manifest.json, samples.jsonl, events/로그
- 이미지 기록: 각 child/container-inspect.json
- 지속 세션: run-qwen-loading-01.sh, qwen-loading-01.log, qwen-loading-01.exitcode(0)
- 기존 분석기 출력: results/qwen-loading-01-probe-analysis/
- 파생 CSV: paired_process_stage, repeat_summary, block_mode_comparison, mode_effect_summary, quality_audit, run_provenance, cross_model_comparison
- 증거 hash: evidence_sha256.csv, source_verification.csv, input_verification.csv, artifact_sha256.json

## 해석과 다음 방향

모듈 로딩 정책의 큰 outside 효과는 GPT-2에만 국한되지 않고 Qwen에서도 재현됐습니다. 반면 EAGER의 최종 outside는 두 모델에서 달랐으므로 GPU·정책만으로 항상 같은 상수를 쓸 수 있다고 단정할 수 없습니다. 최소한 목표 이미지/로딩 정책/사용한 연산 단계와 워크로드 범위를 함께 기록해야 합니다.

outside를 순수 CUDA context 크기로 보거나 모델 reserved 비율로 해석하지 않습니다. 새 GPU 접근을 필수 입력으로 바꾸거나 복잡한 학습을 바로 도입할 근거도 아닙니다. 단일 GPU·이미지에서 관측한 패턴이며 미측정 GPU 예측 정확도는 여전히 미검증입니다.

추가 대형모델을 계속 늘리기보다 (1) 모델×모드의 동시기2×2 통제 비교로26MiB 차이의 재현을 확인하거나 (2) backend 등 하나의 소프트웨어 요인을 통제해 비교하는 것이 다음 C단계 후보입니다. 이후 D단계에서는 동일 이미지의 독립 GPU/환경 표본 확보가 필요합니다. 이번 요청은 계획된6개와 분석까지 완료했고 후속 실험이나 유료 자원은 시작하지 않았습니다.

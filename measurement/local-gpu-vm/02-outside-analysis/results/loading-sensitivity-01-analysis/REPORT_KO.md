# LAZY/EAGER outside 민감도 실험 결과

2026-11-08, inugpu@gpuvm-1. 사전 설계: outside-lab/LOADING_SENSITIVITY_DESIGN_KO.md.

## 결론

**12/12 fresh worker가 정상 종료했습니다. EAGER에서 첫 forward 이후 outside는 1,040MiB, LAZY에서는 312MiB로, 블록별 차이는 모두 +728MiB였습니다.** 3개 블록과 direct/performance_first 모두 재현되었고, mode/history/stage 50개 그룹의 프로세스 중앙값 범위는 모두 0MiB였습니다. CUDA 모듈 로딩 설정이 이 GPU·이미지·워크로드의 outside에 큰 영향을 준다는 탐색적 증거입니다. 12개 독립 GPU 환경 표본으로 해석하지 않습니다.

## 설계 및 실행 검증

GPT-2 BF16, batch1/입력256/생성64, eager attention, TF32=false, cudnn_benchmark=false, native allocator, CPU threads4, worker timeout900초. revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. 3개 블록 × LAZY/EAGER × direct/performance_first = worker12개, 컨테이너6개이며 모두 순차 실행했습니다. 글로벌 반복 식별자는 orchestration block이고, 하위 suite의 repeat=0은 그대로 보존했습니다.

고정 seed20261108로 정한 모드 순서는 블록0 LAZY→EAGER, 블록1 LAZY→EAGER, 블록2 EAGER→LAZY입니다. 각 하위 suite 내부 이력 순서는 동일한 pilot seed에 의해 performance_first→direct였습니다. 모드 실행 순서는 3블록이라 완전히 균형적이지 않지만 마지막 역순 블록에서도 같은 값이 재현되었습니다.

orchestration status complete, 6개 컨테이너 exit0, 6개 하위 suite complete/각2개 exit0, child manifest ok12, PID12, canonical GPU UUID1을 확인했습니다. tmux 지속 세션과 로그/exitcode 파일을 사용했고 재시도·정밀도 변경·원본 삭제 없이 완료했습니다. 실제 late CUDA API 모드는 의도한 LAZY/EAGER와 전부 일치했습니다.

실행 시작 전 GPU compute 프로세스가 없었고 고정 이미지/probe와 읽기 전용 cache의 GPT-2 파일 존재 검사가 통과했습니다. 원시 측정에서도 다른 GPU PID는 관측되지 않았습니다. inuos CPU 작업에는 접근·중단·환경 변경하지 않았습니다. 패키지/driver/image/cache는 변경하지 않았고 유료 AWS 리소스를 실행하지 않았습니다.

## outside 단계 변화

아래는 clean stable 샘플의 프로세스 중앙값입니다. 각 mode/history별 3반복 모두 같은 값이고 두 이력 간 공통 단계 차이도 0입니다. 단위 MiB.

|단계|LAZY outside|EAGER outside|블록별 EAGER−LAZY|
|---|---:|---:|---:|
|CUDA 초기화 후|254|824|+570|
|모델 로딩 후|296|824|+528|
|첫 forward 후|312|1040|+728|
|첫 generate 후|312|1040|+728|
|warmup 및 반복 추론 후|312|1040|+728|
|final idle|312|1040|+728|

LAZY는 초기화→로딩 +42MiB, 로딩→첫 forward +16MiB였습니다. EAGER는 초기화→로딩 0MiB, 로딩→첫 forward +216MiB였고 이후 평탄했습니다. 모드 차이는 단계에 따라 달라 단일 초기화 차이만 최종 추론에 더해서는 맞지 않습니다. 다만 이 측정만으로 증가분을 특정 라이브러리/모듈 또는 순수 CUDA context로 귀속할 수 없습니다.

## paired 메모리와 관측 최대

각 열은 같은 clean stable 관측 집합에서 구한 process 중앙값입니다. outside는 원시 샘플마다 해당 PID NVML−실제 reserved로 먼저 계산했습니다. allocated 또는 xMem 오차를 포함하지 않았습니다. 단위 MiB.

|모드|단계|NVML|allocated|reserved|outside|
|---|---|---:|---:|---:|---:|
|LAZY|after_cuda_init|254|0|0|254|
|LAZY|after_model_load|768|250.487|472|296|
|LAZY|after_first_forward|788|259.487|476|312|
|LAZY|after_first_inference|802|259.487|490|312|
|LAZY|final_idle|802|9|490|312|
|EAGER|after_cuda_init|824|0|0|824|
|EAGER|after_model_load|1296|250.487|472|824|
|EAGER|after_first_forward|1516|259.487|476|1040|
|EAGER|after_first_inference|1530|259.487|490|1040|
|EAGER|final_idle|1530|9|490|1040|

|모드|관측 reserved 최대|관측 해당 PID NVML 최대|
|---|---:|---:|
|LAZY|490|802|
|EAGER|490|1530|

최대값은 별도 기술 열이며 그 차이를 outside로 사용하지 않았습니다. 로딩 중 연속 관측이 없고 짧은 연산의 transient peak를 놓칠 수 있어 실제 최대 보장은 아닙니다. 모드별 stable reserved와 allocated는 같은 단계에서 같았고, NVML 및 paired outside에 차이가 있었습니다.

## 품질 감사 및 통계 단위

원시 5,970개 = stable3,654 + running2,316. CUDA 초기화 전 예상 결측504개(12×2×21)는 initialized=false/own PID absent였으며 실제 실패와 구별했습니다. 그 외 품질 제외는 query>5ms 4개(stable2, running2), reserved bracket 변화1개(running)입니다. 제외 사유는 서로 겹치지 않습니다. 음수 outside, 예상 밖 결측, 다른 GPU PID, 측정 API 오류, device_memory_info 오류는 발견되지 않았습니다.

CUDA 초기화 후 stable process-stage150개는 각각 clean20~21개, 총3,148개입니다. running clean2,313개입니다. 모든 사용 가능한 stable 행의 마지막10개 span은 0MiB로 사전8MiB 기준 안입니다. 전체 조회 지연 중앙값1.5791635ms, 경험적 p95 약2.70182ms, 최대9.019228ms였습니다. TNT 조회 시각/순서와 reserved 전후 변화, PID, raw outside 산술을 독립 검증했습니다. 원본은 모두 유지했습니다.

원래 criteria의 stable21개/50ms, clean최소10, plateau8MiB, 조회5ms, 반복범위/의미있는 효과16MiB를 유지했습니다. 독립 감사240행의 clean count와 outside 중앙값이 기존 analyse_probe.py 결과와 일치했습니다. 기존 analyser의 repeat_groups150개는 child suite별 그룹이므로 각1프로세스이며, 모드별 3반복 판정에는 새 분석기의 글로벌 block/mode/history/stage 그룹50개를 사용했습니다.

블록 내 모드 대응75쌍, 이력 대응72쌍입니다. 주요4단계×두 이력에서 모든 블록의 모드 차이가 같은 방향으로16MiB 이상이어서 사전 탐색 기준을 통과했습니다. 이력 대응차이는 전부0MiB입니다. 시간 샘플이나 이력 반복을 독립 GPU 표본으로 세거나 시간 샘플에서 CI/p값을 계산하지 않았습니다.

## 환경·입력 및 증거

RTX3090 24GiB, UUID `GPU-17eb7299-8b39-914d-7c5b-26908502904f`, driver570.211.01. Python3.11.11, torch2.6.0+cu124, Transformers4.52.4, CUDA build/runtime12.4. 드라이버 지원12.8 표시와 구별했습니다.

image ID `sha256:457da255e6e2de7ee05ff7814674c90abcf6d1ed04e73df3cc4eebcdb31fb032`

probe SHA256 `7df9a02ce9ca938b9702ee2c9fd7d0414d0a4cfc0cfe5ab87c97d7a069243d88`

recorded environment 4시점에서 CUDA_MODULE_LOADING을 제외한 항목이 모두 동일했습니다. CUDA_MODULE_DATA_LOADING은 모두 미설정이며 그 내부 기본 동작까지 별도 측정한 것은 아닙니다. case dtype은 bf16이고 일반 CLI 기본 dtype과 구별했습니다. revision·input hash·criteria hash·image·GPU·software, attention/sampling/query order/샘플수/간격/threads를 대조했습니다. orchestration에 기록된7개 source/input hash와 모든 child suite source/input hash가 현재 파일과 일치했습니다.

- raw: `results/loading-sensitivity-01/orchestration.json`, 각 `*_block*_MODE/session/{suite-manifest.json,run/manifest.json,run/samples.jsonl}`
- 컨테이너 metadata: 각 child의 `container-inspect.json`; 로그: root `00.log`~`05.log`와 child worker 로그
- 지속 세션 실행기: `run-loading-sensitivity-01.sh`; 최종 로그/종료코드: `loading-sensitivity-01.log`, `loading-sensitivity-01.exitcode`(0)
- 독립 분석: `results/loading-sensitivity-01-analysis/`의 paired_process_stage, repeat_summary, block_mode_comparison, mode_effect_summary, history_comparison, quality_audit, run_provenance CSV
- 기존 분석: `results/loading-sensitivity-01-probe-analysis/`; 감사 코드: `outside-lab/analyse_loading_sensitivity.py` 및 재사용한 `audit_qwen_bf16.py`
- 원시 manifest/samples의 SHA256: `evidence_sha256.csv`; 실행기 당시 source hash 대조: `source_verification.csv`

기존 Qwen BF16 suite의 LAZY312MiB는 이번 contemporaneous LAZY에서도 재현되었지만 기존 자료를 paired 대조군으로 섞지 않았습니다. 과거 FP16 실패 자료는 그대로 보존했습니다.

## CloudGPUFit에 대한 함의와 다음 단계

현재 설정에서 최종312MiB를 쓸 수 있다는 관찰은 LAZY 환경에 한정해야 합니다. 같은 GPU/이미지/모델도 EAGER에서1,040MiB였으므로 GPU UUID/종류만으로 모든 소프트웨어 설정을 포괄하는 outside 상수는 이 실험과 맞지 않습니다. allocator reserved 예측과 outside를 분리하고, 목표 GPU 런타임의 이미지·로딩 정책·workload 단계를 예측 조건에 포함할 근거가 생겼습니다. 대상 GPU 실측을 필수 입력으로 바꾸거나 복잡한 학습을 바로 도입할 근거는 아닙니다.

다음 우선 실험은 이 큰 차이의 최소10회 확인입니다. GPT-2 BF16 direct로 이력을 줄이고, 동일 GPU·이미지에서 10블록×LAZY/EAGER(총20 fresh worker)를 시간적으로 대응시키며 mode 순서는5블록씩 균형화하는 설계가 적절합니다. 이 탐색 결과를 별도 보존하고 새 확인 suite를 만들며, 확인 후 Qwen BF16로 모델 간 재현을 점검할 수 있습니다. 이번 요청에서는 계획된12개 비교와 분석까지 수행했고 추가20개는 실행하지 않았습니다.

C의 다른 software/backend 및 D의 GPU 간 일반화는 아직 미검증입니다. GPU1개 환경에서의 결과로 새 아키텍처 정확도를 보장하지 않습니다. 향후 GPU 표본 확보 시 driver/software 차이를 함께 기록하고 leave-one-GPU-out을 사용해야 합니다. MAE32/p95절대64/최대과소128MiB 및 ±256MiB 경계, 학습후보20% 개선은 계속 사전 목표로만 남습니다.

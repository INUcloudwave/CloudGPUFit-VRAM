# LAZY/EAGER 10블록 확인 실험 보고서

작성일: 2026-11-08 (한국 표준시). 실행 서버: inugpu@gpuvm-1.

## 결과

**20개 worker 모두 정상 종료했습니다. LAZY의 추론 후 outside는312MiB, EAGER는1,040MiB였으며, 10블록 모두 +728MiB 차이가 재현됐습니다.** 각 모드의10개 새 프로세스에서 안정 단계 중앙값 범위는0MiB입니다. 이는 이전3반복 탐색을 별도 확인 suite에서 재현한 결과입니다. 독립 GPU 환경 표본 수는 여전히1입니다.

|주요 단계|LAZY outside|EAGER outside|10블록 EAGER−LAZY 범위|
|---|---:|---:|---:|
|after_cuda_init|254MiB|824MiB|+570~+570MiB|
|after_model_load|296MiB|824MiB|+528~+528MiB|
|after_first_forward|312MiB|1040MiB|+728~+728MiB|
|final_idle|312MiB|1040MiB|+728~+728MiB|

first_inference, warmup, repeat block0/1/2 이후에도 각 모드의 outside는312/1040MiB로 유지됐습니다. 주요4단계 모두 모든10블록에서 같은 방향으로16MiB 이상인 사전 확인 항목을 충족했습니다. 단, 확인 설계는 앞선 +728MiB 탐색 결과를 본 뒤 정했으므로 최초부터 결과를 몰랐던 사전등록이라고 주장하지 않습니다.

## 설계 및 실제 실행

GPT-2 BF16 direct, B1/T256/N64, eager attention, TF32=false, cudnn_benchmark=false, native allocator,4 CPU threads,900초 timeout입니다. direct에도 첫 forward/generate/warmup/반복 추론이 포함되며 cold/warm 비교는 아닙니다.

LAZY→EAGER5블록과 EAGER→LAZY5블록을 seed20261109로 섞었습니다(seed는 날짜가 아닌 식별 정수). 글로벌 repeat0~9는 실제 probe의 independent_repeat에도 기록했습니다. 모드별10개 worker,20개 컨테이너를 모두 순차 실행했습니다. 탐색12개와 확인20개를 풀링하지 않았습니다.

실제 worker 관측 구간: 2026-11-08T17:11:18.645103+09:00 ~ 2026-11-08T17:21:36.632227+09:00. tmux 지속 세션에서 실행했고 최종 종료코드는0입니다. orchestration complete, 하위20개 suite complete/각exit0, child manifest ok20, 고유PID20, UUID1을 확인했습니다. 각 worker의 late CUDA API가 의도한 LAZY/EAGER를 보고했습니다. 실패·timeout·재시도는 없었습니다.

실행 전 고정 이미지/probe hash와 읽기 전용 캐시 파일 검사, GPU compute 프로세스 부재를 확인했습니다. 이번 별도 실행기는 기존 probe를 컨테이너에서 직접 실행합니다. 이전 탐색 direct와 case/revision/input hash/기록 환경/image/software/allocator/TF32를 대조하여 같음을 확인했습니다. 새 launcher와 실험 identity, 경로 및 시간이 다른 사실은 기록하고 통계는 분리했습니다.

## paired 메모리

단위MiB. 같은 clean stable 관측 집합의 process 중앙값입니다. 모든 반복에서 값이 같았고 allocated만 소수셋째자리로 표시했습니다. outside는 각 raw의 해당 PID NVML−실제reserved를 먼저 계산해 집계했습니다.

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

|모드|관측 reserved 최대|관측 해당PID NVML 최대|
|---|---:|---:|
|LAZY|490MiB|802MiB|
|EAGER|490MiB|1530MiB|

최대값은 각각 별도 기술값이며 독립 최대값끼리 차감하지 않았습니다. allocated 차감이나 xMem 오차 흡수도 하지 않았습니다. transient peak를 놓칠 수 있고 로딩 중 연속 관측이 없으므로 실제 최대 보장은 아닙니다.

## 품질 감사와 검증

raw8,879개 = stable5,880 + running2,999개입니다. CUDA 초기화 전 두 단계의 예상 결측840개(20×2×21)는 initialized=false/ownPID absent였으며 실제 실패와 구별했습니다. 추가 제외3개는 모두 query>5ms로 stable2개, running1개입니다. 음수outside·예상밖결측·bracket변화·다른GPU PID·측정API오류·device_memory_info오류는 발견되지 않았습니다. 원본을 삭제하거나 결측/음수를0으로 바꾸지 않았습니다.

CUDA 초기화 후 stable process-stage240개는 clean20~21개씩, 총5,038개를 확보했습니다. running clean2,998개입니다. mode/stage24그룹의 프로세스 중앙값 반복범위는 전부0MiB, last10 span최대도0MiB입니다. 블록 내 모드대응120쌍을 만들었으며 시간 샘플을 독립 반복으로 세지 않았습니다. 조회지연 전체 중앙값1.582502ms, 경험적p95 약2.683412ms, 최대5.589357ms입니다.

원래 criteria의 stable21개/50ms, clean최소10, plateau8MiB, query5ms, 반복범위/효과16MiB를 유지했습니다. 시간 샘플 기반 CI/p값을 계산하지 않았고, 반복값이 같다는 관찰로 미래분산0을 보장하지 않습니다. 기존 analyse_probe.py의380행 clean count/outside median은 독립 raw 감사와 모두 일치했습니다. 기존 analyser의 child별 repeat group240개는 각1프로세스여서 확인 판단에는 글로벌 mode/stage24그룹(각10프로세스)을 사용했습니다.

기존 탐색 direct와24개 mode/stage 집계에서도 outside 차이가 모두0MiB였습니다. 비교는 pilot_confirmation_comparison.csv로 별도 보존했습니다.

## 환경 및 증거 경로

GPU RTX3090 24GiB, UUID GPU-17eb7299-8b39-914d-7c5b-26908502904f, driver570.211.01. Python3.11.11, torch2.6.0+cu124, Transformers4.52.4, CUDA build/runtime12.4. driver 지원12.8표시와 구별했습니다.

image: `sha256:457da255e6e2de7ee05ff7814674c90abcf6d1ed04e73df3cc4eebcdb31fb032`

probe SHA256: `7df9a02ce9ca938b9702ee2c9fd7d0414d0a4cfc0cfe5ab87c97d7a069243d88`

GPT2 revision: `607a30d783dfa663caf39e06633721c8d4cfcd7e`. case dtype는bf16이며 일반CLI dtype기본값과 구별했습니다.

4시점 recorded environment에서 mode변수를 제외한 항목이 동일했습니다. CUDA_MODULE_DATA_LOADING은 미설정이며 내부 기본 동작의 효과까지 분리한 것은 아닙니다. source/input hash를 시작 및 매worker 전에 검사했고 최종 감사에서도 대조했습니다. input token hash는 모든반복/모드에서 같았습니다. 전체 hash는 아래 파일에 보존했습니다.

- 사전설계: outside-lab/LOADING_CONFIRMATION_DESIGN_KO.md
- 계획/실행기: outside-lab/loading-confirmation-plan.json, run_loading_confirmation.py
- raw: results/loading-confirmation-01/orchestration.json 및 각 child/session의 suite-manifest.json, run/manifest.json, samples.jsonl, events/로그
- container metadata: 각 child/container-inspect.json
- 지속실행: run-loading-confirmation-01.sh, loading-confirmation-01.log, loading-confirmation-01.exitcode(0)
- 독립감사코드: outside-lab/analyse_loading_confirmation.py와 audit_qwen_bf16.py
- 기존분석: results/loading-confirmation-01-probe-analysis/
- 독립분석CSV: 이 폴더의 paired_process_stage, repeat_summary, block_mode_comparison, mode_effect_summary, quality_audit, run_provenance, pilot_confirmation_comparison
- 증거hash: evidence_sha256.csv, source_verification.csv, input_verification.csv, artifact_sha256.json

raw·과거 실패·cache·고정image·driver·기존probe/criteria는 변경하지 않았습니다. inuos CPU작업에 접근·중단·환경변경하지 않았으며 유료AWS를 실행하지 않았습니다.

## 무엇을 확인했는가 / 다음 방향

첫 탐색의 모듈 로딩 효과가 각모드10회 확인에서도 재현됐습니다. 이 환경에서 GPU종류만으로 outside상수312MiB를 적용하면 EAGER추론의outside를728MiB 과소평가합니다. 따라서 목표 런타임의 소프트웨어/로딩정책/관측단계를 outside조건으로 기록할 필요가 있습니다. 초기화차이570MiB를 최종차이728MiB와 같다고 가정해서도 안 됩니다.

확인 범위는 RTX3090 한대·고정이미지·GPT2 BF16 direct입니다. 드라이버·CUDA버전·GPU아키텍처의 효과, 다른모델에서도 정확히728MiB인지, outside구성요소별 할당 원인은 아직 모릅니다. outside를 순수context로 단정하지 않습니다. 대상GPU실측을 필수로 요구하는 방식이나 복잡한학습을 도입할 근거로 확대하지 않습니다.

다음 유용한 실험은 같은이미지/GPU에서 Qwen2.5-7B BF16 direct의 LAZY/EAGER를3개 대응블록(총6개 fresh worker)으로 비교하는 것입니다. 이는 설정효과가 다른모델에서도 재현되는지 확인하며 GPU일반화 실험은 아닙니다. 이번 작업에서는 확인20개까지 실행했고 Qwen후속실험은 아직 시작하지 않았습니다. 그 이후에는 같은GPU반복을 무한히 늘리기보다 독립GPU/환경 표본 확보를 우선해야 합니다. 새로운GPU 유료자원은 자동 시작하지 않습니다.

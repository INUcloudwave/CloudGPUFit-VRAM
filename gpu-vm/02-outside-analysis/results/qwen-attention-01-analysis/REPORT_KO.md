# Qwen BF16·LAZY eager attention / SDPA 비교

작성일: 2026-11-08 (한국 표준시). 서버: inugpu@gpuvm-1.

## 결과와 사용자 100MiB 기준

**본 측정6개가 모두 정상 종료했고, 첫 forward 이후 outside는 eager attention과 SDPA 모두312MiB였습니다. 세 블록의 SDPA−eager 차이는 전부0MiB로, 사용자 기준100MiB 이내에서 실용적으로 거의 고정입니다.**

첫 forward, 첫 generate, warmup, repeat block0/1/2, final_idle의6개 프로세스×7개 단계=42개 안정 process 중앙값이 전부312MiB였습니다. 전체 범위도0MiB이므로 블록 대응 차이와 전체 비교조건 범위 모두100MiB 기준을 충족했습니다. 이는 시간 샘플 기반 통계적 동등성 증명이나 미래 오차 보장이 아닙니다.

100MiB 기준은 앞선 결과 이후 사용자가 정한 실용적 기준입니다. 기존 criteria.json의 품질 기준과16MiB 탐색 효과 기준은 변경하지 않고 practical-outside-criteria.json으로 분리해 기록했습니다.

|단계|eager attention outside|SDPA outside|블록별 차이|
|---|---:|---:|---:|
|CUDA 초기화 후|254MiB|254MiB|0MiB|
|모델 로딩 후|296MiB|296MiB|0MiB|
|첫 forward 후|312MiB|312MiB|0MiB|
|첫 generate 및 이후 반복/idle|312MiB|312MiB|0MiB|

초기화 이후 모든 stable 단계까지 함께 보면254~312MiB, 범위58MiB이므로 이 LAZY 조건에서는 단계 차이도 사용자100MiB 기준 안입니다. 원래16MiB 기준에서는 단계적 변화가 있다는 기술적 관찰은 그대로 보존합니다.

## 설계와 실제 실행

Qwen/Qwen2.5-7B revision272aa6d30271028cdf113345b1e82a660437edaa, case BF16, B1/T256/N64, direct, CUDA_MODULE_LOADING=LAZY, CUDA_MODULE_DATA_LOADING 미설정,4 CPU threads, TF32=false, cudnn_benchmark=false, native allocator, timeout900초를 유지했습니다. 바꾼 측정 인자는 --attention eager/sdpa뿐입니다. 일반CLI dtype 기본값과 실제case dtype를 구별했습니다.

3개 대응블록에 각각2개 fresh worker를 순차 실행했습니다. 블록 순서는 eager→sdpa, eager→sdpa, sdpa→eager이며 seed20261111은 날짜가 아닌 식별정수입니다. 홀수블록이라 순서가 완전균형적이지 않으며3반복은 탐색적입니다. global independent_repeat는0~2입니다. direct에도 첫forward/generate/warmup/반복추론이 포함되고 cold 실험은 아닙니다.

본 worker 관측 구간: 2026-11-08T18:03:29.327202+09:00 ~ 2026-11-08T18:08:35.223108+09:00. 본6개 suite complete/각exit0, manifest ok6, PID6, canonical UUID1, 실제CUDA모드LAZY를 확인했습니다. 진단2개도 별도프로세스에서exit0으로 완료했고 전체orchestration complete/최종종료0입니다. 총8개 GPU프로세스를 순차 실행했지만 outside의 프로세스반복수는 본6개뿐이며 독립GPU환경 표본은1개입니다.

## SDPA가 실제로 선택한 연산

본 측정은 원본probe를 그대로 사용했습니다. profiler가 메모리 측정에 영향을 주지 않도록 본6개가 끝난 후 attention별1개의 별도진단프로세스를 실행했습니다. 입력seed/hash, 모델/revision/BF16, LAZY, B1/T256/N64와TF32설정을 대조했습니다. 진단에서는 CPU operator profiler로 PyTorch가 호출한 내부연산을 관찰했고 profile_memory는 껐습니다.

|별도 진단|첫 forward|generate64|
|---|---|---|
|eager attention|aten::softmax 28회|aten::softmax 1792회|
|SDPA|aten::_scaled_dot_product_flash_attention 28회|동일 연산1792회|

SDPA에는 aten::scaled_dot_product_attention과 aten::_flash_attention_forward도 함께 관측됐습니다. Transformers 실제 _attn_implementation은 각각eager/sdpa였고 로컬 Qwen2Attention.forward 및 sdpa_attention_forward의 분기소스/hash도 보존했습니다. 별도진단에서 Flash 경로를 확인한 것이며 본6개 worker의 매호출을 직접프로파일링한 것은 아닙니다. 모든SDPA 설정·다른입력shape·다른GPU에서도같은경로라고 일반화하지 않습니다. FlashAttention 외부패키지를 새로 설치하거나 backend를 강제로 변경하지 않았습니다.

## paired 메모리

단위MiB. 같은 clean stable 집합의 process 중앙값이며 allocated만 소수셋째자리로 반올림했습니다. outside는 각 관측구간 해당PID NVML−실제reserved로 먼저 계산했습니다.

|attention|단계|NVML|allocated|reserved|outside|
|---|---|---:|---:|---:|---:|
|eager|after_cuda_init|254|0|0|254|
|eager|after_model_load|14842|14525.636|14546|296|
|eager|after_first_forward|15180|14533.761|14868|312|
|eager|after_first_inference|15190|14533.761|14878|312|
|eager|final_idle|15190|8.125|14878|312|
|sdpa|after_cuda_init|254|0|0|254|
|sdpa|after_model_load|14842|14525.636|14546|296|
|sdpa|after_first_forward|15178|14533.761|14866|312|
|sdpa|after_first_inference|15190|14533.761|14878|312|
|sdpa|final_idle|15190|8.125|14878|312|

|attention|관측 reserved 최대|관측 해당PID NVML 최대|
|---|---:|---:|
|eager|14878MiB|15190MiB|
|SDPA|14878MiB|15190MiB|

첫forward checkpoint의 reserved는eager14868MiB, SDPA14866MiB로2MiB 차이가 있었지만 paired outside는 같았습니다. 최대값은 독립 기술열이며 그 차이를 outside 산출에 사용하지 않았습니다. allocated차감·xMem오차흡수·결측0치환도 하지 않았습니다. 로딩 중 연속 관측이 없고 짧은transient peak를 놓칠 수 있어 실제최대보장은 아닙니다.

## 품질 감사

본raw4,249개 = stable1,764 + running2,485개입니다. 초기화전 예상결측252개(6×2×21)는 initialized=false/ownPID absent였으며 실패와 구별했습니다. 추가 제외7개는 전부running으로 bracket변화4개, query>5ms3개이며 사유중복은없습니다. 음수outside, 예상밖결측, 다른GPU PID, 주조회/API오류 및 device_memory_info오류는 발견되지 않았습니다. 원본과 실패자료는 보존했습니다.

초기화후 stable process-stage72개는 각21개 clean으로 총1,512개이며 running clean은2,478개입니다. mode/stage24개 반복그룹의 process중앙값범위는 모두0MiB, last10span도0MiB입니다. 블록대응36쌍을 만들었고 기존분석기114개행의clean count/median은 독립raw재계산과 전부일치했습니다. 기존분석기의child별repeat그룹72개는각1프로세스이므로3반복판정으로잘못사용하지않았습니다.

전체조회지연 중앙값1.593911ms, 경험적p95 약2.760094ms, 최대6.467493ms입니다. 원래품질기준 stable21개/50ms, clean최소10, last10span8MiB, query5ms와bracket/음수/다른PID/API오류제외를유지했습니다. 시간샘플에서CI/p값을계산하거나프로세스반복을GPU환경반복으로세지않았습니다.

## 환경·입력·증거

RTX3090 24GiB, UUID GPU-17eb7299-8b39-914d-7c5b-26908502904f, driver570.211.01. Python3.11.11, torch2.6.0+cu124, Transformers4.52.4, CUDA build/runtime12.4. driver지원12.8표시와구별했습니다.

image `sha256:457da255e6e2de7ee05ff7814674c90abcf6d1ed04e73df3cc4eebcdb31fb032`

probe SHA256 `7df9a02ce9ca938b9702ee2c9fd7d0414d0a4cfc0cfe5ab87c97d7a069243d88`

4시점recorded environment는모든본worker에서같았고 attention설정만달랐습니다. 이전qwen-loading-01 LAZY와case/revision/input hash/image/driver/software/probe/criteria/allocator/TF32가같음을대조했습니다. 진단의입력hash도본측정과같았습니다. source/input hash를실행전고정하고매worker전에검사했으며최종감사에서도일치했습니다.

- 사전설계: outside-lab/QWEN_ATTENTION_DESIGN_KO.md
- 새실용기준: outside-lab/practical-outside-criteria.json (기존criteria.json유지)
- 계획/실행/분석/진단: outside-lab/qwen-attention-plan.json, run_qwen_attention.py, analyse_qwen_attention.py, diagnose_attention_backend.py
- raw: results/qwen-attention-01/orchestration.json 및 child/session의 suite-manifest.json, run/manifest.json, samples.jsonl, events/로그
- 진단: results/qwen-attention-01/diagnostic_eager/profile/diagnostic.json 및 diagnostic_sdpa/profile/diagnostic.json, 로컬분기소스
- 이미지기록: 각child/container-inspect.json
- 지속세션: run-qwen-attention-01.sh, qwen-attention-01.log, qwen-attention-01.exitcode(0)
- 기존분석: results/qwen-attention-01-probe-analysis/
- 파생CSV: paired_process_stage, repeat_summary, block_mode_comparison, mode_effect_summary, quality_audit, run_provenance, practical_assessment, diagnostic_attention_operators
- hash: evidence_sha256.csv, source_verification.csv, input_verification.csv, artifact_sha256.json

cache는읽기전용이며고정probe/criteria/image/driver/패키지/과거실패자료를변경하지않았습니다. inuos CPU작업에는접근·중단·환경변경하지않고유료AWS는시작하지않았습니다.

## 현재까지100MiB 기준으로 해석

- 같은로딩정책에서 GPT2↔Qwen: LAZY차이0MiB, EAGER차이26MiB로조사범위에서는실용적으로거의고정입니다(별도suite비교).
- 이번Qwen·LAZY의 eager attention↔SDPA: 추론이후차이0MiB로실용적으로거의고정입니다.
- CUDA모듈 LAZY↔EAGER: 차이728~754MiB로같은상수로묶을수없습니다.
- 초기화부터추론단계: LAZY254~312MiB의범위58MiB는100이내입니다. 반면EAGER는824→1040/1066MiB로216~242MiB증가하여초기화값과최종값을같게볼수없습니다.

따라서지금까지의고정GPU/이미지·LAZY·조사한모델과backend범위에서는추론outside312MiB를기준값후보로둘근거가늘었습니다. 모든모델/shape/소프트웨어/GPU에서그상수가유효하다고확정하거나outside를순수context크기로단정하지않습니다. 대상GPU실측을필수입력으로바꾸거나복잡한학습을바로시작할근거도아닙니다.

아직B1/T256/N64로고정했으므로다음로컬검증후보는배치·입력길이·생성길이변화입니다. 또는동일이미지의독립GPU환경표본으로일반화범위를확장해야합니다. 이번작업은6개본측정과2개분리진단·분석까지완료했고추가실험/유료자원은실행하지않았습니다.

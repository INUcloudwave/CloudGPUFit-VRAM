# 로컬 RTX 3090 마무리 검증

2026-11-10 (한국 표준시). 서버 inugpu@gpuvm-1.

## 결론

**로컬 추가5회가 모두 정상 종료했습니다. SDPA의 배치16·입력2048·생성512 조건과 오늘의eager/SDPA기본형상 모두 첫forward이후outside312MiB였습니다.** 11월8일과동일기본형상의차이도0MiB입니다. 사용자실용기준100MiB이내를충족했습니다. 조건별새실행은1회라반복안정성확인/통계적동등성검정은아닙니다.

**기존성공모델실행까지총190개프로세스·1375개추론후안정process-stage를재계산했습니다.** LAZY312MiB상수의최대오차/과소추정2MiB, EAGER1040MiB상수의최대오차/과소추정26MiB이며100MiB초과프로세스는0개입니다. 프로세스가많아도독립GPU환경표본은RTX3090한개입니다.

현재조사범위에서단일GPU·고정driver·모듈로딩정책별상수로충분하다는근거가확보됐습니다. 추가로큰모델/같은조건반복을계속늘리기보다이제AWS의독립GPU환경표본확대로넘어갈시점입니다. 미측정GPU예측정확도를검증한것은아닙니다.

## 새5개실행

|attention|조건(B/T/N)|관측reserved최대|관측해당PID NVML최대|추론후outside|
|---|---|---:|---:|---:|
|eager|1/256/64|14878MiB|15190MiB|312MiB|
|sdpa|1/2048/64|17074MiB|17386MiB|312MiB|
|sdpa|1/256/512|14914MiB|15226MiB|312MiB|
|sdpa|1/256/64|14878MiB|15190MiB|312MiB|
|sdpa|16/256/64|19606MiB|19918MiB|312MiB|

최대값은각각독립적인기술열이며그차이를outside로계산하지않았습니다.
outside는동일조회구간NVML해당PID−실제PyTorch reserved로먼저계산했습니다. allocated차감·xMem오차흡수·결측/음수0치환을하지않았습니다. loading연속관측이없고짧은transient peak를놓칠수있어실제최대보장은아닙니다.

실제worker구간 2026-11-10T18:49:04.251214+09:00 ~ 2026-11-10T18:56:09.941793+09:00. 이전날짜대조는2026-11-08의qwen-attention-01(동일기본형상각3프로세스)이며이번오늘각1프로세스와별도비교했습니다. 오늘shape입력hash/케이스/revision/환경은이전qwen-shape-01과같았고attention은설계대로달랐습니다.

고정조건: Qwen/Qwen2.5-7B revision272aa6d30271028cdf113345b1e82a660437edaa, case BF16, LAZY, direct, TF32false, cudnn_benchmarkfalse, native allocator,4CPU threads, timeout900초. 새GPU처음접근을예측필수입력으로바꾼것이아니며이실험은분석용실측입니다.

새조건에서는SDPA자동선택을사용했고내부kernel을직접프로파일링하지않았습니다. 이전기본형상진단의Flash선택을배치16/입력2048/생성512에까지일반화하지않습니다.

## 상수적용오차: 전체190개성공모델실행

|실제모듈모드|고정outside|프로세스수|프로세스중앙값MAE|경험적절대오차p95|최대stage절대오차/과소추정|100MiB초과|
|---|---:|---:|---:|---:|---:|---:|
|LAZY|312MiB|171|0.211MiB|2MiB|2MiB|0|
|EAGER|1040MiB|19|4.105MiB|26MiB|26MiB|0|

LAZY최대2MiB는비전모델314MiB, EAGER최대26MiB는Qwen1066MiB에해당합니다.
LAZY171개/EAGER19개의불균형과모델·suite별반복수차이가있습니다. MAE/p95는서로다른GPU표본의일반화성능이아니고같은GPU의프로세스기술통계입니다. 시간샘플을독립표본으로세지않았으며CI/p값을계산하지않았습니다.

집계대상은RTX3090동일UUID/driver570.211.01의status ok model worker입니다. core연산대조군·smoke·실패·진단·AWS는주평가에서분리했습니다. 초기화/모델로딩단계도CSV에보존했지만상수주평가는after_first_forward부터final_idle까지입니다. 성공worker도품질통과샘플만사용했습니다. FP16 Qwen실패를성공추론으로포함하지않았습니다.

LAZY312/EAGER1040은이전GPT2결과에서고정했으며이번190개오차를최소화하도록재적합하지않았습니다. 원래데이터를이미본후의사후기술평가이므로홀드아웃모델/GPU예측검증이라고주장하지않습니다. 단순상수를우선할근거이며복잡한학습의개선근거는아직없습니다. 기준값출처는baseline_definition.json에보존했습니다.

## 품질·재현확인

새실험orchestration complete,5개exit0, child ok5, 고유PID5, canonical UUID1, 실제CUDA모듈모드LAZY이며tmux로그/종료코드0을보존했습니다. source/input hash는매worker전및최종감사에서일치했습니다. 기존probe/criteria/image/driver/cache/과거실패자료는변경하지않았습니다. inuos CPU작업에접근·중단·환경변경하지않았습니다.

새raw6253개=stable1470+running4783. 초기화전예상결측210개(5×2×21), 추가제외3개는전부n512실행중query>5ms입니다(5.845176/179.306081/5.208189ms). 179ms지연도원래기준으로제외하고원본을유지했습니다. 음수outside/다른GPU PID/주조회API오류/bracket변화는새샘플에서발견되지않았습니다.

초기화후stable60개행은각21clean(총1260),추론후35개행은모두312MiB입니다. last10span0MiB. 기존analyse_probe.py의새stable70행(예상결측단계포함)은독립재계산과clean개수/median이전부일치했습니다. 조건별1회이므로이번새실험만으로3반복범위를주장하지않습니다.

환경:
RTX3090 24GiB, GPU-17eb7299-8b39-914d-7c5b-26908502904f, driver570.211.01, Python3.11.11, torch2.6.0+cu124, Transformers4.52.4, CUDA build/runtime12.4(드라이버지원12.8과구별).

고정image: sha256:457da255e6e2de7ee05ff7814674c90abcf6d1ed04e73df3cc4eebcdb31fb032
probe SHA256: 7df9a02ce9ca938b9702ee2c9fd7d0414d0a4cfc0cfe5ab87c97d7a069243d88

소프트웨어변형suite도그룹별imageID/runtime를CSV에보존했습니다. 과거wrapper소스가나중에수정된경우까지현재파일과같다고가정하지않았고,각worker기록probe hash와raw를기준으로재계산했습니다. 새실험의source/input은현재파일과직접대조했습니다.

## 산출물과AWS인계

- 설계/실행/감사: outside-lab/LOCAL_CLOSEOUT_DESIGN_KO.md, run_local_closeout.py, audit_local_closeout.py, local-closeout-plan.json
- raw: results/local-closeout-20261110/ 및각child의manifest/samples/events/로그/이미지inspect
- 실행로그/종료: local-closeout-20261110.log, local-closeout-20261110.exitcode(0)
- 전체로컬CSV: all_local_process.csv, all_local_stable_stage.csv, constant_error_summary.csv, environment_summary.csv
- 새조건/날짜/품질CSV: today_vs_sdpa_base.csv, cross_day_comparison.csv, new_quality_audit.csv
- provenance/hash: run_provenance.csv, baseline_definition.json, excluded_manifests.csv, evidence_sha256.csv, artifact_sha256.json
- 기존분석대조: results/local-closeout-20261110-probe-analysis/

AWS준비점검은루트AWS_NEXT_STEP_20261110.md를참조하십시오. 사용자가인스턴스를직접준비하기로했습니다. 다음표본은기존L4이후A10G(g5)입니다. 접속대상이준비되면SSH/SSM정보로번들전송·사전점검·실험을이어갈수있습니다. 로컬AWS CLI는없고유료자원을생성/시작하지않았습니다.

기존aws-bundle.tar.gz의10개파일은디렉터리내용과일치하고probe hash/구문검사가통과했습니다. 요약문서에있던자동shutdown문구는실제코드에없어정정했습니다. 실험후결과회수와콘솔Stop을명시한절차를정리했습니다. 실제imageID/전이의존성/driver가이전과같다고가정하지않고새manifest로대조해야합니다. 가격/AMI유효성/할당가능여부는접속·실행단계에서확인할항목입니다.

로컬결론의범위는현재GPU·driver·조사한모델/shape/backend입니다. 이후AWS에서GPU간차이와driver교란을구분하고leave-one-GPU-out에필요한독립GPU환경표본을확보해야합니다. 기존에측정한GPU값을조회하는것과미측정GPU예측정확도를혼동하지않습니다.

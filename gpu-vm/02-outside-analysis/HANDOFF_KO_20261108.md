# CloudGPUFit outside 인계 업데이트

2026-11-08, inugpu@gpuvm-1에서 실제 자료 확인 및 CPU 분석 완료.

- Qwen BF16 본 실험: suite complete, 12/12 exit0, child ok12, PID12, GPU 환경1. smoke는 별도입니다.
- GPT-2/Qwen 모두 outside 254→296→312MiB. 첫 forward 이후 312 유지. 안정 process-stage150개, 반복 그룹50개, 각 중앙값 반복 범위0. 실행 이력 공통72쌍 차이0.
- raw7802개 중 초기화 전 예상 결측504개, running 품질 제외2개. 음수·예상 밖 결측·다른 PID·측정 API 오류 없음. 기존 분석240행 clean count/median과 독립 재계산 일치.
- large-01(11월4일) GPT-2 FP16와 본 BF16(11월8일) 공통75쌍 outside 차이0. 별도 suite/date 비교입니다.
- FP16 Qwen smoke는 nonfinite logits 실패이며 삭제/재시도하지 않았습니다. 실패 이전 관측은 partial CSV로 별도 보존했습니다.
- 원시·image·probe·criteria·cache 및 inuos 환경은 변경하지 않았습니다. GPU 실험이나 AWS 리소스를 시작하지 않았습니다.

보고서: results/qwen-bf16-01-independent-audit/REPORT_KO.md
기존 분석기 출력: results/qwen-bf16-01-analysis/
독립 감사/CSV/hash: results/qwen-bf16-01-independent-audit/
감사 소스: outside-lab/audit_qwen_bf16.py (출력 디렉터리가 이미 있으면 중단하여 덮어쓰기 방지)

다음 작업: outside-lab/LOADING_SENSITIVITY_DESIGN_KO.md의 LAZY/EAGER 비교. 실행기 outside-lab/prepare_loading_sensitivity.py 및 loading-sensitivity-plan.json 준비. CPU dry-run/구문/블록 검증만 수행했으며 실행 전입니다. 3블록×2모드×2이력=12 workers. 실행 후 mode/블록을 보존한 독립 비교가 필요합니다.

한계: 단일 GPU의 새 프로세스 반복이며 새 GPU 예측 정확도는 미검증. 학습보다 환경 민감도·독립 GPU 표본 확보가 우선입니다. CPU-before-rental 목적과 xMem/outside 분리, 같은 조회구간 NVML−실제reserved 정의를 유지하십시오.

추가 대조: large-01은 CUDA_MODULE_LOADING 미설정, 본 suite는 LAZY 명시. late CUDA API 실제 모드는 둘 다 LAZY입니다. 정밀도 비교의 환경 기록 차이로 보존했습니다.

## C단계 LAZY/EAGER 실행·분석 완료

- loading-sensitivity-01: 3블록×2모드×2이력=12 worker 모두 ok/exit0; 전체 종료0. GPU환경1, PID12.
- LAZY outside254→296→312MiB, EAGER824→824→1040MiB. 초기화/로딩/첫forward/최종 차이+570/+528/+728/+728MiB가 모든블록/이력에서 재현. reserved는 대응 stable 단계에서 동일.
- raw5970, 예상 preinit결측504, 추가제외5(query4/bracket1). stable150행 clean20~21, 반복그룹50 범위0. 원본보존. 독립분석240행이 기존분석기와 일치.
- 보고서 results/loading-sensitivity-01-analysis/REPORT_KO.md, 파생CSV/증거hash 동봉. 기존분석 results/loading-sensitivity-01-probe-analysis/.
- actual runtime mode 확인, recorded environment 차이는 CUDA_MODULE_LOADING 하나. 다른 module data loading 내부 기본효과까지 분리한 것은 아님.
- GPU만의 outside 상수는 설정을 포괄할 수 없음. LAZY312를 EAGER에 적용하면 이 조건에서728MiB 과소추정. 복잡한학습 시작보다 설정조건 명시와 확인실험 우선.
- 다음 제안: GPT2 BF16 direct, 10블록×2모드=20 fresh worker 균형순서 확인. 아직 실행하지 않음. 이후 Qwen 확장 검토.

## C단계 10블록 확인 완료

- loading-confirmation-01: GPT2 BF16 direct,10균형블록×LAZY/EAGER=20fresh worker/컨테이너 모두ok/exit0, orchestration complete/전체종료0. PID20, GPU환경1.
- outside LAZY254→296→312MiB, EAGER824→824→1040MiB. 첫forward/최종차이728MiB가10블록 모두 재현. 초기화570/로딩528차이도 전부동일. mode/stage24그룹 각10반복범위0.
- raw8879: stable5880/running2999, 예상preinit결측840, 추가query>5ms제외3(stable2/running1). 음수/bracket변화/다른PID/API오류없음. stable240행 clean20~21, last10span0. 독립감사380행 기존분석과일치.
- 탐색direct와입력hash/환경/케이스/이미지/소프트웨어대조일치. 탐색과확인풀링안함.
- 보고서 results/loading-confirmation-01-analysis/REPORT_KO.md, derived CSV/hash 동봉. 원본고정probe/criteria/image/cache와 과거실패자료유지.
- 다음제안 Qwen7B BF16 direct,3대응블록×2모드=6fresh worker(미실행). 이후독립GPU/환경표본확보우선. GPU단일환경확인을미측정GPU정확도로해석하지말것.

## Qwen 7B 모듈 로딩 후속 비교 완료

- qwen-loading-01: Qwen7B BF16 direct,3블록×LAZY/EAGER=6 worker/컨테이너 모두ok/exit0, 전체종료0. UUID1/PID6.
- outside LAZY254→296→312MiB, EAGER824→824→1066MiB. 첫forward/최종 모드차이754MiB가3블록 모두 재현. mode/stage24그룹 각3반복범위0.
- GPT2 확인10블록 EAGER1040MiB 대비 Qwen은26MiB 높고 LAZY는같음. 기록환경은일치하지만 별도suite/시점/모델/입력 비교이므로 순수모델 인과효과 확정금지.
- raw4310, preinit예상결측252, 추가제외running bracket변화1. stable72행 모두clean21, span0. 음수/지연5ms초과/다른PID/API오류없음. 독립114행 기존분석과일치.
- 보고서 results/qwen-loading-01-analysis/REPORT_KO.md; CSV/cross_model_comparison/hash동봉. 원시와FP16실패등보존.
- 다음: 모델×모드 동시기2×2 통제 비교 또는 backend요인 비교 후보. 이후독립GPU환경표본확보. 추가큰모델증가보다통제요인/일반화분리에집중. 후속실험미실행.

## 사용자100MiB 실용기준 및 attention비교 완료

- 사용자: outside차이100MiB이하는거의고정으로판정. 기존criteria16MiB는유지하고 practical-outside-criteria.json으로별도기록(이전결과후선정).
- qwen-attention-01: QwenBF16/LAZY/direct/B1T256N64, eager attention대SDPA 3대응블록=6worker 모두ok/exit0. 분리진단2개도ok/exit0. 전체종료0. outside반복수6/GPU환경1, 진단은미포함.
- 두attention 모두 초기화254→로딩296→첫forward이후312MiB. 추론이후42개process-stage중앙값전부312/범위0, 블록차이0. 사용자100MiB기준실용적으로거의고정.
- 분리CPUoperator프로파일:SDPA 첫forward aten::_scaled_dot_product_flash_attention28회, generate64 1792회. eager는softmax28/1792회. 본worker매호출직접추적은아니며다른shape/GPU선택보장아님.
- raw4249, 예상preinit결측252, running제외7(bracket4/query3), stable72행clean21. 음수/다른PID/API오류없음. 독립114행기존분석과일치.
- 보고서 results/qwen-attention-01-analysis/REPORT_KO.md, 파생CSV/진단연산/hash동봉. 원시/캐시/고정probe/criteria/image/과거FP16실패보존.
- 100MiB기준으로이전모델차이0~26은거의고정,모듈정책차이728~754는고정아님. LAZY초기화~추론범위58도100이내,EAGER단계증가216~242는100초과.
- 다음후보:현재미검증인batch/입력길이/생성길이변화또는동일이미지독립GPU환경확장. 추가실험/유료자원미실행.

## 배치·입력길이·생성길이 비교 완료 (Claude Code 진행)

- qwen-shape-01: QwenBF16/LAZY/eager/direct, base B1T256N64 · b16 · t2048 · n512 · b4t1024n256, 3블록×5=15worker 모두 ok/exit0. PID15/GPU환경1.
- outside 전조건 254→296→312MiB, 조건−base 차이 전부0MiB(100MiB·16MiB 기준 모두 충족). running 범위296~312.
- reserved 최대 14,878~20,686MiB로 workload 5.8GiB 증가에도 outside 불변.
- device used−process NVML=448.9MiB 일정(드라이버 Reserved442MiB). outside와 별개, 용량 판정 시 명목총량에서 차감 검토 필요(bench_worker_v3 memory guard는 명목 total 사용).
- 보고서 results/qwen-shape-01-analysis/REPORT_KO.md. 실행기 outside-lab/run_qwen_shape.py, 분석기 analyse_qwen_shape.py(python3 -I 불가: 같은 디렉터리 모듈 import).
- 다음: 사용자 계획대로 AWS 단계. 동일 이미지/프로토콜, GPU별 outside + device gap(드라이버 예약분) 함께 기록.

## 소프트웨어·이미지 민감도 완료 (RTX 3090)

- S1 base이미지, S2 cu126, S3 torch2.5.1, S4 torch2.8.0+cu128 각각 S0 대조 12worker, 총48 모두 exit0. 실제 runtime 12.4/12.6/12.4/12.8 확인.
- 전 변형 outside 254→296→312MiB, Sx−S0 전부0MiB. device gap 448.9MiB 동일.
- 결론: 3090/driver570/LAZY에서 이미지·torch·CUDA runtime·cuBLAS/cuDNN 영향 없음. outside 변화 요인은 지금까지 모듈로딩(LAZY/EAGER)과 GPU뿐. 남은 소프트웨어 후보 driver → AWS에서 AMI(driver) 비교 우선.
- 보고서 results/software-summary-01/REPORT_KO.md, 개별 results/software-s{1..4}-01-analysis/. 실행기 outside-lab/run_software.py, 분석 analyse_software.py, 설치 images/install_variant.sh(마운트 방식), variants.json.

## AWS g6.xlarge(L4) 파일럿 완료
- outside 186→216→232MiB, S0/S4·모델 차이 0, 반복 범위 0. 기존 L4 232와 일치. device gap 476.9MiB. 보고서 results/aws-g6-xlarge-01/REPORT_KO.md

## 2026-11-10 로컬 마무리 및 AWS 인계

- local-closeout-20261110: QwenBF16/LAZY/direct, eager기본+SDPA기본/배치16/입력2048/생성512 총5개 모두ok/exit0. 조건별1회인소규모점검.
- 새조건전부추론후outside312MiB, 11월8일동일기본형상과차이0. 입력/환경hash도대조. 실제날짜11월10일확인.
- 전체성공local model190개(단일3090환경),추론후stable process-stage1375개재계산: LAZY171개기준312MiB최대오차/과소추정2MiB, EAGER19개기준1040MiB최대26MiB.100MiB초과0. 사후기술평가이며미측정GPU검증아님.
- 새raw6253,preinit예상결측210,running query>5ms제외3(최대179.306ms). stable60행clean21,last10span0. 독립70stable행기존분석과일치.
- 보고서 results/local-closeout-20261110-analysis/REPORT_KO.md,전체/조건/날짜CSV및증거hash보존.
- 로컬추가실행은마무리. 사용자가AWS인스턴스직접준비선택. 다음A10G(g5),접속주소/사용자/키경로또는SSM정보대기. 유료자원미기동.
- AWS_NEXT_STEP_20261110.md에번들검증/전송/실행/결과회수/콘솔Stop절차정리. 기존번들에는자동shutdown코드없음(과거요약정정). awsCLI현재미설치. 원본번들미수정.

## AWS 수동 실행 패키지 제공 (2026-11-10)

- 최신 사용자 지시: AWS 인스턴스 생성·실험 실행은 사용자 직접 수행. 에이전트는 AWS접속/실험실행하지 않고코드제공·결과분석만함.
- cloudgpufit-aws-manual-20261110.zip 생성(원본aws-bundle보존), README_KO에us-east-1/g5.xlarge/기존L4 AMI우선+공식Ubuntu22대체조회/EBS150GiB gp3/SCP/수동시작/회수/수동Stop기록.
- 원본probe/revision유지. packagehash검사,1GPU검사,source변경검사,worker상태/UUID/실제LAZY검사,로그/종료코드·실패아카이브보존. 자동중지/기동/Terminate코드없음.
- Python/셸구문,정상12회·저VRAM6회·첫실패시중단의CPU모의검사,ZIP추출후hash검사통과. 실제AWS/GPU실행하지않음.
- ZIP SHA256 a2a3f7e6f56fce1b736f01e744aa46acaf40c7e49488f474210d8ca71ae53826. AMI현재availability는여기서AWS API미확인,사용자CloudShell조회명령제공.

## AMI 사용자 정정

- 사용자는 지금까지 선택한 AMI가 ami-0bc11c94e8c4894e4, 표시이름 Deep Learning OSS NVIDIA Driver AMI GPU PyTorch2.13(Ubuntu26.04)라고 확인함. 앞으로 이 선택을 우선하며Ubuntu22.04로임의대체하지않음. 표시이름/현재가용성/리전은API미검증.
- 로컬보관 aws-results/g6.xlarge-20261108T125436Z.tar.gz 내부env/instance.txt는 ami-0d220f0a0f14edbef로기록. 이출처를사용자현재AMI와동일로단정해안내한것이오류. 원시보존하고instance-id/리전/driver로후속실행구분필요.
- README와AMI조회도우미를사용자ID로정정. ZIP/manifest/hash재생성. AWS실행안함.

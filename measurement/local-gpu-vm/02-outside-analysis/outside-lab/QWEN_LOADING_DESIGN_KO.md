# Qwen 7B BF16 모듈 로딩 민감도: 실행 전 설계

2026-11-08. GPT-2의 LAZY/EAGER 추론 outside 차이728MiB가 다른모델에서도 재현되는지 확인하는 별도 탐색 suite입니다. GPU 아키텍처 일반화나 순수 파라미터수 개입은 아닙니다.

고정모델 Qwen/Qwen2.5-7B, revision272aa6d30271028cdf113345b1e82a660437edaa, case BF16/B1/T256/N64, direct, eager attention, TF32false, native allocator,4CPU threads, timeout900초. 기존 probe/image/driver/criteria/cache는 그대로 사용합니다. FP16 실패는 별도 보존하고 다시 시도하지 않습니다.

3개 대응블록×LAZY/EAGER=6 fresh worker/컨테이너를 모두순차실행합니다. LAZY→EAGER2블록, EAGER→LAZY1블록을 seed20261110으로 섞습니다(seed는 식별정수). 홀수블록이라 완전균형은 아니며 순서효과의 잔여가능성을 기록합니다. 실제 independent_repeat는 글로벌block0~2입니다. direct에도 최초forward/generate/warmup/반복추론이포함되며 cold비교는 아닙니다. 환경변수 CUDA_MODULE_LOADING만 의도적으로 변경하며 CUDA_MODULE_DATA_LOADING은 미설정 유지, 실제lateCUDA API 모드확인합니다.

주요단계 after_cuda_init/after_model_load/after_first_forward/final_idle의 cleanstable process중앙값, 블록내EAGER−LAZY차이, 반복범위를 보고합니다. 첫추론이후 GPT2의728MiB와 같을것이라고 가정하지않습니다. 모드효과는 새Qwen블록안에서대응하고, 기존GPT2확인10블록과 과거QwenLAZY결과는 날짜/suite/모델이 다른기술비교로만둡니다.

기존criteria.json의 stable21개/50ms, clean최소10, last10span≤8MiB, query>5ms/bracket변화/음수/다른PID/API오류제외, 반복범위와의미효과16MiB를유지합니다. 결측과음수를0으로대체하지않으며 pairedNVML해당PID−실제reserved로outside를구합니다. 독립최대값차감이나allocated차감은하지않습니다. 초기화전예상결측은실패와구별하고 raw/실패자료를보존합니다.

3반복은탐색적프로세스반복이며 GPU환경표본은1입니다. 시간샘플기반CI/p값이나새아키텍처예측정확도를주장하지않습니다. source/input hash를시작및각worker전에검사하고, 실패/timeout/실제모드불일치시자동중단하며재시도하지않습니다. timeout시직접생성한컨테이너만정지합니다. tmux지속세션,로그,종료코드를보존합니다. 캐시는읽기전용이고패키지/driver/image/inuos환경을변경하거나유료AWS를시작하지않습니다.

완료후suite/manifest/exitcode/UUID/모델dtype/revision/image/driver/입력hash/실제모드와원시품질을감사하고 한국어보고서·CSV·hash·인계노트를새경로에저장합니다. 여기서정한6개이상의실험은결과를본후별도결정합니다.

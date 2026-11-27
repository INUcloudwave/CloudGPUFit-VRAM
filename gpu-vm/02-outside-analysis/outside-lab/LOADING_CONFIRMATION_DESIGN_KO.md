# LAZY/EAGER 10블록 확인 실험 사전 설계

2026-11-08 작성. 탐색 loading-sensitivity-01과 분리한 새 suite이며 결과를 보기 전에 고정합니다.

목표: 같은 RTX3090/고정 이미지/GPT-2 BF16 direct에서 LAZY와 EAGER의 단계별 outside 차이를 각 모드10개 fresh worker로 확인합니다. 독립 GPU 환경은1개입니다.

10개 블록마다 두 모드를 순차 실행합니다. LAZY→EAGER 5블록, EAGER→LAZY 5블록을 고정 seed20261109로 섞습니다(seed는 날짜가 아닌 식별 정수). 글로벌 repeat는 block0~9입니다. 총20개 별도 컨테이너/worker이고 동시 실행하지 않습니다. direct는 기존과 같이 첫 forward, generate, warmup, 3개 반복 블록을 수행합니다. performance_first만 생략하며 cold 실험으로 해석하지 않습니다.

고정 요소: 기존 probe SHA256 7df9a02ce9ca938b9702ee2c9fd7d0414d0a4cfc0cfe5ab87c97d7a069243d88; 고정 이미지 sha256:457da255e6e2de7ee05ff7814674c90abcf6d1ed04e73df3cc4eebcdb31fb032; GPT2 revision607a30d783dfa663caf39e06633721c8d4cfcd7e; case dtypeBF16/B1/T256/N64/eager attention/TF32 false/native allocator/threads4/timeout900초. CUDA_MODULE_DATA_LOADING은 미설정. 바꾸는 환경변수는 CUDA_MODULE_LOADING뿐입니다. 새 실행기는 동일 Docker mount/환경 설정으로 probe를 직접 실행하며 launcher/실험 식별·시간의 차이는 기록하고 기존 탐색과 풀링하지 않습니다.

주요 단계는 after_cuda_init, after_model_load, after_first_forward, final_idle. clean stable process 중앙값과 블록 내 EAGER−LAZY 차이, 각 모드10반복 범위를 보고합니다. 원래 criteria.json의 21개/50ms, clean최소10, last10 span≤8MiB, query>5ms·bracket 변화·음수·다른PID·API오류 제외, 반복범위16MiB 및 의미있는 차이16MiB를 유지합니다. 모든10블록의 주요단계 차이가 같은 방향으로16MiB 이상인지 사전 확인 항목으로 사용하며 원래 기준을 유리하게 조정하지 않습니다. 이 항목은 탐색 결과 +728MiB를 본 뒤 선택한 확인 설계이지 완전한 최초 사전등록이 아닙니다.

시간 샘플 기반 CI/p값은 산출하지 않습니다. 10개 프로세스 반복을 GPU10개로 세지 않습니다. 안정값이 모두 같아도 미래 변동0을 보장하지 않습니다. 최대 reserved/NVML은 별도 기술열이고 outside는 같은 관측구간 NVML해당PID−실제reserved로 계산합니다. missing/negative를0으로 바꾸지 않고 원본을 보존합니다.

고정 source/input hash는 실행 전 orchestration에 기록하고 매 worker 전 불변을 검사합니다. smoke/실패를 반복으로 대체하지 않습니다. worker 실패/timeout/실제모드 불일치 시 자동 중단하며 재시도하지 않습니다. timeout이면 이 실행기가 생성한 이름의 컨테이너만 정지합니다. 다른 작업·driver·패키지·이미지·모델 캐시를 변경하지 않습니다. 유료 AWS를 시작하지 않습니다.

실행은 tmux 지속 세션, 로그와 종료코드 파일을 사용합니다. 완료 후 각 suite/child manifest/exit code, case dtype/revision/UUID/image/driver/runtime mode/source/input hash, raw 품질을 감사하고 한국어 보고서/CSV/인계노트를 새 경로에 저장합니다. 기존 탐색12개는 별도 비교표로만 둡니다. 후속 Qwen 재현은 이번20개 결과 확인 후 별도 결정하며 이번 작업 범위에는 포함하지 않습니다.

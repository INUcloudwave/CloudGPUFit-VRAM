# Qwen BF16·LAZY attention backend 비교 사전 설계

2026-11-08. 사용자 지시에 따라 outside 차이100MiB 이하는 실용적으로 거의 고정이라고 평가합니다. 이 기준은 앞선 결과를 본 뒤 사용자가 정했으며 원래 criteria.json(효과16MiB 등)은 덮어쓰지 않습니다. 새 practical-outside-criteria.json으로 구분합니다. 통계적 동등성 검정이나 미래오차 보장이 아닙니다.

Qwen/Qwen2.5-7B revision272aa6d30271028cdf113345b1e82a660437edaa, BF16, B1/T256/N64, direct, RTX3090/고정이미지/driver, LAZY,4 CPU threads, TF32false/native allocator/timeout900초를 고정합니다. 바꾸는 것은 --attention eager 또는 sdpa뿐입니다. CUDA_MODULE_LOADING은 전부LAZY, CUDA_MODULE_DATA_LOADING은 미설정 유지합니다. 일반CLI dtype 기본값 대신 case dtypeBF16을 기준으로 합니다.

3개 대응블록×2 attention=6개 fresh worker/컨테이너를 순차 실행합니다. eager→sdpa2블록, sdpa→eager1블록을 seed20261111로 섞습니다(seed는 날짜가 아닌 식별정수). 홀수블록이라 완전 균형은 아니며 이 한계를 명시합니다. 각 worker의 repeat는 글로벌block0~2입니다. direct는 첫forward/generate/warmup/반복추론을 포함하며 cold비교가 아닙니다. 고정 probe 원본은 수정하지 않습니다.

주요 결과는 첫forward부터final_idle까지 clean stable process 중앙값과 블록내SDPA−eager차이, 모드별 반복범위, 모든 비교조건을 합친 process중앙값 범위입니다. 각 주요단계에서 모든 블록의 절댓값차이≤100MiB인지 평가하고 비교대상 전체 process중앙값 범위≤100MiB도 별도 보고합니다. 초기화/모델로딩은 보조단계로 분리합니다. running은 transient 기술값이며 stable과 섞지 않습니다.

품질 기준은 기존 stable21개/50ms, 최소clean10, last10span≤8MiB, query>5ms/bracket변화/음수/다른PID/API오류 제외를 유지합니다. outside는 같은 관측구간의 NVML해당PID−실제reserved로 구하며 allocated차감·xMem오차흡수·독립최대값차감·결측0치환을 하지 않습니다. 음수/결측/실패/원본을 보존하고 초기화전 예상결측과 실제실패를 구별합니다.

실제 SDPA 구현 확인은 본6개 측정이 끝난 뒤 별도 diagnostic 프로세스를 attention별1회(총2회) 순차 실행합니다. 동일 고정모델/입력seed/forward 및generate64를 CPU operator profiler로 관찰하고 aten attention backend operator 이름/count와 Transformers 실제 설정/로컬분기소스를 저장합니다. 메모리 profiler는 끄며 diagnostic의 메모리는 outside분석에 사용하지 않습니다. 진단은 별도프로세스이므로 본측정의각호출선택을직접관측했다고 주장하지않습니다. 첫forward와generate가서로다른SDPA구현을선택할수있으므로단계별로보고합니다.

tmux지속세션·로그·종료코드·source/input hash를 보존합니다. worker/diagnostic 실패·timeout·실제CUDA모드불일치시중단하며자동재시도하거나backend를강제로변경하지않습니다. timeout이면이번실행기가만든컨테이너만정지합니다. cache는읽기전용, 패키지/driver/image/원본probe/criteria/inuos작업은변경하지않고유료AWS를실행하지않습니다.

완료후 manifest/계획수/exitcode/PID/UUID/dtype/revision/image/driver/입력hash/조회품질을독립감사하고한국어보고서·CSV·hash·인계노트를새경로에저장합니다. 프로세스반복은GPU환경반복이아니며 GPU환경표본은1입니다. 추가실험은이결과를본후별도결정합니다.

# 로컬 마무리 점검 (2026-11-10)

목적: AWS로 넘어가기 전 SDPA의 형상변화와 날짜가 다른 실행의 재현성을 빠르게 점검하고, 기존 전체 raw에 고정 outside 상수를 적용한 오차를 집계합니다.

본 실험은 Qwen BF16·LAZY·direct·고정 RTX3090/driver/image/probe에서 5개 fresh worker입니다. eager/base(B1T256N64), sdpa/base, sdpa/b16(B16T256N64), sdpa/t2048(B1T2048N64), sdpa/n512(B1T256N512)를 seed20261110으로 섞어 순차 실행합니다. 조건별1회뿐이며 반복 안정성 확인이나 통계적 동등성 증명은 아닙니다. 기존11월8일 suite들은 별도 날짜/suite의 기술 비교로 유지합니다. 1개 GPU의 날짜 반복을 GPU환경표본으로 세지 않습니다.

첫 forward부터 final_idle까지의 clean stable process 중앙값을 주요 결과로 합니다. SDPA형상−오늘SDPA기본형상, 오늘eager/SDPA기본형상−11월8일동일조건, LAZY312MiB상수에 대한 signed/absolute error 및 최대과소추정을 보고합니다. 사용자 practical 기준≤100MiB, 기존품질기준stable21/50ms·최소clean10·last10span8MiB·조회5ms/bracket변화/음수/다른PID/API오류제외는 유지합니다. 원본criteria/probe/image/driver/cache는 변경하지 않습니다.

outside는 동일관측구간NVML해당PID−실제reserved이며 independent maxima차감/allocated차감/xMem오차흡수를 하지 않습니다. 결측·음수·실패를 보존하고 초기화전 예상결측을 따로 집계합니다. 실패/timeout은자동재시도하지않으며직접생성한컨테이너만정지합니다. 측정은tmux지속세션으로하고로그·종료코드·source/inputhash를보존합니다.

SDPA는자동선택설정입니다. 기존기본형상별도진단은Flash연산을확인했지만 이번큰형상에서실제내부kernel을추적하지않으므로Flash로단정하지않습니다. 속도측정은목적이아닙니다.

기존raw재분석은같은RTX3090의성공model worker에서첫forward이후안정단계만주평가합니다. LAZY312MiB/EAGER1040MiB 기준은기존기본형상GPT2결과에서동결하며이번데이터로재적합하지않습니다. 이미본결과에대한사후기술평가이며미관측GPU의홀드아웃예측검증이라고주장하지않습니다. 프로세스단위요약과stage별오차를분리하고시간샘플을독립표본으로세지않습니다. 초기화/로딩/연산대조군·FP16실패·smoke·diagnostic·AWS는주집계와구분합니다.

AWS는다음단계준비물/이전L4결과/실행·회수·중지절차를점검하되유료AWSGPU를자동시작하지않습니다. 최초사용자운영규칙의유료자원자동실행금지를유지합니다. inuos CPU작업에는접근·중단·환경변경하지않습니다.

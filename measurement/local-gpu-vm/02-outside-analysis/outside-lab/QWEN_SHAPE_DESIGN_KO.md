# Qwen BF16·LAZY 배치·입력길이·생성길이 비교 사전 설계

2026-11-08. 목적: 같은 GPU·소프트웨어에서 워크로드 형태(B/T/N)를 바꿔도 첫 forward 이후 outside가 사용자 실용 기준(차이 ≤100MiB, practical-outside-criteria.json) 안에 머무는지 확인. 원래 criteria.json(16MiB)은 변경하지 않고 함께 보고.

고정: Qwen/Qwen2.5-7B rev 272aa6d3…, BF16, eager attention(이전 suite와 연속성), CUDA_MODULE_LOADING=LAZY, DATA_LOADING 미설정, direct, 4 threads, TF32 false, 고정 이미지/driver/RTX3090, timeout 900s, probe.py 원본(sha 7df9a02c…).

조건(case): base B1/T256/N64(이전 suite와 동일), b16 B16/T256/N64, t2048 B1/T2048/N64, n512 B1/T256/N512, b4t1024n256 B4/T1024/N256(복합). 한 번에 한 요인 + 복합 1개.

설계: 3블록 × 5조건 = 15 fresh worker/컨테이너 순차 실행. 블록마다 조건 순서를 seed 20261112로 독립 셔플. 각 worker repeat=block.

주요 결과: after_first_forward~final_idle clean stable process 중앙값에서 블록 내 (조건−base) 차이, 각 단계에서 모든 블록 |차이|≤100MiB 여부, 전체 조건 process 중앙값 범위. 초기화/모델 로딩은 보조. 보조 기록: reserved 최대(워크로드 크기 확인용), device used−process NVML 간격(드라이버 예약분, outside와 별개).

품질 기준·outside 정의는 기존과 동일(stable 21×50ms, clean≥10, last10 span≤8MiB, query>5ms/bracket 변화/음수/다른 PID/API 오류 제외, 원본 보존, 0 치환 금지).

실패/timeout/OOM 시 중단, 자동 재시도·조건 변경 없음. 결과는 단일 GPU 환경 1개의 프로세스 반복이며 GPU 일반화가 아님.

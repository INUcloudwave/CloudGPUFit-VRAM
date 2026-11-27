# 소프트웨어·이미지 민감도 종합 (RTX 3090, 2026-11-08)

설계: outside-lab/SOFTWARE_DESIGN_KO.md. 각 변형마다 3블록 × {S0, Sx} × {GPT-2, Qwen7B} BF16 B1/T256/N64, eager, LAZY, direct = 12 fresh worker. S0는 매 suite 동시 대조군.

| 변형 | 바뀐 것 | torch / CUDA runtime(실측) | cuBLAS / cuDNN | Sx−S0 (주요 단계 42쌍) |
|---|---|---|---|---:|
| S1 | base 이미지(python slim, conda 없음), 이미지 빌드 | 2.6.0+cu124 / 12.4 | 12.4.5 / 9.1.0 | 0MiB |
| S2 | CUDA wheel 묶음 | 2.6.0+cu126 / 12.6 | 12.6.4 / 9.5.1 | 0MiB |
| S3 | torch 버전 | 2.5.1+cu124 / 12.4 | 12.4.5 / 9.1.0 | 0MiB |
| S4 | 최신 스택 전체 | 2.8.0+cu128 / 12.8 | 12.8.4 / 9.10.2 | 0MiB |

- 모든 변형·모델: 초기화 254 → 로딩 296 → 첫 forward 이후 312MiB, 반복 범위 0, plateau 미달 0
- device used − process NVML: 전 suite 448.9MiB
- 48 worker 모두 exit 0, 실제 runtime 버전이 의도와 일치(late CUDA API)

해석
- driver 570.211.01 / RTX 3090 / LAZY 조건에서 컨테이너 base·torch(2.5.1~2.8.0)·CUDA runtime(12.4~12.8)·cuBLAS/cuDNN 버전은 outside에 영향 없음
- 지금까지 outside를 바꾼 요인은 CUDA_MODULE_LOADING(LAZY↔EAGER)과 GPU(AWS 자료)뿐 → 남은 소프트웨어 후보는 driver
- 한계: 단일 GPU·단일 driver. 버전 변경은 종속 라이브러리 묶음 효과. transformers 버전·cuDNN benchmark·다른 프레임워크(vLLM 등)는 미검증

방법 기록
- S1은 빌드 이미지(id 07143fdc…)로 측정 후 디스크 부족으로 이미지 삭제
- S2~S4는 python:3.11.11-slim(sha256:081075da…) + 호스트 패키지 디렉터리 읽기 전용 마운트(PYTHONPATH) 방식, 측정 후 디렉터리 삭제. 설치 lock은 results/software-sX-01/variant-environment.lock.txt
- S2 첫 설치는 별도 pip 호출로 CUDA13 패키지가 섞여 폐기 후 단일 resolve로 재설치. S4 첫 설치는 cross-device 이동 오류로 재설치(측정 전)
- run_software.py는 S1 실행 후 마운트 모드 추가로 수정됨 → S1 분석 재실행 시 source hash 불일치(분석은 수정 전 완료)

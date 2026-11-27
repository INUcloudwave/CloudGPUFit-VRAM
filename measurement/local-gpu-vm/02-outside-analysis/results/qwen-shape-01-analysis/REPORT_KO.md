# qwen-shape-01 분석 보고서 (2026-11-08)

## 설정
- Qwen2.5-7B rev 272aa6d3, BF16, eager attention, CUDA_MODULE_LOADING=LAZY, direct, RTX 3090 / 고정 이미지 / driver 570.211.01
- 조건: base B1/T256/N64, b16 B16/T256/N64, t2048 B1/T2048/N64, n512 B1/T256/N512, b4t1024n256 B4/T1024/N256
- 3블록 × 5조건 = 15 fresh worker (seed 20261112, 블록별 셔플). 설계: outside-lab/QWEN_SHAPE_DESIGN_KO.md

## 결과
- 15/15 exit 0, manifest ok, PID 15개, GPU 환경 1개, 실제 모듈 로딩 모드 전부 LAZY
- outside: 모든 조건 254(초기화) → 296(모델 로딩) → 312MiB(첫 forward 이후 ~ final_idle)
- 조건−base 블록 차이: 주요 단계 전부 0MiB → 100MiB 기준·16MiB 기준 모두 충족
- 반복 그룹 범위 전부 0MiB, plateau 미달 0건
- running(추론 중) clean 샘플 outside 범위 296~312MiB, 312 초과 없음
- reserved 최대(MiB): base 14,878 / n512 14,916 / t2048 17,938 / b16 19,832 / b4t1024n256 20,686 → workload 메모리가 최대 5.8GiB 증가해도 outside 불변
- device used − process NVML: 전 샘플 448.9MiB (nvidia-smi 드라이버 Reserved 442MiB 대응). outside와 별개의 디바이스 단위 값

## 품질
- raw 24,618 (stable 4,410 / running 20,208), 초기화 전 예상 결측 630
- 제외: query>5ms 11건, bracket 변화 2건 (모두 running). 음수·다른 PID·API 오류 없음. 원본 보존

## 한계
- 단일 GPU 환경의 프로세스 반복. GPU 일반화 아님
- eager attention 고정(SDPA는 qwen-attention-01에서 base 형상만 확인)
- 단계 체크포인트·샘플링(50ms) 해상도 밖의 순간 변화는 관측 불가

## 파일
process_stage.csv, repeat_summary.csv, block_case_vs_base.csv, case_effect_summary.csv, device_gap.csv, quality_audit.csv, run_provenance.csv, source_verification.csv, evidence_sha256.csv, summary.json

# AWS g6.xlarge (L4) 파일럿 (2026-11-08)
- us-east-1, AMI ami-0d220f0a0f14edbef, driver 595.112.07, L4 23034MiB (FB Reserved 469MiB)
- 3블록 × {S0 2.6.0+cu124, S4 2.8.0+cu128} × {GPT-2, Qwen7B} BF16, LAZY, 12 worker 모두 exit 0
- outside: 초기화 186 → 로딩 216 → 첫 forward 이후 232MiB. 반복 범위 0, S4−S0 전부 0MiB, 모델 차이 0
- 기존 AWS L4 자료 232MiB(226–236)와 일치
- device used − process NVML = 476.9MiB 일정
- 품질 필터 제외 2샘플. 원본: aws-results/g6.xlarge-20261108T125436Z.tar.gz, 분석 outside-lab/analyse_aws.py

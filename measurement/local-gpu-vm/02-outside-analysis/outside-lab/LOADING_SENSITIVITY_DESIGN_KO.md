# C단계 첫 비교: CUDA 모듈 로딩 민감도

작성일: 2026-11-08. 준비 완료, GPU 실행 전입니다.

질문은 같은 RTX 3090와 고정 이미지에서 CUDA_MODULE_LOADING만 LAZY/EAGER로 바꿀 때 단계별 paired outside가 달라지는지입니다. 현재 두 BF16 모델 모두 254→296→312MiB이므로 첫 비교는 작은 GPT-2로 제한합니다. 새로운 GPU 아키텍처나 라이브러리별 인과효과 검증이 아닙니다.

- GPT-2 고정 revision, BF16, B1/T256/N64, eager attention, TF32=false, native allocator, 4 CPU threads, 900초 timeout을 유지합니다.
- 3개 시간 블록에서 LAZY와 EAGER 컨테이너 실행 순서를 고정 seed 20261108로 무작위화합니다. 각 컨테이너는 direct/performance_first 각각 새 worker를 순차 실행하므로 총 12개 프로세스입니다.
- 글로벌 반복 번호는 orchestration의 block입니다. child suite 내부 repeat=0을 독립 반복 번호로 합치지 않습니다. 각 child suite와 mode를 함께 보존합니다.
- 이미지 ID·probe hash는 기존 launcher가 실행 전에 검사합니다. 원본 probe/criteria/image/driver는 변경하지 않습니다. worker의 late_cuda_diagnostics 실제 module_loading 모드도 매 컨테이너 종료 후 검사하며 불일치·실패 시 중단합니다.
- CUDA_MODULE_DATA_LOADING은 미설정 상태를 유지하고 기록합니다. 의도한 환경 차이는 CUDA_MODULE_LOADING 하나입니다. baseline과 모든 환경 변수도 manifest에서 대조합니다.
- 입력 cache는 읽기 전용이며 다운로드·업데이트하지 않습니다. GPU worker는 동시 실행하지 않습니다. 다른 GPU 프로세스가 있으면 probe가 중단하며 자동 재시도하지 않습니다.

주요 결과는 after_cuda_init, after_model_load, after_first_forward, final_idle의 clean stable outside 프로세스 중앙값과 블록별 EAGER−LAZY 차이입니다. 나머지 단계, running 관측, allocated/reserved/NVML은 보조 지표로 보고합니다. 최대값은 별도 열로 유지합니다. 초기화 전 결측과 실제 실패를 구별하고 모든 raw를 보존합니다.

criteria.json을 그대로 적용합니다: stable 21개/50ms, clean 최소10개, 마지막10개 span≤8MiB, 조회>5ms 제외, bracket 변화/음수/API 오류/다른 프로세스 제외, 의미 있는 차이와 반복 범위 기준16MiB. 각 mode/history의 3개 프로세스 중앙값과 범위를 보고하고, block 내 차이가 같은 방향으로 16MiB 이상이면 후속 확인 후보로 삼습니다. 시간 샘플로 CI를 만들지 않으며 중요 차이는 이후 최소10회 확인을 검토합니다. 결과를 본 뒤 threshold를 조정하지 않습니다.

실행 준비 확인은 다음 CPU 전용 명령입니다.

```bash
cd /home/inugpu/cloudgpufit-outside
python3 outside-lab/prepare_loading_sensitivity.py
```

실제 실행 시 새 출력 경로를 지정하고 지속 세션에서 로그·종료 코드를 보존합니다. 아래는 실행 예시이며 이번 분석에서는 실행하지 않았습니다. Docker가 sudo를 요구하면 실행기 인자에 --sudo를 추가합니다.

```bash
tmux new-session -s outside-loading
cd /home/inugpu/cloudgpufit-outside
python3 outside-lab/prepare_loading_sensitivity.py --execute --output results/loading-sensitivity-01 > loading-sensitivity-01.log 2>&1
loading_exit=$?
printf '%s\n' "$loading_exit" > loading-sensitivity-01.exitcode
```

각 child/session에 analyse_probe.py를 별도 출력 경로로 적용한 뒤 orchestration block/mode로 대응시켜 비교합니다. 기존 analyser는 suite별 그룹을 분리하므로 모든 child를 한 번에 넣은 repeat_groups.csv가 3반복을 자동 구성한다고 가정하면 안 됩니다. 모드 비교는 새 블록 내 자료로 수행하고 11월8일 기존 LAZY suite는 날짜가 다른 참고값으로만 둡니다.

차이가 작으면 해당 설정에서의 안정성을 추가로 지지할 뿐 모든 환경 변화의 무효과는 아닙니다. 차이가 크면 초기화/로딩/연산 단계 중 언제 생기는지 확인하고 Qwen BF16로 확장할지 결정합니다. driver 변경, 패키지 설치, AWS 실행은 이 계획에 포함하지 않습니다.

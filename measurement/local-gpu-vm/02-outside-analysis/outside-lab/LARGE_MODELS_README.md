# 큰 모델 outside 비교

이 확장은 현재 outside-lab에 새 실행 파일과 실험 계획만 추가합니다. probe.py·pilot.py·기존 결과·판정 기준은 변경하지 않습니다.

## 목적과 조건

FP16, B1, eager attention, TF32=false, 동일 이미지·driver·GPU에서 모델 크기가 outside에 미치는 영향을 확인합니다. 단계별 raw NVML−실제 reserved를 계산합니다. xMem 예측 잔차를 사용하지 않습니다.

| 모델 계열 | 모델 | 고정 입력 |
|---|---|---|
| GPT-2 | 기본 / medium / XL | 입력 256, 생성 64 |
| T5 | small / base / 3B | encoder 256, decoder prefix 1, 생성 64 |
| Vision | ResNet-50 / ViT-large | RGB 224×224 |

8모델 × direct/performance5 선행 2절차 × fresh process 3회 = 48프로세스입니다. 기존 작은 모델 3개도 같은 실험 묶음에서 다시 실행하여 세션 차이를 확인합니다. Vision은 구조도 달라지므로 순수 모델 크기 효과라고 해석하지 않습니다. 같은 GPT-2/T5 계열도 층수·차원 등이 달라져 파라미터 수 하나의 인과 효과를 분리한 실험은 아닙니다.

조건별 진행 순서는 기존 pilot.py의 반복 블록별 무작위 순서를 사용합니다. 무작위 seed는 기존 20261103을 유지합니다. CPU에서 선택한 plan이 cache-check, plan, 실제 pilot에 모두 전달되는지 확인한 테스트 4개가 있습니다. 기본 코드와 합친 로컬 CPU 테스트는 12개 통과했습니다. 큰 모델 GPU 실행 자체는 아직 검증 전입니다.

## 추가 파일

- large-plan.json: 기존 AWS FP16 기본 조건에서 선택한 8개 case. 기존 3개 pilot 조건은 그대로 포함합니다.
- container_run_large.py: 기존 이미지 ID와 probe.py 해시를 확인하며 큰 모델 plan을 실제 실행에 전달합니다. 기본 실험 코드를 수정할 필요가 없습니다.
- test_large_launcher.py: CPU에서 custom plan 전달과 48개 스케줄을 검증합니다.
- large-schedule.json: 준비 환경의 CPU dry-run 출력. 실제 실행에서는 현재 파일 해시로 새 manifest가 생성됩니다.

각 GPU 프로세스의 timeout은 최대 600초입니다. timeout 설정은 중단 한도이며 추론 반복 수·입력·샘플링은 바꾸지 않습니다. 에러 발생 시 전체 스케줄은 중단되며 실패한 raw와 로그를 보존합니다. 기존 결과 경로를 재사용하지 않습니다.

## 진행 순서

1. 이 확장 ZIP을 `/home/inugpu/cloudgpufit-outside`에 풀어 기존 outside-lab 디렉터리에 새 파일을 추가합니다.
2. GPU 없는 cache-check로 8개 모델의 고정 revision·config·가중치 파일이 존재하는지 확인합니다. 필요한 revision 파일은 기존 outside-lab/revisions.json을 사용합니다. 누락이 있으면 GPU 없는 cache-download를 수행하고 다시 검사합니다. 캐시 검사는 파일 존재·index shard 확인이며 내용 전체의 무결성 검사나 실제 모델 실행 검증과 구분합니다.
3. plan dry-run에서 48개 작업인지 확인한 후 tmux에서 pilot을 실행합니다. 구체적인 명령은 단계별로 안내합니다.

기존 GPU UUID: GPU-17eb7299-8b39-914d-7c5b-26908502904f

이미지 ID: sha256:457da255e6e2de7ee05ff7814674c90abcf6d1ed04e73df3cc4eebcdb31fb032

probe.py SHA256: 7df9a02ce9ca938b9702ee2c9fd7d0414d0a4cfc0cfe5ab87c97d7a069243d88

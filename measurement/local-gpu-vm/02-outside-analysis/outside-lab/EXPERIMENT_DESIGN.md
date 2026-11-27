# CloudGPUFit outside 실험 설계 — 2026-11-03

목표는 미측정 GPU에 대한 예측 가능성을 판단할 데이터를 확보하는 것입니다. 이번 패키지는 학습기를 만들거나 xMem 예측·보정표를 변경하지 않습니다. inuos와 AWS 자원에는 접속하거나 변경하지 않았습니다. 첫 신규 측정 환경은 사용자가 선택한 로컬 RTX 3090입니다.

## 1. 정의와 분석 단위

`outside(t) = NVML의 해당 PID 메모리(t 부근) - torch.cuda.memory_reserved(t 부근)`입니다. allocated를 빼지 않고, 서로 다른 시각의 최대값을 빼지 않습니다. 음수·NVML sentinel·조회 실패·PID 미일치는 원본 그대로 남깁니다. outside를 CUDA context의 순수 크기나 CPU/xMem 예측 잔차로 취급하지 않습니다.

시간 샘플 → 프로세스 내 단계 요약 → 독립 프로세스 반복 → GPU UUID/호스트·소프트웨어 환경 → GPU 모델/아키텍처 순서로 집계합니다. 342조건은 워크로드 적용 범위이며 독립 하드웨어 표본 342개가 아닙니다. 같은 물리 GPU에서 새 프로세스를 10번 실행해도 GPU 모델 표본은 1개입니다. L4 인스턴스 3종도 GPU 모델은 1종입니다. GPU별 집계는 조건 수가 많다는 이유로 가중하지 않습니다.

## 2. 실제 확인한 자료와 기존 결과

- GitHub `codex/vram-native-inference-20261103` head: `7f822d76af13bea39191beb1852ec67513b18f15`. `vram_policy.py`는 reserved+GPU별 outside를 사용하며 미보정 GPU를 구분합니다.
- 제공한 Notion 두 페이지를 읽고, 기록 페이지의 원본 AWS 압축 파일 6개를 내려받아 분석하였습니다.
- 첨부 `bench_worker_v3.py` 6개는 같은 SHA256 `4018cd8ef09cef14be3e474845b3476495201769d47a7a0763f1f2961e07f651`입니다.
- raw 아카이브 해시·실행 manifest는 `analysis/archive_provenance.json`, `analysis/legacy_audit.json`에 보존하였습니다. 원본 JSON을 변경하지 않았습니다.

| 첨부 기준 인스턴스 | GPU 기록 | 유효 memory 조건 | memory 시간 샘플 | 조건 중앙값 중앙값 / 범위 (MiB) | 샘플 2개 이하 조건 |
|---|---|---:|---:|---|---:|
| g4dn.2xlarge | Tesla T4 | 228 | 24,335 | 130 / 128–132 | 6 |
| g5.2xlarge | NVIDIA A10G | 342 | 29,813 | 308 / 302–312 | 69 |
| g6.xlarge | NVIDIA L4 | 342 | 28,126 | 232 / 226–236 | 69 |
| g6.2xlarge | NVIDIA L4 | 342 | 27,691 | 232 / 226–236 | 69 |
| g6.4xlarge | NVIDIA L4 | 342 | 28,115 | 232 / 226–236 | 69 |
| g6e.2xlarge | NVIDIA L40S | 342 | 26,310 | 510 / 504–514 | 76 |

**인스턴스 라벨 충돌:** T4, A10G, L4 16스레드 실행의 manifest와 조건 environment에 `g6.2xlarge`가 남아 있습니다. 표의 인스턴스명은 사용자가 제공한 첨부명·대응표 기준이며 EC2 메타데이터로 독립 확인된 값은 아닙니다. 결과 CSV에는 원래 declared_instance와 별도 source를 함께 기록합니다. GPU 종류는 실제 environment.gpu로 구분합니다. UUID/instance-id가 없어 기존 자료로 물리 호스트 정체성을 복원할 수는 없습니다.

실측에서 확인한 범위:

1. 지원된 memory 조건은 총 1,938개, 시간 샘플은 164,390개입니다. 각 조건의 기록된 memory 샘플 안에서 outside 범위는 모두 0MiB였습니다. components로 넘어가도 조건별 중앙값 차이는 모두 0MiB였습니다. startup·components를 합친 전체 분석 샘플은 305,813개이며 유효 쌍의 음수와 저장 outside 재계산 불일치는 없었습니다. 이는 이 샘플링 해상도에서의 관측 결과입니다.
2. memory의 설정 간격은 50ms, 조건별 간격 중앙값들의 중앙값은 인스턴스별 약 50.41–50.51ms입니다. 전체 조회 시간의 조건별 중앙값을 다시 집계하면 약 0.32–0.43ms입니다. memory 샘플 2개 이하가 358조건이므로 매우 짧은 추론의 순간 최대값을 평가하기에는 부족합니다. 모든 시점의 조회가 원자적이라는 의미는 아닙니다.
3. memory startup은 매 조건 2샘플뿐이며 `model_reused=true`인 재사용 모델 관측입니다. 실제 첫 CUDA 초기화·첫 모델 로딩 구간이 아닙니다. performance와 first_logits에는 outside를 계산할 NVML 시계열이 없고, tokens도 outside 시계열이 없습니다. components는 동기화 hook을 사용한 별도 계측 구간입니다.
4. 예를 들어 L4의 GPT-2·T5는 FP32 226MiB, FP16/BF16 232MiB가 관측되었습니다. Vision은 대체로 각각 228MiB, 234MiB이고 일부 FP16 조건은 236MiB였습니다. 모델 크기보다 정밀도/연산 경로와 관련된 작은 차이가 의심되지만 원인을 분리한 실험은 아닙니다.
5. 같은 조건을 같은 UUID·같은 환경에서 새 프로세스로 반복한 설계는 없습니다. 각 아카이브의 repeat는 0 하나입니다. 조건마다 새 프로세스를 사용했다는 실행 구조는 확인했지만, 서로 다른 조건의 프로세스를 동일 조건 재현성 반복으로 세지 않습니다. 케이스 순서도 고정되어 시간 변화와 조건 변화가 혼재합니다.

| 질문 | 기존 자료로 가능한 것 | 기존 자료로 불가능한 것 |
|---|---|---|
| 워크로드 영향 | GPU·모델·정밀도·배치·길이별 기술 통계, 대응 조건 차이 | 고정 순서·미반복 조건을 이용한 엄밀한 인과 효과 |
| 시간 변화 | performance 뒤 memory의 startup/실행 및 이후 components 비교 | import, 초기화, 첫 커널의 상승 원인 |
| 재현성 | 기록 구간 안의 일정함, L4 3개 실행의 조건별 대응 | 동일 조건 fresh-process 반복 및 부팅/물리 GPU 간 변동 |
| 환경 민감도 | 동일 이미지·명시된 버전 범위 확인 | 다른 driver/runtime/backend/모듈 로딩의 효과 |
| GPU 일반화 | 4개 GPU 모델 사이 수준 차이 | 하드웨어 변수별 효과 분리, 미측정 아키텍처 정확도 보장 |
| 용량 판단 | outside 오차가 더해지는 정도의 재계산 | 실제 OOM 경계 정확도; 기존 실행 최대치는 모두 여유가 있었음 |

## 3. 신규 측정 전에 고정할 기준

아래는 기존 자료를 본 뒤 제안한 **앞으로의 실험 기준**입니다. 기존 자료에 대한 사전등록으로 소급하지 않습니다. `criteria.json`에 보존하고 첫 GPU 실행 후에는 유리한 기준으로 변경하지 않습니다. 변경이 필요하면 이유와 새 버전을 만들고 이후 검증 세트를 분리합니다.

- 기본 단위: MiB=2^20 byte. 모든 원본은 byte 정수, 시간은 monotonic ns와 UTC wall ns로 기록합니다.
- 안정 관측: checkpoint에서 명시적으로 synchronize한 뒤 50ms 간격 21회(약 1초)를 관측합니다. 유효 샘플 최소 10개가 필요합니다. 마지막 10개 outside 범위 ≤8MiB를 plateau 진단 기준으로 둡니다. plateau 미달을 삭제해서 안정적인 결과만 남기지 않고, 미안정 단계로 따로 보고합니다.
- 조회 품질: 주 조회 창 ≤5ms, bracket 양끝 값 일치, 해당 PID 식별, 조회 오류 없음, 다른 GPU 프로세스 없음. 초과·음수·누락은 raw에 보존하고 품질 플래그로 분리합니다. CUDA 초기화 전 PID 부재는 예상 가능한 결측이며 outside=0으로 만들지 않습니다.
- 동일 조건 재현성 파일럿: fresh process 3회 중앙값의 최대–최소 ≤16MiB면 확장 후보입니다. 3회만으로 안정성 확정/유의확률을 주장하지 않습니다. 확증이 필요하면 10회 이상과 별도 세션 반복으로 확장합니다.
- 실용 효과: 변화량 16MiB 이상을 후속 조사 기준으로 둡니다. 유의성은 시간 샘플 수로 만들지 않습니다. 10개 이상 독립 프로세스 블록에서 사전 대응 차이와 프로세스 단위 bootstrap 90% 구간을 평가합니다. 구간이 [-16,+16] 안이면 실용 동등, 전부 +16 초과 또는 -16 미만이면 실용 변화, 나머지는 판단 보류입니다. 여러 비교는 탐색 결과로 표시하고 별도 확인 반복을 둡니다.
- 향후 예측 기준(이번에는 학습하지 않음): held-GPU 평균절대오차 ≤32MiB, 절대오차 p95 ≤64MiB, 관측 최대 과소추정 ≤128MiB를 우선 목표로 제안합니다. GPU별 값을 따로 표시하며 GPU 4종만의 p95는 매우 불안정하다고 명시합니다. 신규 독립 GPU·환경에서 다시 평가해야 합니다.
- 용량 경계: outside만 평가할 때 `동일 시점의 실제 reserved + 예측 outside`와 해당 PID NVML을 비교하여 xMem 오차를 섞지 않습니다. 실제 reserved는 **평가용 정답 측 정보**이지 제품의 필수 입력이 아닙니다. 이후 xMem 결합 평가는 별도입니다.
- signed error=예측 outside−관측 outside. 과소추정량=max(관측−예측,0)은 평가 지표에만 사용하며 raw outside를 clamp하는 것과 구분합니다.
- 평가용 용량 여유를 -512,-256,-128,-64,0,+64,+128,+256,+512MiB로 구성하여 적합/부적합 결정 전환을 계산합니다. ±256MiB 이내는 보류 영역으로 표시합니다. 보류 밖의 거짓 ‘적합’ 0건을 목표로 두되 유한 데이터에서의 목표이지 미래 보장치가 아닙니다. 관측 표본의 최대 outside를 이론적 상한이라고 부르지 않습니다.

## 4. 가설별 실험표

| 가설 | 바꾸는 변수 | 고정 변수 | 독립 반복 단위 | 관측값 | 판정 기준과 해석 |
|---|---|---|---|---|---|
| H0 계측이 대상을 바꾸는가(추가) | pre-init memory API 호출 유무; 실행 중 50ms 샘플링 유무; TNT/NTN 조회 순서(각각 별도 비교) | 같은 UUID, 이미지, driver, 연산, 반복 횟수 | 설정당 새 Python 프로세스 3회, 필요 시 10회 | API 전후 is_initialized, NVML raw, 단계 outside, action 시간 | 초기화 상태가 바뀌거나 plateau 차이 ≥16MiB면 계측 효과 조사. 처리시간 5% 이상 변화도 계측 부하로 기록. checkpoint 동기화·NVML 자체의 완전 무관측 상태까지 분리한 실험은 아님 |
| H1 새 프로세스 안정성 | fresh-process 반복 번호; 별도 실행 세션 | UUID, 이미지·환경변수, dtype, 입력, 단계 | 조건당 fresh process 3회→10회, 별도 세션 | 단계별 프로세스 중앙값, 범위, 재현성 | 파일럿 최대–최소 ≤16MiB; 확장 후 동등 구간 판단. 같은 호스트 반복을 GPU 모델 독립 표본으로 세지 않음 |
| H2 워크로드보다 환경 차이가 큰가 | 모델 크기, dtype, B, 입력 T, 출력 N을 각각 하나씩 | 같은 GPU와 소프트웨어·준비 이력 | 같은 조건별 fresh process 3회, 무작위 블록 | 조건 대응 차이와 GPU별 범위 | 조건 변화가 ±16MiB 이내인지 확인. GPU/환경 간 차이와 크기를 비교하되 hardware/driver를 동시에 바꾼 비교는 인과 분해하지 않음 |
| H3 초기화 이후 plateau인가 | 첫 연산 종류와 순서, 첫 호출/반복 횟수, performance 선행 | UUID·환경·dtype·입력 형태 | 단독 gemm/conv/attention 및 혼합 순서별 fresh process 3회 | 단계 전후 outside, 실행 중 raw, 이후 2개 checkpoint | ≥16MiB 상승이 다음 두 단계에도 남으면 지속 상승 후보. 후속 동작 자체가 추가 할당하므로 부품별 순수 메모리로 합산하지 않음 |
| H4 소프트웨어 환경 민감도 | LAZY/EAGER, attention 경로, cuDNN benchmark, 검증된 wheel stack, driver(별도 단계) | 각 비교에서 같은 GPU·조건과 나머지 설정 | 설정별 새 프로세스 3회→10회; driver는 A–B–A 세션 | 단계 차이, 실제 로드 라이브러리·환경 manifest | 각 변경의 대응 차이·동등 구간. bundled dependencies가 함께 바뀌면 ‘소프트웨어 묶음 효과’로만 해석 |
| H5 GPU 특성의 설명력 | 물리 GPU 모델·사전 확보 특성 | 같은 이미지·프로토콜, 가능하면 같은 driver | GPU 모델과 환경 cell. 프로세스 반복은 cell 오차 추정용 | GPU별 요약, VRAM·SM·구조 후보와 잔차 | 4종에서 다변량 회귀 확정 금지. 단순 용량/구조별 가정의 반례 탐색. 서로 다른 물리 GPU 반복도 필요하면 추가 |
| H6 미측정 GPU 실용성 | GPU 전체 holdout; architecture 전체 holdout | 평가 시 학습에 넣지 않은 GPU의 정보만 사용 | GPU 또는 아키텍처 단위 외부 검증 | MiB 오차, 과소추정, 용량 결정 전환, 보류율 | 고정 허용 기준과 공통상수·그룹상수·범위 기준선 비교. 새 구조 외삽은 별도 미검증/불확실로 유지 |

## 5. 최소 비용 순서와 확장 조건

### A: 이미 완료한 CPU 재분석

새 GPU를 켜지 않고 6개 첨부를 분석하였습니다. 코드·manifest 감사, model/dtype/condition 집계, memory↔components 대응, 조회 시간과 누락 검사를 완료했습니다. 기존 데이터를 근거로 342조건 전체를 다시 돌릴 이유는 아직 없습니다.

### B0: 로컬 환경 확인과 CPU 준비

현재 RTX 3090 호스트의 OS·driver·GPU UUID·다른 GPU 프로세스, Docker 권한, 기존 이미지 ID, HF 캐시 위치를 확인합니다. 이전 기억을 현재 상태로 가정하지 않습니다. CPU에서 할 작업은 ZIP 전송/검사, dry-run 계획·테스트, 이미지 inspect, **GPU 옵션 없이** import·pip check·모델 cache 검증입니다. 실행 중인 inuos 또는 그 이미지·driver를 바꾸지 않습니다.

기존 이미지 ID `sha256:457da255e6e2de7ee05ff7814674c90abcf6d1ed04e73df3cc4eebcdb31fb032`가 있으면 그것을 우선 사용합니다. 로컬 driver가 AWS와 달라도 문제를 숨기지 않고 새로운 환경 cell로 기록합니다. 이미지가 없으면 기존 tar.gz/이미지 보관 위치를 먼저 확인합니다. 원본 Dockerfile·wheel lock이 없는데 임의 재구성한 이미지를 동일 환경이라고 부르지 않습니다.

### B1: 계측·초기화 파일럿

먼저 init 시나리오 1회로 PID/UUID 일치와 raw 저장을 확인합니다. 그다음 `pilot.py --tier core` 계획은 **10종 절차 × 새 프로세스 3회 = 30개 프로세스**입니다. 기본은 GPU 1개·프로세스 1개만 순차 실행하며, 반복 블록 안에서 순서를 무작위로 섞습니다.

| 절차 | 개수 | 목적 |
|---|---:|---|
| init 대조군 / pre-init memory API 호출군 | 2 | API 관측자 효과 |
| gemm / conv / eager attention 단독 | 3 | 연산별 최초 호출 |
| 세 가지 혼합 순서 | 3 | 앞서 사용한 연산 이력 영향 |
| gemm의 실행 중 sampler 없음 / NTN 조회 | 2 | 빈번한 조회와 순서 효과를 각각 기본 gemm과 비교 |

새 프로세스 반복과 프로세스 내부 반복은 다릅니다. 각 단독/혼합 연산에서는 첫 호출, 3회 warmup, 10회씩 3블록 반복, 텐서 해제 뒤 cache 유지 단계를 봅니다. checkpoint는 각 약 1초이며 전체 시간은 첫 실행 결과로 추정합니다. 파일럿에는 수백 조건이나 큰 pretrained 모델 다운로드가 필요 없습니다.

### B2: 실제 pretrained 모델 연결

B1에서 측정 품질이 확보되면 고정 revision의 GPT-2·ResNet-50·T5-small을 사용합니다. 기본 FP16 조건 3개 × direct/performance5 선행 2절차 × 새 프로세스 3회 = **18프로세스**입니다. 실제 generate/forward를 실행하며 LLM·T5는 생성 길이를 고정합니다.

이 단계는 원본 bench를 그대로 복제하지 않습니다. 먼저 finite forward와 첫 generate를 구분해 관찰하고, 추가 warmup 후 performance5의 추가 영향만 비교하는 **단계 분해 프로토콜**입니다. original hybrid와 완전히 같은 상태인지는 기존 bench의 memory-only와 performance→memory 경로를 따로 실행해 교차 확인해야 합니다. 원본과 같은 5 performance·1 warmup·3 memory 수만 맞추었다고 전체 이력이 같다고 주장하지 않습니다.

H2를 확장할 필요가 있으면 GPT-2에서 B 1→4, 입력 256→512, 출력 64→128을 각각 바꾼 3조건과 GPT-2 XL 기본 조건을 추가합니다. ResNet은 B만, T5는 encoder 길이만 추가할 수 있습니다. 먼저 한 변수씩 바꾼 조건을 고정한 새 plan으로 저장합니다. dtype FP32/BF16도 한 번에 한 가지씩 비교하고 실제 지원 여부를 확인합니다.

### C: 환경 민감도 — 필요할 때만

| 비교 | 첫 대상 | 고정/달라지는 범위 | 준비 및 제한 |
|---|---|---|---|
| 기본/LAZY/EAGER | init, 혼합 연산, GPT-2 | 이미지·driver 동일, 시작 전 env 한 항목 변경 | 실제 after-import/after-init env와 cuModuleGetLoadingMode를 기록. unset이 곧 고정된 의미는 아님 |
| eager/SDPA math/flash/efficient | 작은 attention 연산 | 같은 dtype/shape | backend를 명시하여 fallback 금지. 미지원은 실패로 남김. HF 모델은 eager/sdpa 별도 비교 |
| cuDNN benchmark false/true | conv·ResNet | 나머지 동일 | 원본 false. true는 별도 버전의 plan/worker에서 명시적으로 적용한 뒤 수행; 현재 runner에는 자동 sweep 없음 |
| torch 2.6.0 cu124 ↔ 2.6.0 cu126 | 작은 연산 후 대표 모델 | nominal torch는 같지만 CUDA·cuDNN·cuBLAS 등 종속 wheel이 달라질 수 있음 | 공식 제공 wheel 조합인지 확인 후 CPU에서 별도 이미지 build/import/pip check. 순수 CUDA runtime 효과라고 부르지 않음 |
| torch 2.5.1 cu124 ↔ 2.6.0 cu124 | 동일 | runtime 표기가 같아도 의존 library 차이 가능 | Transformers·accelerate 호환성 확인. 설치 결과 전체 diff 보존. torch 단독 인과 추정으로 단정하지 않음 |
| driver A ↔ B ↔ A | 격리한 유휴 로컬 또는 별도 AWS 호스트 | 가능한 한 같은 GPU·이미지·host OS | 현재 작업 VM 변경 금지. driver 업데이트/reboot는 지금 실행하지 않음. 별도 작업 창과 복구 계획 후 진행 |

NVIDIA 문서상 모듈 lazy loading은 실제 필요 시점까지 커널/데이터 로딩을 미룰 수 있으므로 H3/H4의 근거가 됩니다. PyTorch 2.6은 cu118/cu124/cu126 공식 wheel을 제공합니다. 단, 특정 호스트에서의 driver 지원과 모델 실행까지 자동 보장하는 것은 아니므로 이미지 후보는 CPU 설치 검증 후 GPU smoke를 통과해야 합니다. 환경 파일을 숫자 하나로 인코딩해 새 버전을 자동 일반화하지 않습니다.

### D: 유료 GPU 또는 새로운 GPU

로컬 측정 후 동일 이미지의 AWS GPU 1종에서 차이를 확인할 이유가 있을 때만 진행합니다. 기존 L4를 다시 빌리면 로컬↔AWS 비교와 절차 재현에 도움이 되지만 독립 GPU 모델 종류는 늘지 않습니다. 일반화 검토가 목적이면 기존과 겹치지 않는 GPU 모델·구조가 더 유용할 수 있습니다. 특정 SKU 구매는 가용 리전·요금·목적을 확인한 뒤 결정합니다.

유료 GPU 생성 전 완료 조건: 스크립트 해시 고정, 실행 계획·기준 고정, CPU 테스트, 이미지 빌드/보관, 필요한 세 모델 캐시·revision 검증, 결과 저장 위치, 압축/다운로드 명령, CPU 분석 dry-run, 예상 실행량과 비용 상한 결정. 큰 이미지·캐시는 GPU 없는 머신이나 저장소에서 먼저 준비합니다. 생성 뒤 전송이 필요하면 사전 체크섬·전송량·경로를 준비하고 전송 시간을 비용에 포함합니다. GPU idle 상태로 빌드·다운로드를 기다리지 않습니다. 현재 AWS 자원이 켜져 있다고 가정하지 않습니다.

## 6. GPU 특성과 제품 입력의 경계

| 특성 | EC2 DescribeInstanceTypes | 외부 정적 자료 | 대상 GPU 접근 | CPU 사전 예측 사용 |
|---|---|---|---|---|
| GPU명·제조사·개수·GPU별/전체 명목 메모리 | 가능 | 검증 보조 | 불필요 | 가능 |
| fractional GPU 비율·논리 개수 | 응답/SDK에서 제공되는 경우 가능 | 서비스 문서 보조 | 실제 상태 검증용 | 제공 시 가능; 누락값 추정 금지 |
| 아키텍처·compute capability | 일반 GPU Info 필드에는 없음 | NVIDIA 문서/정적 매핑 | CUDA/NVML로 관측 확인 | 출처 있는 정적 매핑으로 가능 |
| SM 수·메모리 대역폭·버스 등 | 일반 GPU Info 필드에는 없음 | 모델/SKU별 공식 명세에 있으면 가능 | 실제 활성 SM 등 확인 | 정적 확인 값만; CUDA core 수를 임의 SM 수로 환산하지 않음 |
| UUID·현재 free VRAM·현재 프로세스·실제 MIG/MPS | 아님 | 해당 없음 | 필요 | 필수 사전 입력 금지. 데이터 QA와 선택적 실측용 |
| 실제 driver·설치 wheel·환경변수 | 아님 | 사용자 제공 목표환경/이미지 manifest | 설치 상태 검증 시 필요 | 아는 경우 범위 지정; 모르면 unknown 환경으로 보류/범위 제공 |

SM 수를 조회할 수 있다는 것과 제품에서 미측정 GPU의 SM 수를 사전에 알 수 있다는 것은 다릅니다. `torch_arch_list`는 현재 torch binary의 지원 아키텍처이며 GPU 하드웨어 특성과 구분합니다. UUID·free memory를 학습용 설명변수로 넣지 않습니다. GPU 특성 표에는 값·단위·SKU 범위·출처 URL·조회일·예측 시점 사용 가능 여부·누락 이유를 함께 저장합니다.

## 7. 출력 데이터와 분석

| 파일 | 내용 |
|---|---|
| suite-manifest.json | 계획된 모든 작업, 반복·랜덤 순서·코드 해시, 실제 명령·종료 코드 |
| 각 run/manifest.json | PID·독립 반복 번호, UUID·가용/총 메모리·driver·구조 정보, 이미지 ID, 실제 package 버전·torch config·binary 지원 arch, 입력 해시·model revision·backend·생성 설정, 환경변수 전/후, 실패 traceback |
| 각 run/samples.jsonl | stage·kind·wall/monotonic 시각, 각 torch/NVML API 구간, PID별 raw usedGpuMemory·오류·다른 PID, 같은 짧은 조회 창의 allocated/reserved/NVML/outside, bracket 변화, 별도 시각의 device free/used/total |
| 각 run/events.json | import/init/연산/동기화 단계 경계. synchronized 안정 관측과 실행 중 관측을 구분 |
| process_stage.csv | 프로세스·단계별 샘플 수·품질·중앙값·범위·시작/끝 변화 |
| repeat_groups.csv | 같은 suite·환경·절차·단계의 독립 프로세스 중앙값 분포 |
| quality_flags.csv / audit.json | 누락·음수·조회 지연·다른 PID·bracket 변화. 원본을 삭제하지 않음 |

TNT는 torch→NVML→torch, NTN은 NVML→torch→NVML입니다. 주 outside는 첫 NVML값과 첫 torch reserved의 차이이며 각 API의 타임스탬프와 양끝 변화를 함께 남깁니다. 완전 원자성이나 NVML 내부 갱신 주기를 보장하지 않습니다. sampled peak는 아주 짧은 GPU transient를 놓칠 수 있습니다.

현재 PyTorch 2.6.0 소스에서 memory_allocated/reserved는 memory_stats를 경유하고 초기화 전 stats는 빈 dict를 반환합니다. 그래서 API가 항상 CUDA를 초기화한다고 가정할 필요는 없습니다. 그래도 기본 계측은 pre-init memory API를 호출하지 않으며 API probe 별도 셀에서 실제 전후 상태를 확인합니다. synchronize, device properties, cuDNN 버전 조회 등 상태를 바꿀 가능성이 있는 진단은 단계로 표시하거나 마지막으로 미룹니다. NVML import/init 이전의 완전 무관측 상태는 이 in-process 설계로 측정하지 않으며, 초기 단계에서 원인 불명 변화가 보이면 별도 외부 관측자 프로토콜을 먼저 추가합니다.

다른 compute/graphics GPU PID가 시작 시 발견되면 현재 파일럿은 중단하고 manifest를 남깁니다. RTX 3090이 디스플레이 GPU라면 GUI 프로세스를 먼저 확인하고, 임의 종료하지 말고 측정 가능 시간/구성을 결정합니다. 실행 중 다른 PID가 생기면 샘플 품질 플래그에 남깁니다. device-wide memory를 우리 PID 메모리로 대체하지 않습니다. 컨테이너는 `--pid=host`로 PID namespace를 맞추고 단일 GPU UUID만 노출합니다. MIG/MPS/다중 GPU/다중 추론 프로세스는 별도 프로토콜 대상입니다.

## 8. 학습 진행 여부 결정

지금은 독립 GPU 모델 4종이며 software stack도 사실상 한 조합입니다. 복잡한 학습으로 갈 단계가 아닙니다. H0/H1/H3와 대표 모델 연결부터 검증합니다.

| 결과 | 다음 결정 |
|---|---|
| 같은 GPU·지원 환경의 outside가 작게 변동 | 환경 범위를 명시한 상수/관측 범위를 우선 사용 |
| software/backend/이력에 의미 있는 차이 | 알려진 환경 그룹별 보정 또는 구간; 미지 환경은 넓은 범위/보류 |
| 사전 확보 GPU 특성이 일관된 설명력 | 단순 예측식 후보를 검토하되 GPU 전체 holdout으로 평가 |
| GPU명 조회는 잘 맞지만 미측정 GPU는 실패 | 보정표 조회 기능과 미측정 예측 기능의 결과를 별도 표기 |
| 특성으로 설명되지 않는 GPU 차이·새 구조 | 점 추정 신뢰 부족을 표시하고 독립 GPU/환경 데이터를 추가 |

기준선은 (1) 학습 GPU를 동일 가중한 공통 중앙값, (2) 사전 정의·출처 있는 GPU군별 상수와 unknown fallback, (3) 학습 환경의 범위+명시적 여유입니다. 같은 GPU의 행을 학습/검증에 나눠 이름을 외우게 하지 않습니다. GPU별 보정표는 학습 대상 GPU를 실제 측정한 기능이며 heldout GPU 기준선에 정답 보정값을 제공할 수 없습니다.

외부 평가에서 GPU 하나를 통째로 제외하고, 선택/튜닝은 제외한 GPU를 보지 않고 수행합니다. 아키텍처 검증은 그 구조의 GPU 전부를 제외합니다. L4 3개 인스턴스나 수천 시계열 샘플이 fold 표본 수를 늘려주지 않습니다. 같은 아키텍처 내 새 SKU와 새 아키텍처 외삽은 구분합니다. 현재 4종에서는 fold별 결과와 실패 사례를 보여주고 일반 성능 보장을 하지 않습니다. 가설 하나가 틀리면 모델·그룹·범위·추가 데이터 중 무엇을 바꿀지 검토합니다.

단순 기준선보다 heldout MAE가 최소 20% 좋아지고 과소추정/용량판정이 악화되지 않는 것을 학습 후보 선택 기준으로 제안합니다. 해당 기준을 충족해도 별도 독립 GPU·환경 외부 검증 전에는 제품 일반화 성능으로 주장하지 않습니다.

## 9. 코드 상태와 다음 실행

CPU 회계/품질/반복단위 테스트와 dry-run은 실행했습니다. **GPU 실측은 아직 실행하지 않았고 GPU API 호환성도 실제 호스트에서 smoke로 확인해야 합니다.** CUDA 이미지·로컬 HF cache가 현재 존재하는지도 확인 전입니다. 환경 sensitivity 자동화와 실제 OOM 테스트는 후속 단계이며 이번 파일럿에서 수행했다고 주장하지 않습니다.

지금 사용자에게 안내할 명령은 로컬 RTX 3090의 `nvidia-smi`와 기존 Docker 이미지 inspect 두 개입니다. 결과를 받은 다음 CPU 이미지/캐시 준비를 2–3단계씩 진행하고, 모든 준비가 끝난 뒤 GPU smoke를 실행합니다.

## 근거

- 저장소 브랜치: https://github.com/kyj2001/CloudGPUFit/tree/codex/vram-native-inference-20261103
- AWS 분석: https://app.notion.com/p/40dae5cc98b980e3b748cc42b7cf3390
- 실험 기록·원본 6개 첨부: https://app.notion.com/p/40cae5cc98b980baa7aecd1ed1da966c
- 실제 PyTorch 2.6 메모리 API 소스: https://github.com/pytorch/pytorch/blob/v2.6.0/torch/cuda/memory.py
- PyTorch 2.6 CUDA semantics: https://docs.pytorch.org/docs/2.6/notes/cuda.html
- CUDA 12.4.1 모듈 로딩: https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html#lazy-loading
- PyTorch 공식 이전 버전 설치 조합: https://pytorch.org/get-started/previous-versions/
- NVML process/device queries: https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html
- EC2 GPU 특성 필드: https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_GpuDeviceInfo.html

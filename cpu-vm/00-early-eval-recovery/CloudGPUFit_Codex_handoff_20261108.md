# CloudGPUFit CPU VRAM 예측 복구 — VS Code Codex 인수인계 프롬프트

지금부터 CloudGPUFit의 **CPU 기반 xMem VRAM 예측 경로**에서 발생한 RAM 폭증, OOM, 긴 실행시간 문제를 이어서 해결해 주세요. 저는 결과를 매번 압축해서 ChatGPT에 보내는 대신, VS Code의 Codex가 실제 코드와 VM 로그를 직접 읽고 수정·검증하기를 원합니다.

이 문서를 읽고 설명만 하는 것으로 끝내지 말고, 접근 가능한 환경에서 아래 작업을 실제로 진행해 주세요. 다만 과거 기록을 현재 상태로 간주하거나, 검증되지 않은 수정본을 이미 해결된 것으로 취급하지 마세요. 한국어 존댓말로 핵심부터 보고하고, 중간 보고는 확인한 사실·남은 문제·다음 2~3개 작업 중심으로 간결하게 해 주세요.

## 1. 이번 작업의 목적과 범위

CloudGPUFit은 모델·정밀도·배치·입력/출력 길이 등의 추론 조건을 입력받아 CPU에서 GPU VRAM 요구량을 예측하고, 적합한 GPU 인스턴스의 비용·성능을 비교하는 프로젝트입니다.

이번 작업은 **응답시간 예측기나 GPU outside 실험이 아니라 CPU xMem 예측 실행의 복구**입니다. 다른 대화에서 진행하던 inugpu의 RTX 3090/Qwen/outside 실험과 혼동하지 마세요.

현재 CPU 예측은 실제 CPU forward/generate에서 텐서 storage 할당·해제 수명을 기록한 뒤 xMem allocator에 재생합니다. CPU RAM 사용량을 VRAM으로 환산하거나 CPU 실행시간을 GPU 응답시간으로 사용하는 방식이 아닙니다.

당장의 목표:

1. 이미 확보한 성공 결과·실패 로그·원본 코드를 보존합니다.
2. RAM과 시간이 어느 단계에서 증가하는지 확인합니다.
3. 추론 및 allocator 의미를 유지하는 기록 최적화를 검토·적용합니다.
4. 작업별 자원 제한·종료 처리·오류 진단을 실제 VM에서 검증합니다.
5. 소수 대표 조건에서 정확성과 비용을 확인한 뒤 실패·미완료 조건을 복구합니다.

처음부터 342개 전부를 다시 돌리지 마세요. 새로운 AWS GPU 인스턴스를 만들 필요도 없습니다. 장기적으로 사용자 PC에서 실행 가능한 예측 속도가 필요하지만, 이번 복구를 계기로 전체 추론 구조를 무리하게 다시 작성하지는 않습니다.

## 2. 작업 환경을 먼저 구분하세요

과거 실행 환경:

- 호스트: `INUOSVM-1`, 사용자: `inuos`
- 준비된 CPU 컨테이너: `xmem-cpu`
- 기존 결과: `/home/inuos/cloudgpufit-predictions/native-v3-t4-j6`
- 기존 콘솔 로그: 위 디렉터리의 `console.log`
- RAM: 약 47GiB, 가용 논리 CPU: 24개라고 기록되어 있습니다. 현재 값은 재확인하세요.
- 컨테이너 환경: Python 3.11.11, torch 2.6.0+cu124, CUDA runtime 12.4, Transformers 4.52.4, GPU 비활성
- 기존 CPU 이미지 ID: `sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee`
- 실제 확인된 bind mount: `/home/inuos/xmem-work` → `/work`, 읽기/쓰기
- HF cache mount: `/home/inuos/.cache/huggingface` → `/root/.cache/huggingface`

VS Code가 Remote SSH로 inuos에 연결되어 있는지, 로컬 PC에서 열려 있는지 먼저 확인하세요. `pwd`, `hostname`, `whoami`, Git·Docker 접근 상태를 확인하면 됩니다. 로컬 PC의 다른 Docker를 inuos 컨테이너로 착각하지 마세요. SSH 접근이 없다면 이 제한을 명시하고 실제로 필요한 연결 정보만 요청하세요.

이 문서에 등장하는 경로는 과거 확인된 경로 또는 앞으로 만들 예정인 경로입니다. 존재 여부와 현재 상태를 직접 확인하세요. ChatGPT 검증 환경의 `/workspace/...`는 VM 경로가 아니므로 사용하면 안 됩니다.

## 3. 저장소와 기준 버전

- 저장소: https://github.com/kyj2001/CloudGPUFit
- 기존 작업 브랜치: `codex/vram-native-inference-20261103`
- 검토 및 진단 manifest 기준 커밋: `7f822d76af13bea39191beb1852ec67513b18f15`
- xMem upstream: https://github.com/Stone-ResearchLife/xMem
- xMem 기준 커밋: `1ee4bd08dda277a6248abee5b6d40e0d6067efaa`
- allocator source lock: `benchmarks/aws342/xmem-allocator-lock.json`

현재 HEAD·dirty 상태·worktree를 직접 확인하고 실제로 읽거나 실행한 버전을 기록하세요. 원본 checkout에 변경이 있으면 덮어쓰지 마세요. `.git`이 디렉터리가 아니라 파일인 정상 worktree도 있으므로 `git rev-parse`로 판별하세요.

관련 Notion 문서 — 접근 가능할 때만 참고:

- AWS 분석: https://app.notion.com/p/40dae5cc98b980e3b748cc42b7cf3390
- 실험 기록과 xMem 수정 정리: https://app.notion.com/p/40cae5cc98b980baa7aecd1ed1da966c

Notion 접근이 없어도 VM의 코드·원본 결과·동봉 문서로 우선 진행할 수 있습니다. 접근하지 못한 자료를 읽었다고 하지 마세요.

## 4. 중단된 실행과 진단 결과

기존 명령은 다음 설정이었습니다. **이 명령을 그대로 다시 실행하라는 뜻이 아닙니다.**

```bash
python3 -u tools/run_prediction_suite.py \
  --output-root ~/cloudgpufit-predictions/native-v3-t4-j6 \
  --case-set all --threads 4 --jobs 6 --ram-budget-gb 36 --prepare
```

사용자는 오래 걸리고 오류가 많아 이 실행을 중단했습니다. 마지막 확인 당시 컨테이너에는 `sleep infinity`만 남았습니다. 이 확인은 11월 5일 기록이며 지금도 멈춰 있다고 단정하지 말고 다시 확인하세요.

진단 자료는 2026-11-05 13:51:06 KST에 수집되었습니다. ChatGPT가 11월 8일 `diagnostics.tar(1).gz`를 분석한 결과:

| 항목 | 값 |
| --- | --- |
| manifest completed_cases | 260 |
| manifest success_count | 244 |
| manifest elapsed_seconds | 154837.26초, 약 43.01시간 |
| 파일 기준 성공 | 244개 |
| 파일 기준 오류 | 17개 |
| 예측 파일 없는 조건 | 81개 |
| 커널 Python OOM 종료 | 16건 |

17개 오류 중 16개는 커널 OOM 종료와 오류 JSON 저장 시각이 각각 0~2초 차이로 일대일 대응합니다. 나머지는 마지막에 직접 중단한 `core__decoder_llm__large__fp16__06`입니다. 원래 실행기가 조건별 host PID를 남기지 않아 **시각 기반 대응이며 PID까지 직접 연결한 증거는 아니라는 한계**가 있습니다.

manifest는 마지막 수동 중단 전 상태에서 멈춰 260개로 남고, 파일은 261개입니다. 이를 결과 손상이나 성공 260개로 해석하면 안 됩니다. 기존 manifest의 `status: running`도 실제 현재 실행을 뜻하지 않을 수 있습니다.

주요 근거:

- Docker `Memory=0`, `MemorySwap=0`으로 실제 컨테이너 RAM 제한이 없었습니다.
- GPT-2 XL FP16 `03`(B4, 입력256, 생성64): 약 11.44시간 후 단일 프로세스 RSS 약 46.25GiB에서 OOM.
- GPT-2 XL FP16 `04`(B8, 입력256, 생성64): 약 12.87시간 후 RSS 약 46.25GiB에서 OOM.
- 이 XL 구간은 콘솔 기록상 예측 한 개씩 실행됐습니다. 병렬 수만 줄이는 것으로 해결되지 않습니다.
- 성공한 GPT-2 XL FP16 `02`도 20957.7초, 약 5시간 49분이며 텐서 수명 371334개를 기록했습니다.
- 성공한 GPT-2 medium B8·입력256·생성64는 FP16 12882.4초, FP32 130.8초, BF16 196.2초였습니다. 동시 부하가 같지 않은 전체 실행시간이므로 이를 순수 CPU kernel 속도비로 단정하지 마세요. 그래도 CPU FP16 연산 비용을 별도로 확인할 강한 근거입니다.
- `gpuGetDeviceCount failed with code 35`는 성공한 244개 로그에도 모두 나옵니다. 이 경고만 보고 실패 원인을 CUDA 오류로 단정하면 안 됩니다.
- 진단에 복사된 `predict_vram.py`, `cpu_tensor_trace.py` hash는 기존 manifest와 일치했고, 복사된 lock 대상 allocator 파일도 lock과 일치했습니다.

오류가 난 형태는 조건 파일에서 확인했습니다. GPT-2 계열 공통 번호는 다음과 같습니다.

| 번호 | 배치 | 입력 길이 | 생성 길이 |
| --- | ---: | ---: | ---: |
| 10 | 1 | 256 | 256 |
| 11 | 4 | 512 | 128 |
| 12 | 4 | 768 | 128 |
| 14 | 1 | 768 | 256 |

OOM과 대응한 조건들:

- small: FP16 `12,14`; FP32 `11,12,14`
- medium: FP16 `11,12,14`; FP32 `10,11,12,14`; BF16 `11,12`
- large: FP16 `03,04`

정확한 전체 case ID·현재 상태·입력은 반드시 원본 `.input.json`과 조건 계획을 다시 읽어 대응시키세요.

## 5. 예측 경로와 코드에서 확인된 원인

실제 흐름:

조건 선택 → 예약량 기반 스케줄링 → docker exec로 조건별 Python 시작 → 모델/입력 준비 → warmup 1회 → CPU 추론 3회 profiler 기록 → Chrome trace export → JSON 파싱 → 할당/해제 수명 변환 → xMem allocator 재생 → JSON 저장 → 프로세스 종료.

기존 구현의 주요 파일:

- `tools/run_prediction_suite.py`: 조건 선택, 병렬 실행, 환경·source hash 검사, 저장·재개
- `predict_vram.py`: from_config 모델 생성, native dtype 입력/실행, 결과 출력
- `cpu_tensor_trace.py`: `selected_lifetimes`, `trace_and_estimate`
- `orchestrator/setup_manager.py`: 준비된 컨테이너와 host bind mount, 예측 코드 동기화
- xMem의 `perf_estimator/allocator/allocator.py`: `Trace`, `CachingAllocator`
- xMem의 `perf_estimator/allocator/simulator.py`: `AllocatorSim`
- `benchmarks/aws342/selected-plan.json`, `revisions.json`, `xmem-allocator-lock.json`

확인된 문제:

1. `ram_reservation_gb`는 small=6, medium=10, large는 LLM20/Vision16/T5 28이라는 스케줄링 추정 숫자입니다. 정밀도·배치·길이를 반영하지 않고 실제 RAM 사용을 제한하지 않습니다. 코드상 bytes 변환이 없어 엄밀한 GB/GiB 단위도 정의하지 않았습니다. jobs=6도 실행 설정이지 코드에 고정된 상한이 아닙니다.
2. 한 조건 안에서 모델·입력, native profiler 결과, 전체 trace JSON, 수명 쌍·MemoryBlock, 시간별 시뮬레이션 맵과 상세 이력이 겹쳐 살아 있습니다. 전체 trace 문자열은 파싱 중 일시 중첩되며 시뮬레이션 끝까지 별도 문자열로 보관되는 것은 아닙니다.
3. `selected_lifetimes`는 전체 이벤트를 timestamp로 정렬한 뒤 필요한 메모리 이벤트만 처리합니다.
4. xMem `Trace._record`는 이벤트마다 dict·message·`list(self._segments)`를 저장합니다. 깊은 복사는 아니지만 이벤트 수 E와 segment 수 S에 따라 대략 O(E×S) 규모의 참조 목록 비용이 생깁니다. 이것은 시각화/디버깅용으로 peak 계산 자체에 필수는 아닙니다.
5. 정상 종료하면 조건별 Python 메모리는 회수됩니다. 부모는 작은 결과 요약만 모읍니다. 명시적인 출력/KV 텐서 누적은 발견하지 못했습니다. 주된 근거는 조건 내부 peak·동시 작업 합계이며, 실제 자식 잔존이나 다른 누수는 따로 확인해야 합니다.
6. 기존 실행기는 결과 JSON을 먼저 파싱하고 종료 코드를 나중에 확인해 `Prediction JSON missing`이 실제 종료 원인을 가립니다. 종료 137 하나만으로 OOM을 확정하면 안 됩니다.
7. upstream `AllocatorSim.simulate`는 메모리 부족 외 예외까지 `oom=True`로 바꿉니다. 모든 simulation 오류를 용량 부족으로 표시하면 안 됩니다.
8. `gpu_memory_gb=80`은 시뮬레이션 GPU 용량 80GiB이며 CPU RAM 보호와 무관합니다.

기존 3회는 평균 산출용이 아닙니다. 세 번의 수명을 하나의 allocator에 연속 재생하고 전체 최대 allocated/reserved를 구합니다. CPU warmup은 실행하지만 그 할당 이력을 시뮬레이터에 재생하지 않습니다. KV 진단 hook의 최대값에는 warmup도 포함됩니다. 검증 없이 3회를 1회로 줄이거나 dtype을 바꾸지 마세요.

## 6. 실험 대상과 GPU 비교의 한계

342개 조건은 9개 모델과 FP32/FP16/BF16 및 선택된 형태 조합입니다. 번호나 수를 보고 배치·길이를 추정하지 말고 plan을 읽으세요.

| 유형 | small | medium | large |
| --- | --- | --- | --- |
| Decoder LLM | openai-community/gpt2 | openai-community/gpt2-medium | openai-community/gpt2-xl |
| Text encoder–decoder | google-t5/t5-small | google-t5/t5-base | google-t5/t5-3b |
| Vision | google/mobilenet_v2_1.0_224 | microsoft/resnet-50 | google/vit-large-patch16-224 |

CPU는 임의 초기화 `from_config` 가중치와 합성 입력, GPU 실측은 pretrained 가중치를 사용합니다. CPU/GPU의 입력 형태는 대응하지만 입력 난수 생성 방식도 달라 값까지 같지는 않습니다. 이 불일치를 이번 메모리 최적화와 섞어 수정하지 말고 별도 정확도 검증 과제로 기록하세요.

GPU 실측 코드인 `bench_worker_v3.py`, `run_bench_v3.py`는 CPU 예측 코드가 아닙니다. GPU의 hybrid 실행은 performance→memory 등의 allocator 캐시를 공유합니다. 원본 g6.2xlarge GPT-2 FP16 기본 조건의 reserved는 performance 준비 후 474MiB, performance 최대 492MiB, memory 시작·최대도 492MiB였습니다. CPU의 빈 allocator 재생과 초기 이력이 다릅니다.

따라서 allocated↔allocated, reserved↔reserved, reserved+outside↔NVML을 구분하고, reserved 차이를 전부 CPU 텐서 예측 오류나 전부 캐시 탓으로 단정하지 마세요. 과거의 고정 2GB margin, KV 수식 별도 추가, 정밀도 일괄 환산, 경험적 tensor correction은 현재 복구 목표에 포함하지 않습니다.

## 7. ChatGPT가 이미 준비한 수정본 — 원격에는 없음

제공된 파일 이름: `CloudGPUFit_CPU_recovery_20261108.zip`

ZIP 내용:

- `README.md`: 적용·검증 안내
- `diagnosis.md`: 상세 진단과 OOM 대응표
- `fix.patch`: 기준 7f822d7에 적용할 변경
- `prepare_recovery.py`: 기존 저장소와 분리된 worktree를 만들고 patch 적용
- `checksums.json`: 패키지 파일 checksum

GitHub 업로드는 **이전 ChatGPT 환경의 자동 승인 검토에서 차단되어 수행되지 않았습니다.** 수정본은 GitHub 원격 브랜치나 PR로 존재하지 않습니다. Git pull만 하면 수정본이 올 것이라고 가정하지 마세요. 이것이 VS Code 환경의 Git 권한까지 없다는 뜻은 아닙니다. 이번 작업은 우선 로컬 적용·검증으로 진행하고 원격 반영은 별도 범위로 다루세요.

ZIP이나 해제 폴더가 VS Code에서 접근되는지 직접 확인하세요. ChatGPT 파일 링크에 VS Code Codex가 자동 접근할 수 있다고 가정하지 마세요. 없다면 사용자에게 이 ZIP의 실제 위치 또는 workspace 첨부만 요청하세요.

준비된 변경은 다음과 같습니다. **정답으로 믿지 말고 실제 diff를 읽고 독립적으로 검토하세요.**

1. 새 `inference_trace_runtime.py`
   - stdlib JSONDecoder로 traceEvents를 순차 읽고 메모리 이벤트/REGION만 보관합니다.
   - 파일 순서가 시간순이라는 가정을 하지 않고 기존 stable sort/매칭을 유지합니다.
   - 실제 upstream Trace의 byte counter는 그대로 쓰고 `_record`만 생략하는 SummaryTrace를 사용합니다. 전역 monkey patch나 upstream allocator 파일 수정은 없습니다.
   - phase 로그, 현재/최대 RSS·PID·가능한 cgroup memory counter, simulation 오류 구분을 추가합니다.
2. `cpu_tensor_trace.py`
   - profiler export를 별도 helper scope로 분리해 파싱 전에 profiler 참조를 놓습니다.
   - 메모리 이벤트 필터링 후 기존 수명 매칭을 사용합니다.
   - warmup1/측정3/native dtype/full generate와 상주 storage 순서는 유지합니다.
   - `--allocator_history` 디버깅 경로와 `--trace_output` 고정 trace 비교 경로를 지원합니다.
3. `predict_vram.py`
   - 모델 초기화 phase, 오류 stage·traceback, 위 진단 옵션을 추가합니다.
4. 새 `tools/run_prediction_recovery.py`
   - 선택한 조건만 작업 1개씩 실행합니다.
   - 준비된 `xmem-cpu`의 writable layer를 로컬 Docker image로 snapshot합니다. 기존 base image만 새로 실행하면 나중에 pip로 설치된 패키지가 누락될 수 있기 때문입니다.
   - 원본 `/work`, HF cache, 새 predictor source overlay를 읽기 전용으로 연결합니다. 기존 `/work/xMem` 소스를 동기화하지 않습니다.
   - predictor를 작업 컨테이너 PID1로 실행하고 작업별 Docker RAM hard limit·swap 비활성·timeout을 설정합니다.
   - 종료 코드, Docker State.OOMKilled, 마지막 phase, timeout/cancel을 별도 기록하고 작업 컨테이너를 정리합니다.
   - 새 출력 디렉터리를 요구합니다. `--compare-with`로 기존 성공값을 정확 비교하고 차이/입력 불일치가 있으면 다음 작업을 중단합니다.
   - `--remaining-from`은 원본을 읽기만 하고 성공을 제외한 조건을 선택합니다.
5. 버전/cache key 갱신, 새 helper 동기화 대상 추가, 회귀 테스트와 진단 문서를 추가했습니다.
6. 구형 shared-container 실행기는 명시적 legacy opt-in 없이는 실제 실행을 막고 recovery 실행기로 안내합니다. preflight-only는 남아 있습니다.

현재 SummaryTrace도 O(E) counter 목록과 수명 맵을 보관합니다. 완전한 상수 메모리 구현이 아닙니다. `del`/scope 종료가 RSS 즉시 반환을 보장하지 않으며, 실제 CPU 수치 연산도 그대로 남아 있습니다.

## 8. 완료된 검증과 미완료 검증을 구분하세요

ChatGPT의 별도 검증 환경에서 완료:

- Python 3.12 / torch 2.6.0+cpu / Transformers 4.52.4
- 회귀 검사 39개 통과
- 작은 GPT-2 실제 trace를 동일하게 원본/최적화 allocator에 재생해 allocated와 reserved peak가 byte 단위 일치
- 합성 겹침 이벤트에서 전체 allocated/reserved counter sequence 일치
- 작은 GPT-2/T5/ViT/ResNet/MobileNetV2 × FP32/FP16/BF16 = 통합 검사 15개 통과
- 파싱 경계·잘못된 JSON·주소 재사용·미해제 수명·오류 분류·timeout 정리 로직 검사
- 패치를 별도 로컬 worktree에 적용하고 변경 11개 파일의 결과를 대조

한계:

- VM의 Python 3.11.11 / cu124 환경에서 실행한 것이 아닙니다.
- 검증 시 실제 pinned allocator/profiler 모듈을 사용했지만 무관한 package-level training estimator import는 우회했습니다.
- Docker daemon이 없어 실제 Docker mount·snapshot·메모리 제한·취소는 검증하지 못했습니다. CLI lifecycle 테스트는 mock을 사용했습니다.
- 원래 크기의 실패 모델·조건은 아직 재실행하지 않았습니다.
- RAM 감소량·속도 개선 배수는 측정하지 않았습니다.

따라서 "이미 해결 완료, 98개 실행만 하면 됨"으로 시작하면 안 됩니다. 패치 적용과 실제 VM 검증부터 해야 합니다.

## 9. 정확한 현재 중단 지점

사용자가 `prepare_recovery.py --repo . ...`를 실행했지만 다음 오류가 났습니다.

```text
Git failed: fatal: not a git repository (or any of the parent directories): .git
```

당시 현재 디렉터리가 저장소가 아닌 상태에서 `--repo .`을 사용한 것으로 판단했습니다. 스크립트의 첫 Git 확인에서 실패했으므로 그 시도에서는 patch 적용이 시작되지 않았습니다. 이후 사용자가 추가로 작업했을 가능성은 직접 확인하세요.

저장소의 실제 host 경로는 아직 이 대화에서 확인하지 못했습니다. 다음 찾기 명령을 안내했으나 출력은 받지 못했습니다.

```bash
rg --files --hidden \
  -g 'run_prediction_suite.py' \
  -g '!**/.git/**' -g '!**/.cache/**' \
  "$HOME"
```

**즉, 첫 실제 작업은 저장소와 ZIP 위치를 찾고 `--repo`에 검증한 절대경로를 넣는 것입니다.** `~/CloudGPUFit`이라고 추측하지 마세요. 이 VS Code workspace가 이미 원본 저장소라면 그 경로를 검증해 사용하면 됩니다.

예정했던 별도 worktree 경로는 `/home/inuos/CloudGPUFit-cpu-recovery-20261108`입니다. 존재하면 무조건 지우거나 재적용하지 말고 Git 상태·적용 여부를 확인하세요. 기존 script는 대상이 있으면 안전하게 실패하도록 되어 있습니다.

## 10. 바로 이어서 할 일

### A. 현재 상태 확인과 패치 검토·적용

1. 실행 호스트, 원본 repo 절대경로, HEAD·dirty/worktree, ZIP/해제 경로를 확인합니다.
2. 현재 예측 프로세스와 컨테이너, 메모리 여유·디스크를 확인합니다. 과거 PID 468979·519373을 재사용해 kill하지 마세요. 다른 작업을 일괄 종료하지 마세요.
3. 원본 결과와 manifest·조건별 JSON/로그를 읽고 수집 이후 변경이 있는지 확인합니다.
4. ZIP checksum과 diff를 읽고, 특히 순차 파서·동률 순서·상주 storage·예외 처리·snapshot mount·취소/timeout cleanup이 타당한지 검토합니다. 추가 버그가 있으면 별도 작업 폴더에서 수정하세요.
5. 확인한 원본 repo 절대경로로 `prepare_recovery.py --repo <실제경로> --target <새작업경로>`를 실행하거나 동등한 isolated worktree 절차로 적용합니다. 기존 bind-mounted xMem을 덮어쓰지 마세요.

### B. 작은 기존 성공 조건 2개를 실제 VM에서 검증

아래는 경로와 기존 실행 여부를 확인한 뒤 사용할 시작 명령입니다. out이 이미 있으면 새 이름으로 바꾸세요.

```bash
cd /home/inuos/CloudGPUFit-cpu-recovery-20261108
python3 -u tools/run_prediction_recovery.py \
  --output-root /home/inuos/cloudgpufit-predictions/recovery-check-01 \
  --case-id core__decoder_llm__small__fp32__01 \
  --case-id core__decoder_llm__small__fp16__01 \
  --compare-with /home/inuos/cloudgpufit-predictions/native-v3-t4-j6 \
  --threads 4 --memory-gib 32 --host-reserve-gib 6 --timeout-seconds 3600
```

두 조건은 GPT-2, 배치1·입력256·생성64이며 정밀도만 다릅니다.

32GiB는 필요 RAM 예측값이 아니라 **작업 보호용 상한**이고, host reserve 6GiB는 신규 시작 판단 기준입니다. 실제 host 상태를 보고 타당성을 검토하세요. 한 시간도 완료 예상시간이 아니라 운영 제한입니다. 제한 때문에 실패하면 원인을 기록하고 설정/구조를 재검토하세요. 같은 조건을 무의미하게 반복하지 마세요.

확인할 것:

- manifest와 `.prediction.json`, `.log`를 직접 읽습니다.
- 기존 request·revision·threads가 같은지 확인합니다.
- allocated/reserved 각각 byte 단위 차이가 0인지 확인합니다. 차이가 나면 수명 목록·동률 순서·allocator 전이를 분석하고 허용 오차를 임의로 넓히지 마세요.
- phase별 시간·현재/최대 RSS를 확인해 CPU 추론/프로파일러 종료, export, 파싱, 수명 매칭, allocator 중 병목을 분리합니다.
- 현재 측정 phase 이름이 `measured_inference_and_profiler_stop`으로 합쳐져 있습니다. 이 둘의 비중을 알아야 한다면 저비용 계측을 더해 분리하세요. 독립 실행 중인 모델/trace 변환도 원인이 될 수 있어 전체 시간만으로 추론하지 마세요.

### C. 실패 대표·다른 유형·실제 정리 동작 검증

- 이전 실패 1개: `core__decoder_llm__small__fp32__12` (B4, 입력768, 생성128).
- T5의 self/cross-KV가 포함되는 기존 성공 조건과 Vision 성공 조건 각 1개를 plan과 결과에서 선정합니다.
- 필요하면 가장 큰 모델을 보호 상한·시간제한 안에서 1개 검증합니다. 처음부터 11~13시간 걸린 XL FP16 실패 두 개를 무제한으로 시작하지 마세요.
- 작은 작업으로 수동 취소·timeout·낮은 메모리 제한을 시험해 개별 작업만 정리되는지, OOMKilled와 일반 종료가 구분되는지 확인합니다. 호스트 전체 OOM을 일부러 재현하지 마세요.
- cgroup 제한과 source/cache read-only mount가 실제로 적용됐는지 확인합니다. RSS 합산은 공유 페이지가 중복될 수 있으므로 cgroup 값을 함께 보고 PSS는 필요한 경우 낮은 빈도로 수집합니다.

### D. 실패·미완료 복구와 서비스 관점 평가

- 대표 검증이 통과하면 자원·예상 총시간을 보고 실패·미완료를 순차 복구하는 단계로 진행합니다.
- 이 snapshot 기준 `--remaining-from` 대상은 17개 오류 + 81개 미실행 = 98개입니다. 현재 파일이 달라졌으면 다시 산출하세요.
- 기존 성공 244개는 지우거나 덮어쓰지 않습니다. 최적화 전후 동등성 근거와 provenance가 있어야 같은 분석에 재사용할 수 있습니다. 버전이 다른 결과를 출처 없이 섞지 마세요.
- dtype·조건별 실패율·시간·peak RAM을 정리하고, 여전히 FP16이 느리면 CPU 수치 연산·profiler 비용을 구분해 개선안을 제시하세요.
- 추론 횟수/초기 allocator 상태/FakeTensor 경로가 바뀌는 개선은 별도 버전·검증 과제로 분리합니다. 실제 CPU matmul을 생략하는 구조가 필요할 수 있지만 단순 weight+KV 수식 합산으로 대체하자는 뜻은 아닙니다.

## 11. 보존 원칙과 작업 방식

- 기존 결과·로그·실패 증거·컨테이너 의존성·HF cache를 삭제하거나 덮어쓰지 마세요.
- 원본 checkout과 `/home/inuos/xmem-work/xMem`은 bind mount로 연결될 수 있습니다. 원본 host 파일 수정도 실행 코드에 영향을 줄 수 있습니다.
- 원본에 `--prepare`, pip install, git reset/clean, Docker prune 같은 광범위한 변경을 하지 마세요.
- 검증용 worktree·출력·작업 컨테이너를 분리하고, 이 작업에서 만든 것으로 확인된 것만 정리하세요.
- `docker exec` CLI만 종료하면 내부 Python이 남을 수 있습니다. host CLI의 자식 트리만 보지 말고 내부 PID/cgroup과 연결하세요.
- SIGKILL/호스트 장애까지 Python finally가 처리할 수 있다고 주장하지 마세요. 실제 보장 범위와 남는 작업을 확인하세요.
- 조건별 프로세스 종료 여부·return code·signal·stage·자원·source/environment fingerprint를 남기세요.
- 무거운 원본 trace나 full allocator history는 작은 대표 조건에서만 비교하세요. 기록 최적화 확인을 위해 다시 호스트 OOM을 일으키지 마세요.
- 가능한 확인·수정·실험은 VS Code의 terminal/file 도구로 직접 수행하세요. 사용자가 매번 명령을 복사하거나 결과를 압축해 보내는 방식으로 되돌리지 마세요. 실제 접근 권한·필수 파일이 없을 때만 구체적으로 요청하세요.

## 12. 첫 응답과 완료 보고

첫 응답에는 현재 실행 환경·실제 repo/ZIP/worktree 위치·HEAD·진행 중 프로세스·패치 적용 여부를 확인해 짧게 보고하고, 바로 다음 2~3개 작업을 수행하세요. 아직 VM에서 검증하지 않은 부분을 완료라고 보고하지 마세요.

완료/중간 보고에는 다음을 포함하세요.

1. 오류 원인: 직접 확인 사실과 가설, OOM/timeout/수동 종료/일반 예외 구분
2. 실제 변경 파일·이유·source version
3. 조건별 allocated/reserved의 수정 전후 값과 byte 차이
4. 단계별 시간·peak RAM, 비교 환경·동시 부하
5. 프로세스 정리·작업 제한 검증 결과
6. 기존 결과 재사용 가능 범위와 남은 조건
7. 실제 CPU 연산을 유지한 현재 방식의 남는 속도 한계와 다음 조치

이제 환경과 경로 확인부터 실제로 시작해 주세요.

# Reserved 예측 v3 실험 과정 (2026-11-16, INUOS)

INUGPU 가이드 `RESERVED_PREDICTION_GUIDE_KO.md` §0을 받아, 가이드 검증 → 구현 → 전체 예측 → 비교 → 원인 확인 순서로 진행했다.
생산 코드(cpu_worker2)와 v2 결과는 바꾸지 않았다.

## 1. 가이드 주장 검증 — `guide_verification/`

- 도구: 고정된 xMem(1ee4bd0)에 GPU 요청 스트림과 CPU 요청 스트림을 각각 재생한다.
- GPU 트레이스 12건
  - 11-12 받은 6건: opt, t5-small, qwen, resnet-50, gpt2 b8, gpt2 b16
  - 11-16 받은 6건: gpt2 b128 2건, tinyllama 2건, t5-base, opt b16

| 단계 | 스크립트 | 결과 파일 | 확인한 것 |
|---|---|---|---|
| ① 주소 규칙 | `verify_guide.py` (`run.sh`, `specs.txt`) | `verify.jsonl` | stock xMem 주소 대신 내림차순 주소를 쓰면 GPU 스트림 재생이 실측에 맞는다 (12/12). CPU와 GPU의 로드 단계 요청은 같다 |
| ② 작업 공간 | `wstest.py`, `align.py`, `loadlen.py` (`run2.sh`, `specs_ws*.txt`) | `wstest_round1.jsonl`, `wstest.jsonl` | CPU 스트림에 GPU 전용 cuBLAS 작업 공간 8,519,680 B를 GPU와 같은 위치에 넣으면 남은 오차가 대부분 사라진다. 1차에서 기준 요청 찾기 오류(cuBLASLt 1 MiB)를 고친 뒤 2차 |
| ③ 재시도 | `retry.py` (`run3.sh`) | `retry.jsonl` | 용량을 실효 용량(cuda_total − 312 MiB)으로 두면 tinyllama b128/128/512는 오차 0, t5-base b128/128/512는 −84 MiB |
| 보조 | `patchtest.py` | — | xMem 인스턴스 패치가 결정에 주는 영향 확인 |

- 경로 `/h12`, `/h16`, `/runs`, `/out`은 docker 마운트다. 각각 handoff 20261112, handoff 20261116, 실험 runs, 이 폴더를 가리킨다.

## 2. 작업 공간 위치를 CPU만으로 찾기 — `runs/gemm_exact/`, `runs/gemm_probe/`

- ② 단계는 GPU 트레이스에서 위치를 가져왔다. 실제 예측에서는 GPU가 없으므로 CPU 프로브로 바꿨다.
- 1차 `cgf680.gemm_probe` (B=1 축소 실행, `runs/gemm_probe/`)
  - 실제 조건과 위치가 달라서 폐기했다.
- 2차 `gemm_probe --exact` (`tools/run_gemm_exact_all.py`)
  - 실제 조건과 같은 B·길이·generation config·원래 스레드 수로 실행하고, 첫 GEMM 8회 뒤 멈춘다.
  - 522/522 조건에서 CPU 저장 스트림 접두부와 같다. 결과: `runs/gemm_exact/positions.jsonl` (사본 `gemm_exact_positions.jsonl`)
  - 고친 불일치 원인
    - 한 프로세스에서 여러 모델을 실행함
    - `generation_config` 설정 누락
    - 스레드 수가 원래 실행과 다름
    - shadow 프레임 건너뛰기 실패 (그래서 `gemm_shadow.py`로 분리)

## 3. 전체 예측 — `rtx3090_v3/`

- `src/cgf680/replay3.py`: 내림차순 주소, 작업 공간 삽입, LightTrace, R_inf / R_at / R_min
- `tools/reserved_predict.py`: 컨테이너 3개로 실행했다(`reserved_v3_rtx3090_v3{,b,c}.console.log`).
  - 결과: 522개 모두 ok, 오류 0
  - 중복 행이 있으면 마지막 ok 행을 사용한다.
- 같은 실행 안에서 비교용 변형 2개도 계산했다.
  - stock 주소 + 작업 공간 없음 (= v2)
  - 내림차순 주소 + 작업 공간 없음

## 4. 비교 — `handoff_codex/reserved_prediction_*`

- `tools/compare_reserved.py`
  - GPU L256과 L1024 per-condition CSV를 쓴다.
  - 허용 오차 100 MB, 32 MiB 결과도 함께 낸다.
  - 오차가 큰 조건은 할당량 일치 여부로 원인을 나눈다.

## 5. 원인 확인 — `confirm/`

| 대상 | 스크립트 | 결과 |
|---|---|---|
| CNN | `cnn_attr.py`, `run_all.sh` | `attr.jsonl`: 합성곱 내부 버퍼가 장치마다 다르다 (확정) |
| qwen | `qwen_segtrace.py` | `qwen_seg_b32_in1024_new128.json`: GPU에만 560 MiB 세그먼트가 하나 더 있다. GPU 트레이스 요청 |
| 재시도 | `gpu_segment_addr.py`, `addrmodel_va.py`, `reusetest.py`, `retrypos.py` | `retry_reuse_policy.jsonl`, `retry_release_positions.json`: 해제 구간 재사용으로는 설명되지 않는다. 첫 해제 뒤에 갈라진다 (미확정) |

- 정보 출처(조건별 CPU 입력 / GPU 종류별 고정값)는 `handoff_codex/reserved_prediction_inputs_KO.md`에 정리했다.

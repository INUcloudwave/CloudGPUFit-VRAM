# 00. 보고서용 전체 실험 지도 — INUGPU · INUOS · AWS 통합 (2026-11-18 기준)

> **보고서를 쓰는 AI는 이 문서를 가장 먼저 읽는다.**
> 이 문서만으로 "무엇을, 왜, 어떻게 했고, 결과가 어땠고, 그래서 무엇을 반영했는지"를 알 수 있게 썼다.
> 수치의 근거 파일은 각 줄의 경로나 상세 문서(`docs/01~07`, `cpu-vm/docs/01~03`)에 있다.

- 표기
  - "확인됨": 결과 파일로 확인한 사실
  - "추정": 근거는 있지만 직접 검증하지 않은 해석
  - "미확인": 아직 검증하지 않은 내용
- 단위: 원값 byte, MiB = 2^20 B

---

## 1. 프로젝트와 역할

- **CloudGPUFit**: GPU를 빌리기 전에 CPU만으로 추론 워크로드의 GPU VRAM 사용량을 예측하고, 그 결과로 AWS GPU 인스턴스를 추천하는 졸업작품이다.
- 예측식: **예측 VRAM = 예측 reserved(PyTorch 할당기가 잡는 메모리) + outside(할당기 밖, GPU별 상수)**
  - reserved는 xMem(PyTorch CUDA caching allocator를 CPU에서 흉내 내는 시뮬레이터)으로 예측한다.

| 주체 | 하드웨어 | 역할 | 레포 경로 | 맥락 문서 |
|---|---|---|---|---|
| INUGPU | RTX 3090 24GB VM | GPU 실측(정답값), 측정 코드·AWS 패키지 제작, reserved 원인 분석 | `gpu-vm/` | `docs/ai-context/AI_CONTEXT_KO.md` |
| INUOS | CPU 전용 VM (Xeon 24 vCPU, 47 GiB) | CPU에서 같은 조건을 실행 → 요청 목록 기록 → xMem 재생으로 GPU VRAM 예측 | `cpu-vm/` | `cpu-vm/INUOS_CONTEXT_KO.md` |
| AWS | g4dn(T4), g5(A10G), g6(L4), g6e(L40S), g7(Blackwell) 등 | 같은 패키지로 다른 GPU 정답값. 사용자가 직접 실행하고 INUGPU가 패키지를 만들고 분석 | 결과는 `gpu-vm/` 아래 | `docs/02`, `docs/06` |

- 두 VM의 에이전트는 서로 대화하지 못한다.
  - 문서·trace는 INUGPU → INUOS 방향 scp와 handoff 폴더로만 전달했다(`gpu-vm/05-fp16-vram/handoff_inuos/`).
  - 반대 방향 키는 사용자가 의도적으로 열지 않았다.

## 2. 전체 시간순 이력 (주체 · 실험 · 왜 · 결과 · 반영)

| 날짜 | 주체 | 실험 | 왜 | 결과 (핵심) | 반영한 것 | 상세 |
|---|---|---|---|---|---|---|
| 03-31 ~ 09-14 | INUGPU | 초기 벤치 (ResNet/YOLO), Qwen 14B 적재 | 초기 탐색 (CloudSim 입력값) | 14B FP16은 24 GB에 strict로 안 들어감 | — | `docs/01` |
| 08-16 ~ 08-23 | INUGPU | xMem 단독 검증 프로브 | xMem 신뢰성 | 결과 대부분 stdout만 남음 | — | `docs/03` |
| 10-06 ~ 10-27 | INUGPU | cloudgpufit-bench 342조건 × 5모드 (v1 → v3) | GPU 정답값 1차 | v1 17.8h → v3 3.7h. memory peak 342 중 326 일치 | 이후 측정 재설계의 출발점 | `docs/01` |
| 10-26 ~ 11-10 | INUOS | CPU 예측기 v2 휴리스틱 → native v3 → singlepass v4 (342조건) | CPU 예측 | v4: 325 ok / 17 실패 | 휴리스틱 폐기, trace + xMem 방식 확정 | `cpu-vm/docs/01` |
| 11-03 ~ 11-10 | INUGPU + AWS | **outside 연구** | 예측식의 outside 정의·측정 | 3090 첫 추론 후 312 MiB 고정(254 → 296 → 312). EAGER면 +728 MiB. AWS: T4 130, A10G 308, L4 232, L40S 510 | outside = GPU별 상수, LAZY 고정 | `docs/02` |
| 11-10 ~ 11-11 | INUOS 요청 → INUGPU 수집 | **reserved 오차 원인 분리** | 342 CPU 예측 vs AWS L4 reserved 오차가 큼 | 원인은 실행 이력(검증 forward·performance 모드가 캐시를 키움). 할당기 정책은 원인 아님 | **측정 재설계**: 조건마다 새 컨테이너, 로드 → warmup 3 → measure 5 | `docs/03`, `cpu-vm/docs/02` |
| 11-11 | INUGPU | FP16 VRAM 파일럿 v1(eager) / v2(기본 attention), xMem 재생 | 새 측정 방식 검증 | 기본 attention = sdpa(Flash). peak는 warmup에서 확정. GPU trace를 xMem에 넣으면 완전 일치 | v2(기본 attention) 확정. "차이는 요청 목록에서 찾는다" | `docs/04` |
| 11-11 ~ 11-15 | INUOS | FP16 CPU 프로토콜 v1 → v2 (GPU 절차 정렬), main395 L1024 | GPU와 같은 절차로 예측 | 입력 sha 375/375, 로딩 직후 reserved 374/374 일치. allocated −0.1 ~ −1.5% | CPU worker v2 확정 | `cpu-vm/docs/03` |
| 11-12 | INUGPU | 본실험 395 L1024 (3090) | 정답값 | 381 ok / 14 OOM | — | `docs/04` §2.9 |
| 11-15 ~ 11-16 | INUGPU / INUOS | 본실험 395 **L256** (3090 실측 / CPU 예측) | 1024가 너무 오래 걸려 256으로 변경 (사용자 결정) | GPU 384 ok / 11 OOM. CPU 386 ok / 9 cpu_oom | 이후 기준 조건표 | `docs/04` §2.10, `inuos` §4.1 |
| 11-16 | AWS (사용자) | **g4dn.xlarge(T4) L256 395** | 다른 GPU 정답값 | 378 ok / 17 OOM | T4 기준값 | `docs/06` §2 |
| 11-16 | INUGPU | **reserved 분석** E1 ~ E7 (구성 분해, 주소 순서, 반복 변동, R_min) | CPU 예측 reserved 오차의 근본 원인 | slack은 요청 순서·수명·주소가 정함. 내림차순 주소 규칙으로 19/20 일치. 같은 조건도 최대 4.5% 흔들림. xMem R_min ≈ GPU (LLM·T5) | **예측 가이드**(내림차순 주소, cuBLAS 작업공간, R_inf/R_min 판정) → INUOS 전달. 판정 규칙 정정(재시도 구간 reserved는 예측 안 함) | `docs/05` |
| 11-16 | INUOS | **reserved 예측 v3** (가이드 반영) | CPU만으로 reserved 예측 | 519조건 중 100 MB 이내 91.9% (v2 80.5%). 일반 96.8%, 재시도 60.9%, CNN 68.3%. 재시도·OOM 판정 23/23, 3/3 | 남은 원인 확인을 위해 INUGPU에 GPU trace 5건 요청 (**미실행**) | `inuos` §4.2 |
| 11-17 | AWS + INUGPU | g7(Blackwell) 실패 → **cu128 이미지** | 신형 GPU | cu124 이미지는 sm_120 미지원. cu128-v1 제작 | 이미지 선택 기능, 라벨 `__cu128` | `docs/06` §3 |
| 11-17 | INUGPU | **3090 cu124 ↔ cu128** 395 | g7 결과를 버전 차이와 GPU 차이로 나누기 위한 기준 | reserved 355/384 동일. CNN·ViT 수십 MiB. outside 변화 없음 | CNN 보정은 cuDNN 버전별로 | `docs/06` §4 |
| 11-17 ~ 11-18 | AWS (사용자) | **g4dn 환경 비교 A → C → B** (39조건) | 팀 요청: 드라이버·PyTorch 배포 영향 분리 | 드라이버 595 → 580, torch 2.6.0 → 2.5.1 모두 allocated·reserved 동일, outside ≤ 2 MiB. torch 2.5.1은 `.bin` 모델 로드 불가 | 드라이버는 예측 입력에서 제외. PyTorch는 호환성 조건 | `docs/06` §5 |
| 11-18 | INUGPU 분석 | T4 ↔ 3090 (같은 이미지) | GPU 종류 효과 | allocated 291/378 동일(나머지 ±2.5 MiB). reserved는 T4 용량에서 재시도가 생긴 조건에서 크게 다름. qwen in512 +928 MiB 미확인 | 요청 목록은 GPU와 거의 무관함을 확인. reserved는 용량 반영 필수 | `docs/06` §6 |

## 3. 최종적으로 정리된 사실 (보고서의 "결과" 장)

1. **측정 방법론**: 조건마다 새 컨테이너, 로드 → warmup 3 → measure 5, 캐시 유지
   - 이 방식에서 peak reserved는 warmup_0에서 정해지고, 측정 5회가 모두 같다(765회 확인).
   - 이전 방식(한 프로세스에서 여러 모드)은 실행 이력 때문에 reserved가 부풀었다.
2. **재현성**
   - 3090: L1024와 L256에 겹치는 245개 중 243개가 바이트 단위로 같았다.
   - 같은 조건을 15회씩 반복하면 일부 조건은 두 값으로 갈렸다(최대 44 MiB, 4.5%).
   - T4: 다른 날·다른 인스턴스·다른 물리 GPU에서도 39/39 바이트 단위로 같았다.
3. **reserved 구성**: 가중치 + 최대 allocated + 라이브러리 작업공간(cuBLAS 8,519,680 B 등) + slack
   - slack은 공식으로 못 맞히고, GPU와 같은 요청 목록을 xMem에 재생해야 맞는다.
   - 요청이 같을 때 남는 오차는 세그먼트 주소 순서에서 온다. 내림차순 규칙으로 해결된다.
4. **용량과 판정**
   - C ≥ R_inf이면 재시도 없이 reserved = R_inf
   - R_min ≤ C < R_inf이면 실행은 가능하지만 reserved는 경로 의존이라 예측하지 않음
   - C < R_min이면 OOM
   - xMem R_min은 LLM·T5에서 GPU와 0 ~ +3.5% 차이였고, 과소 예측은 없었다.
5. **CPU 예측 정확도** (INUOS v3, 3090): 519조건 중 100 MB 이내 91.9%
   - 남은 오차: CNN 합성곱 내부 버퍼 차이(확정), qwen 560 MiB 세그먼트(미확정), 재시도 조건 일부
6. **outside**: GPU 종류별 상수
   - T4 130/132, 3090 312 ~ 316, L4 232, A10G 308, L40S 510 MiB
   - 드라이버(R595/R580), PyTorch(2.5.1 ~ 2.7.1), CUDA(12.4/12.8), 모델·배치와 무관
7. **실행 환경 영향**
   - 드라이버: 0
   - PyTorch 배포판: 메모리 0, 호환성 문제 있음(torch < 2.6에서 `.bin` 모델 로드 불가)
   - CUDA/cuDNN 버전: CNN·ViT 수십 MiB
   - GPU 종류: allocated는 거의 0, reserved는 용량·커널에 따라 다름 (T4는 메모리 효율 attention, 3090은 Flash)

## 4. 의사결정 기록 (누가 왜 결정했나)

| 날짜 | 결정 | 결정자 | 이유 |
|---|---|---|---|
| 11-10 | 측정을 VRAM 전용으로 재설계 | INUGPU (INUOS 진단 결과) | 실행 이력이 reserved 오차의 원인 |
| 11-11 | 기본 attention(v2)을 기준으로 | 사용자·INUGPU | 실제 사용 환경과 일치. eager는 Qwen NaN |
| 11-15 | 조건표 1024 → 256 | 사용자 | 실행 시간 |
| 11-15 | RAM 부족으로 실패한 CPU 조건은 재시도하지 않고 "예측 불가"로 기록 | 사용자 | INUOS RAM 한계 |
| 11-16 | 판정 규칙: 재시도 구간 reserved는 예측하지 않음 | INUGPU (R_min 실측 근거) | reserved가 C 근처가 아니라 경로 의존 |
| 11-17 | g7은 별도 cu128 이미지로 측정, 결과는 별도 라벨 | 사용자 | 기존 이미지 미지원 |
| 11-17 | C 환경 opt·vit 오류는 그대로 기록하고 C′(우회 실행)는 하지 않음 | 사용자·INUGPU 원칙 | 오류도 결과. 조건 대체 금지 |
| 11-18 | C attention 진단은 미실시로 남김 | INUGPU 권고 | 드라이버를 되돌려야 하고 결론에 영향 없음 |

## 5. 코드가 어떻게 바뀌었나

- 요약: 측정 워커는 11-11의 `d06f0f15` 이후 바꾸지 않았다.
  - 추가 분석은 모두 **래퍼와 러너 복사본**으로 했다(`cap_worker`, `outside_worker`).
  - 이미지는 `cu124-v1` → (추가) `cu128-v1`, `pt251-cu124-v1`
  - INUOS도 생산 worker(`cpu_worker2`)는 두고, v3 예측을 새 파일로 추가했다.
- 전체 계보: `docs/07_code_env_lineage_KO.md`

## 6. 남은 일 (2026-11-18)

| 우선 | 항목 | 주체 | 상태 |
|---|---|---|---|
| 높음 | g7(Blackwell) 결과 수신·분석 (3090 cu128과 비교해 GPU 효과, Blackwell outside) | 사용자 → INUGPU | AWS 측정 완료, tar 미수신 |
| 높음 | INUOS GPU trace 5건 요청 (`cpu-vm/03-fp16-cpu-prediction/handoff_codex/INUGPU_TRACE_REQUEST_20261116_KO.md`: qwen 3, resnet-50, mobilenet) | INUGPU | **미실행** |
| 높음 | T4 GPU trace 몇 건 (T4는 메모리 효율 attention → 요청 목록이 3090과 같은지, qwen +928 MiB 원인) | AWS | 제안 |
| 중간 | GPU별 outside 상수표 완성 (g7 + 후보 GPU 작은 조건 실측) | AWS + INUGPU | 일부 있음 |
| 중간 | tinyllama b128/128/256 +204 MiB(cu128) 반복 측정, CNN cuDNN 영향 분리 | INUGPU | 미실행 |
| 중간 | 대형 조건(gpt2-medium/xl, t5-3b) R_inf/R_min, FakeTensor 요청 목록 시험 | INUGPU | 미실행 |
| 낮음 | 재시도 구간 응답시간 영향 | INUGPU | 미실행 |

## 7. 어디를 보면 되나

| 알고 싶은 것 | 문서 |
|---|---|
| 현재 측정 방식의 정확한 정의 | `docs/ai-context/AI_CONTEXT_KO.md` §5, `docs/04` |
| reserved 구성·예측 가이드·R_min | `docs/05`, `gpu-vm/05-fp16-vram/handoff_inuos/` |
| 드라이버·PyTorch·CUDA·GPU·outside 영향 | `docs/06` |
| 코드·이미지·번들 계보 | `docs/07` |
| CPU 예측 방법과 v3 정확도 | `cpu-vm/INUOS_CONTEXT_KO.md`, `cpu-vm/docs/03`, `cpu-vm/03-fp16-cpu-prediction/handoff_codex/reserved_prediction_report.md` |
| outside 초기 연구 | `docs/02`, `gpu-vm/02-outside-analysis/NOTION_SUMMARY_20261108.md` |
| 조건별 GPU 정답값 (byte) | `gpu-vm/05-fp16-vram/results/gpu_*_per_condition.csv` (3090 L1024·L256·cu128, T4 L256) |
| 레포에 없는 대용량 | `EXCLUDED_LARGE_FILES.md` |
| 노션 | "졸업작품 / 실험 정리" 아래 "reserved 분석", "실행 환경 영향 분석", "보고서용 전체 실험 지도" |

# 일반화 시험: 규칙 개발에 쓰지 않은 모델 6개 (2026-11-22~23)

- 모델: convnext-tiny-224, efficientnet-b0, regnet-y-040, Qwen2.5-0.5B-Instruct, SmolLM2-360M-Instruct, pythia-410m-deduped
  - 고정 리비전: `manifest/model_pins_newmodels_20261122.json`
- 조건: 96개
  - CNN: b{1,8,32,128} × r{128,224,256,512}
  - LLM: b{1,8,32,128} × in{128,512} × new{128,512}
- CPU (INUOS): `runs/newmodels_20261122` (+`_retry2` 1개), 96/96 ok
  - 스케줄러: `tools/newmodels_chain.py`, `tools/run_newmodels.py`
- GPU 정답: INUGPU RTX 3090 `cgf-fp16-inuos-handoff-20261122_newmodels`, 96/96 ok (`gpu_rtx3090_newmodels_20261122_per_condition.csv`, m1_peak_reserved)
- 예측
  - LLM: v3 (`tools/reserved_predict.py` → `results/reserved_v3/newmodels_rtx3090*`), cuBLAS 위치 `runs/gemm_exact_newmodels` 96/96
  - CNN: 층 구분 48/48이 저장된 요청 목록과 같음 (`runs/layer_labels_newmodels`)
    - 1단계 + cuDNN 표 + cuBLASLt·max_pool 규칙
    - 표 = 3090 probe v3 3,253개 + 새 형태 953개 (`gpu_probe/results/rtx3090_newmodels`, INUGPU 11-23)
- 스크립트: `newmodels_eval.py` → `summary.json`, `cnn_replays.jsonl`, `llm_rows.jsonl`

## 결과 (RTX 3090, reserved)

| 모델 | 100 MB 이내 | 바이트 일치 | 최대 과대 / 과소 |
|---|---|---|---|
| convnext-tiny | 16/16 | 16 | 0 / 0 |
| efficientnet-b0 | 16/16 | 16 | 0 / 0 |
| regnet-y-040 | 16/16 | 16 | 0 / 0 |
| pythia-410m | 16/16 | 15 | +2 / 0 MiB |
| smollm2-360m | 16/16 | 15 | +58 / 0 |
| qwen2.5-0.5b | 14/16 | 12 | 0 / −610 |

- CNN 단계별 (48개 중 100 MB 이내 / 바이트 일치)

| 방식 | 100 MB 이내 | 바이트 일치 | 최대 과소 |
|---|---|---|---|
| v3 | 31 | 10 | −2,562 MiB |
| 1단계 | 40 | 32 | −2,562 MiB |
| + 기존 표만 (새 형태는 0) | 40 | 32 | −2,562 MiB |
| + 격자 규칙 | 40 | 32 | −2,562 MiB |
| **+ 새 형태 측정 + cuBLASLt·max_pool** | **48** | **48** | 0 |

- regnet의 −2,562 MiB는 grouped 3×3 합성곱의 작업 공간 때문이다.
  - 형태 하나하나의 작업 공간은 작지만(대부분 기존 세그먼트 안에 들어감), 115번의 호출에 걸쳐 영향이 쌓였다.
  - 격자 규칙은 grouped 종류가 격자에 없어서 고치지 못했다.
- qwen −610 MiB (b128 in512 new128/new512): qwen 1.5B에서 찾은 FP32 그림자 현상과 같은 것으로 추정한다(RMSNorm `.to(float32)`).
  - AWS r7i fp16_real로 확인 중이다 (`aws/run_newmodels.sh`).

## AWS 실제 FP16 실행으로 확인 (2026-11-23, `aws_fp16_real/`)

- 실행: AWS r7i(16 vCPU, 124 GiB, Sapphire Rapids), `--dtype-mode fp16_real`, LLM 48개 조건 48/48 ok (2.6시간)
  - 이미지·작업자·스레드 수는 INUOS와 같다. 결과 tar sha256 81c39f7b…dead4 (INUOS `~/aws-result/`)
- 비교 (`aws_fp16_real/compare.py` → `compare.json`, v3 R_at 3090 실효 용량 vs 3090 실측)

| 모델 | INUOS (FP32 + 크기 환산) 100 MB 이내 / 일치 / 범위 | AWS 실제 FP16 |
|---|---|---|
| qwen2.5-0.5b | 14/16, 12, −610 ~ 0 MiB | **16/16**, 13, −76 ~ +2 |
| smollm2-360m | 16/16, 15, 0 ~ +58 | 16/16, 10, −2 ~ +58 |
| pythia-410m | 16/16, 15, 0 ~ +2 | 16/16, 10, −8 ~ +32 |
| 합계 48 | 46, 42 | **48**, 33 |

- qwen −610 / −608 MiB (b128 in512) → AWS 0 / −2 MiB
  - qwen 1.5B와 같은 FP32 그림자 현상(RMSNorm `.to(float32)`)이 다른 크기의 qwen에서도 확인됐다.
- smollm2 +58 MiB (b128 in128 new512)는 두 방식에서 같다 → FP32 현상이 아니다. 원인은 미확인이다.
- 실제 FP16 실행은 다른 조건에서 작은 차이가 생긴다(일치 42 → 33, 최대 +32 / −76 MiB). CPU FP16 커널의 버퍼 차이로 추정한다(11-17 AWS 395개 결과와 같은 경향).
- 결론: 두 방식 모두 장단점이 있다. INUOS 방식에서 RMSNorm 해제 시점만 고치는 것이 가장 좋아 보인다(다음 과제).

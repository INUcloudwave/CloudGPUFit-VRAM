# INUGPU 요청: reserved 오차 원인 확정용 GPU trace 5건 (2026-11-16, INUOS)

- 언제: GPU가 비어 있을 때. 본실험과 겹치지 않게 순차로 실행해 주세요.
- 왜: CPU 예측 v3(`handoff_codex/reserved_prediction_report.md`)에서 100 MB를 넘는 원인 중 CPU만으로는 확정할 수 없는 것이 두 가지 남았습니다.
  - **qwen prefill**: 할당 최대치(allocated)는 GPU와 정확히 같습니다. 그런데 GPU reserved가 정확히 배치 × 입력 길이 × 8960 × 2 B(MLP 중간 텐서 1개)만큼 더 큽니다. 7개 조건에서 나타납니다.
    - 확인할 것: GPU 요청 순서가 CPU와 다른지(어텐션 커널 경로 등), 아니면 allocator 동작 차이인지
  - **CNN GPU 쪽 내부 버퍼**: CPU에서 합성곱 내부 버퍼는 확인했습니다(아래 참고).
    - 확인할 것: GPU 쪽 크기(cuDNN workspace로 추정, 출력 텐서 대비 resnet-50 14.5 %, mobilenet 11 %)를 요청 단위로 확인
- 재시도 조건의 주소 재사용은 이미 받은 t5-base trace로 확인 중이므로 추가 요청은 없습니다.

## 실행 (기존 `run_trace_slack.sh`와 같은 방식)

```
cd /home/inugpu/cgf-fp16-vram-20261111
# L256 조건표
python3 -I src/run_conditions.py --conditions conditions_main395_l256.json \
  --list qwen2.5-1.5b-instruct__b128__in256__new128,qwen2.5-1.5b-instruct__b128__in128__new128,resnet-50__b128__r224,mobilenet-v2-1.0-224__b128__r224 \
  --mode trace --gpu-label rtx3090__trace_confirm --attn default
# L1024 조건표
python3 -I src/run_conditions.py --conditions conditions_main395.json \
  --list qwen2.5-1.5b-instruct__b32__in1024__new128 \
  --mode trace --gpu-label rtx3090__trace_confirm --attn default
```

| 조건 | 목적 |
|---|---|
| qwen b128/in256/new128 | −560 MiB 조건 |
| qwen b32/in1024/new128 | −560 MiB 조건. 토큰 수는 위 조건과 같고 형태만 다릅니다 |
| qwen b128/in128/new128 | 대조군 (오차 +90 MiB) |
| resnet-50 b128/r224 | CNN, GPU 내부 버퍼가 더 큼 (−348 MiB) |
| mobilenet b128/r224 | CNN, CPU 내부 버퍼가 더 큼 (+402 MiB) |

- 결과물: 조건별 `trace.json`, `trace.snapshot.pickle.gz`, `SHA256SUMS`
- 보낼 위치(INUOS): `/home/inuos/cgf-fp16-inuos-handoff-<날짜>/traces/<cid>/`
- 함께 기록할 것: `torch.backends.cudnn.version()`, `cudnn.benchmark` 값, 실행 중 다른 GPU 프로세스가 없었는지

## INUOS에서 할 일

- GPU 요청 스트림을 v3와 같은 방식(내림차순 주소, 실효 용량)으로 xMem에 재생합니다.
  - GPU 값이 재현되면: CPU 요청 목록과 다른 곳을 찾아 원인을 확정합니다.
  - 재현되지 않으면: allocator 모델 쪽 문제로 확정합니다.
- CNN: GPU 요청 중 연산 출력이 아닌 것(cuDNN workspace 등)의 크기와 시점을 CPU 내부 버퍼와 비교합니다.
- 실패한 조건의 배치·dtype은 바꾸지 마세요.

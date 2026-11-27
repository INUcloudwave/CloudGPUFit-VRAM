# INUGPU 요청: CNN 요청 단위 trace 2건 (2026-11-15, INUOS)

- 언제: RTX 3090 L256 본실험(measure + validate)이 **끝난 뒤**. 본실험 중에는 실행하지 마세요(GPU 1개, 순차).
- 왜: CNN만 allocated 피크가 CPU 예측과 크게 다릅니다(RTX 3090 L1024 기준 모델별 중앙값).
  - resnet-18: −16%
  - resnet-50: −9%
  - mobilenet_v2: +28%
  - LLM·T5는 ±1.5% 이내입니다.
  - 원인 후보는 cuDNN workspace, conv 알고리즘의 중간 버퍼, CPU(mkldnn) 쪽 추가 버퍼입니다. 요청 단위 기록이 없어 확인할 수 없습니다.

## 요청 내용
- 대상 condition_id (조건표 `conditions_main395_l256.json`)
  - `resnet-18__b32__r224`
  - `mobilenet-v2-1.0-224__b32__r224`
  - 여유가 있으면 `resnet-50__b32__r224`
- 방법: 본실험과 같은 이미지·worker·인자에서 `--mode trace`만 바꿉니다.
  - `vram_worker.py ... --condition <cid> --mode trace --attn default --out /out/trace.json`
  - worker sha256 `d06f0f15…`는 그대로 둡니다.
- 결과 파일: `trace.json`, `trace.snapshot.pickle.gz`
  - 보낼 위치(INUOS): `/home/inuos/cgf-fp16-inuos-handoff-<날짜>/trace_cnn/<cid>/`
  - `SHA256SUMS`도 함께 보내 주세요.
- 함께 기록할 것
  - `torch.backends.cudnn.version()`
  - `cudnn.benchmark=False` 확인
  - 실행 중 다른 GPU 프로세스가 없음

## INUOS에서 할 일 (참고)
- GPU snapshot의 요청을 CPU trace와 크기·순서로 대조합니다.
  - GPU 전용 요청 확인: cuDNN workspace, cuBLAS 등
  - CPU 전용 요청 확인
  - 같은 텐서의 크기 차이 확인
- 같은 GPU 요청을 xMem에 재생해 실측과 맞는지 확인합니다. allocator 정책을 검증하는 단계입니다.

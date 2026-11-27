# cuDNN 대체 계획 검증 (INUOS가 실행, 2026-11-23, 사용자 승인)
- 조건: regnet-y-040 b128 r512 (새 모델 조건표), 메모리 상한 8개: 7946 5900 5640 5380 5130 4900 4872 4840 MiB
- 코드: src = cgf-newmodels-20261122/src (vram_worker d06f0f15), src_cap·tools/run_conditions_cap.py = cgf-fp16-vram-20261111 복사본 (11-16 R_min 실험과 같음)
- INUOS 사전 예측: cap_predictions_preregistered.jsonl (sha256 dc1e887c...)

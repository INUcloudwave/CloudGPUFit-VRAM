# 저장소에서 제외한 파일 (GPU VM)

> **원본 없음**: 아래 파일은 GitHub 업로드 제약 등으로 저장소에 넣지 않았고, 실험에 쓴 VM이 삭제되어 원본은 더 이상 남아 있지 않습니다. 경로·크기·sha256은 기록용으로만 남깁니다. 분석 결과와 요약은 저장소에 포함된 결과 파일과 docs/ 문서에서 확인할 수 있습니다.

- 기준: 50MB 초과 (GitHub 100MB 제한·저장소 크기) 또는 모델 가중치
- 모델 캐시(hf-cache), Docker 이미지 tar, xMem 원본 저장소(.git) 제외

| VM 경로 (/home/inugpu 기준) | 크기(B) | sha256 | 사유 |
|---|---|---|---|
| `cgf-reserved-diag-20261110/cgf-reserved-diag-20261110-return.tar.gz` | 89,819,792 | `030d347566e02549fb35c95e453b4c08ab6b67118a8fc7fa3755214233d132ec` | over 50MB |
| `cgf-reserved-diag-20261110/return/core__decoder_llm__medium__fp32__12/R0_aws_replica/memory_snapshot.pickle` | 323,947,640 | `13e91ae2e16966e877bc24e1f2667d1eebf536e11386bf2652d5b045917fcb83` | over 50MB |
| `cgf-reserved-diag-20261110/return/core__decoder_llm__medium__fp32__12/R1_controlled/memory_snapshot.pickle` | 129,665,241 | `31dcd3287ffe2ae35aca10267567364108fa41ae7625eb093705422f11db21dc` | over 50MB |
| `cgf-reserved-diag-20261110/return/core__decoder_llm__medium__fp32__12/R1_controlled__stacks_all_dladdr/memory_snapshot.pickle` | 548,129,191 | `7e4a8a4c57da4a27b2da578dab5d1a74093b91ad7c4fa5649a8dee3e977470c2` | over 50MB |
| `cgf-reserved-diag-20261110/return/core__decoder_llm__small__fp16__01/R0_aws_replica/memory_snapshot.pickle` | 102,008,264 | `bdb41028cb0ab902fc60208189fc85e12298ba01d68e12521270e7aa7daa27ed` | over 50MB |
| `cgf-reserved-diag-20261110/return/core__decoder_llm__small__fp16__12/R0_aws_replica/memory_snapshot.pickle` | 209,522,340 | `755d350aa8e5cd49d8961d696715da1fc85913bbef95718976bb3226a817ab57` | over 50MB |
| `cgf-reserved-diag-20261110/return/core__decoder_llm__small__fp16__12/R1_controlled/memory_snapshot.pickle` | 83,884,255 | `2693d7b471500ed8c5e6e9a47d924a66c7ce2ac48ae7b7330302dd7eb5829d04` | over 50MB |
| `yolov5su.pt` | 18,581,255 | `054272ddbbb3035cea7ff6b97e5becea63d2cc57a4f06a2a8133f4d1a56e74ed` | model weights |
| `cgf-fp16-aws-bundle-20261117-cu128/cgf-fp16-aws-cu128-20261117.tar` | 4,266,414,080 | `5030113d70edb5039c30302b04a82784acc31b5d563a6cdda834a4119f5eb8d6` | AWS 번들 (cu128 이미지 포함, g7용, 2026-11-17) |
| `cgf-envcmp-bundle-20261117/cgf-envcmp-g4dn-20261117.tar` | 4,509,859,840 | `a06e8bb2753a789e28b37b640f874adf49676fb6a99b1065fd1316f9695741ac` | AWS 번들 (cu124-v1 + pt251-cu124-v1 이미지 포함, g4dn 환경 비교, 2026-11-17) |
| docker 이미지 `cloudgpufit-bench:cu128-v1` / `pt251-cu124-v1` | — | index `798b909f…` / `0b22fba1…` | 이미지 (정의: `measurement/local-gpu-vm/05-fp16-vram/aws/Dockerfile.cu128`, `Dockerfile.pt251`, `IMAGE_INFO_cu128.json`, `IMAGES_INFO_envcmp.json`) |

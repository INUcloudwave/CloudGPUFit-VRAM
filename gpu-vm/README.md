# gpu-vm/ — INUGPU(RTX 3090 VM)에서 수행한 실험의 코드·결과

| 디렉터리 | 내용 | 상세 문서 |
|---|---|---|
| `00-early-scripts/` | 초기 ResNet-50/YOLOv5 벤치(2026-03), Qwen 14B OOM 테스트(2026-09) | `../docs/01_early_and_bench.md` |
| `01-cloudgpufit-bench/` | 342조건 벤치 하네스 v1~v3와 결과(2026-10) | `../docs/01_early_and_bench.md` |
| `02-outside-analysis/` | outside(NVML − reserved) 분석(2026-11-03~10) | `../docs/02_outside.md` |
| `03-xmem-probes/` | xMem 검증 프로브(2026-08), FP16 재생(11-11), xMem 로컬 수정 패치 | `../docs/03_xmem_probes_reserved_diag.md` |
| `04-reserved-diag/` | reserved 오차 원인 분리 GPU trace 수집(11-10) | `../docs/03_xmem_probes_reserved_diag.md` |
| `05-fp16-vram/` | FP16 VRAM 실측: 파일럿·본실험 395·AWS 패키지(11-11~) | `../docs/04_fp16_vram.md` |
| `05-fp16-vram/` (실행 환경 영향) | T4 L256(AWS), 3090 cu124↔cu128, g4dn 드라이버·PyTorch 비교(envcmp), g7용 cu128 이미지 (11-16~18) | `../docs/06_env_gpu_effects_KO.md`, 계보 `../docs/07_code_env_lineage_KO.md` |
| `05-fp16-vram/` (reserved 분석) | 구성 분해 trace·반복 변동·GPU R_min 실측(11-16). xMem 쪽은 `03-xmem-probes/reserved-anatomy-20261116/` | `../docs/05_reserved_analysis_KO.md` |

전체 맥락: `../AI_CONTEXT_KO.md`

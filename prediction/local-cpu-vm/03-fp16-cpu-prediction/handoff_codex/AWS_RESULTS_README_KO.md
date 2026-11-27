# AWS 실행 결과 (GPU 실측 5종 + L4 반복 1회 + CPU 실제 FP16 예측, 2026-11-16 ~ 11-18)

- 모두 사용자가 직접 AWS에서 실행했다. 결과는 INUOS `~/aws-result`로 받아 INUOS에서 분석했다.
- 전체 맥락은 루트의 `docs/ai-context/REPORT_CONTEXT_KO.md`를 먼저 본다.

## 1. GPU 실측 — `gpu_measurements/`

- 무엇: INUGPU와 같은 측정 패키지(`measurement/local-gpu-vm/05-fp16-vram/aws/RUN_ON_AWS.sh`, 작업자 `vram_worker.py`)로 395개 조건(L256 조건표)을 측정했다.
- 절차: 체크섬 → 이미지 검증 → 모델 캐시 해시 검증 → 파일럿 26개 → 본실험

| 폴더 | 인스턴스 | GPU | 이미지 | 결과 |
|---|---|---|---|---|
| `g4dn.xlarge-l256/` | g4dn.xlarge | Tesla T4 (cc 7.5) | cu124-v1 (torch 2.6.0, cuDNN 9.1.0) | 378 ok / 17 gpu_oom |
| `g5.xlarge-l256/` | g5.xlarge | A10G (8.6) | cu124-v1 | 382 / 13 |
| `g6.xlarge-l256/` | g6.xlarge | L4 (8.9) | cu124-v1 | 382 / 13 |
| `gr6.4xlarge-l256/` | gr6.4xlarge | L4 (8.9) | cu124-v1 | 382 / 13 (g6와 같은 GPU, 다른 인스턴스) |
| `g6e.xlarge-l256/` | g6e.xlarge | L40S (8.9) | cu124-v1 | 390 / 5 |
| `g7.2xlarge__l256__cu128/` | g7.2xlarge | RTX PRO 4500 Blackwell Server Edition (12.0) | **cu128-v1** (torch 2.7.1, cuDNN 9.7.1) | 387 / 8 |

- 드라이버는 모두 595.112.07이다.
- g7은 cu124 이미지에 sm_120 커널이 없어서 파일럿 26개가 모두 실패했다. 그래서 INUGPU가 cu128 이미지를 만들었다(`aws/IMAGE_INFO_cu128.json`).
- gr6은 g6와 같은 L4다. reserved가 376/382 조건에서 g6와 바이트까지 같다(나머지 6개는 실행 간 변동). outside도 같은 232 MiB다.
- 아직 없는 것: g7e(RTX PRO 6000, 미측정)
- 폴더 구성
  - `runs/<label>/<condition_id>/measure.json`: 조건별 원본
    - 기준값: `summary.first_measure_peak_reserved`
    - NVML: `summary.max_nvml_process_peak`
    - 환경: `environment.*`
  - `logs/main395_*_summary.md`: 조건별 요약표(MiB)
  - `logs/RUN_ON_AWS_*.txt`, `pilot_*.txt`, `nvidia-smi_final_*.txt`, `cache_verify_*.txt`
  - `MANIFEST_SHA256SUMS`: 실행 패키지(코드)의 체크섬이다. 결과 파일 목록이 아니다. 코드는 이 폴더에 없어서 대조하면 "읽을 수 없음"이 나오는 게 정상이다.
- 올리기 전에 지운 것
  - 로그의 EC2 인스턴스 ID는 `i-REDACTED`로 바꿨다.
  - 파일 이름의 내부 호스트명(`ip-…`)은 `host-redacted`로 바꿨다.
  - 측정값은 바꾸지 않았다.
- 분석: `prediction/local-cpu-vm/03-fp16-cpu-prediction/results/reserved_v3/aws_gpu_check/README_KO.md`
  - 내림차순 주소 규칙 간접 확인
  - cuBLAS 작업 공간 8,519,680 B 확인
  - 실효 용량 계산: T4 15,500,771,328 / A10G 23,363,977,216 / L4 23,417,978,880 / L40S 47,132,966,912 / RTX PRO 4500 33,354,285,056 B

## 2. CPU 실제 FP16 예측 — `cpu_fp16_real_r7i/`

- 무엇: INUOS의 CPU 예측(FP32 실행 + FP16 크기 환산)을 FP16 명령이 있는 CPU에서 **실제 FP16 실행**으로 다시 했다.
- 인스턴스: r7i.8xlarge (Sapphire Rapids, 32 vCPU, 247 GiB). r7i.16xlarge를 계획했지만 계정의 vCPU 한도 32 때문에 바꿨다.
- 실행: 2026-11-17 16:08 ~ 11-18 02:21 KST (10.2시간)
- INUOS와 같은 것
  - 이미지(ID `sha256:875290b5…`)
  - 작업자 `cpu_worker2`
  - 조건표 395개
  - 모델 revision과 파일 해시
  - 조건별 torch 스레드 수
- 다른 것
  - `--dtype-mode fp16_real`: 환산 없이 실제 FP16으로 실행하고, 같은 실행에서 첫 GEMM 위치를 기록한다.
  - 병렬 배치: 스레드 수는 그대로 두고 배정 코어만 줄였다. 요청 목록이 코어 수와 무관함을 INUOS에서 확인했다.
- 코드: `prediction/local-cpu-vm/03-fp16-cpu-prediction/tools/run_aws.py`, `aws/{setup.sh, run.sh, in_image.sh, download_models.py, model_files.json, inuos_case_stats.json, README_AWS_KO.md}`
- 결과
  - 395/395 ok
    - INUOS에서 CPU 메모리 부족이던 9개도 성공했다.
    - 1단계에서 메모리 부족이던 t5-3b 3개는 2단계(96 GiB)에서 성공했다. 첫 시도 기록은 `cases/*.attempt-*`에 남아 있다.
  - v3 재생: `results/reserved_v3/aws_fp16_l256/predictions.jsonl` 395줄, 오류 0
    - RTX 3090 용량 기준이다. R_inf는 GPU와 무관하다.
  - 스케줄러 기록: `runs/aws_fp16_l256/scheduler_events.jsonl`
- INUOS 방식과의 비교: `docs/ai-context/REPORT_CONTEXT_KO.md` §8.4
  - LLM은 같은 수준이다.
  - T5는 약간 나쁘다.
  - CNN은 CPU FP16 oneDNN 버퍼 때문에 과대예측이 더 크다.

## 3. 재현

- GPU: `measurement/local-gpu-vm/05-fp16-vram/aws/`의 패키지와 안내서를 쓴다.
- CPU: `prediction/local-cpu-vm/03-fp16-cpu-prediction/aws/README_AWS_KO.md`
  - 번들, `setup.sh`, `run.sh` 순서다.
  - 이미지는 `docs/image/export_image.sh`로 다시 만든다.

## 4. 레포에 넣지 않은 것

- CPU 결과의 요청 목록 `invocation_fp16_requests.npz` 396개(합계 433 MB). 크기 때문에 뺐다.
  - 원본은 INUOS `~/aws-result/aws_fp16_l256_results.tar.gz`에 있다(306 MB, sha256 `114d14404d26e594400d4a300d2e2811707b5578c71e3405473db1501afe5d2b`). AWS에서 만든 파일과 해시가 같은 것을 확인했다.
  - 다시 재생하려면(예: CPU 버퍼 제거 실험) 이 압축 파일이 필요하다.
- 원본 profiler trace.json은 작업자가 요청 목록을 뽑은 뒤 지우므로 처음부터 없다(INUOS도 같음).

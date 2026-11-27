# RTX 3090 FP16 VRAM 파일럿 — attention eager 고정 vs transformers 기본값 통합 보고서

- 작성: inugpu, 2026-11-11
- 대상: 같은 26개 파일럿 조건을 두 attention 설정으로 측정한 GPU 결과 + GPU allocator 요청의 xMem 재생 결과
- 목적: (1) 두 설정 비교, (2) 같은 모델·입력을 CPU(xMem)에서 돌렸을 때 어디서 차이가 나는지 분석할 수 있도록 GPU 쪽 근거 제공
- 이 보고서는 관측값만 정리. 원인 판정은 하지 않음 (§7은 CPU 분석 시 확인할 항목 목록)

## 1. 두 실행의 차이

| 항목 | 실행 A: eager 고정 (v1) | 실행 B: 기본값 (v2) |
|---|---|---|
| 로딩 인자 | `attn_implementation="eager"` 명시 | 인자 생략 → transformers 4.52.4 기본값 |
| 실제 attention | 26개 전부 eager | sdpa: opt-125m, TinyLlama, Qwen2.5, gpt2·medium·xl, ViT (7개 모델 14조건) / eager: T5 3종(sdpa 미지원), ResNet·MobileNet(attention 없음) (12조건) |
| GPU sdpa 커널 | 해당 없음 | 14조건 모두 FlashAttention (`pytorch_flash::flash_fwd_kernel`), prefill·decode 모두 (§4) |
| 결과 경로 | `runs/rtx3090/` | `runs/rtx3090__attn-default/` |
| worker sha256 | measure·validate: `1e86ee7e…` / gpt2 b16 trace만 `d06f0f15…` | `d06f0f15…` (`--attn` 옵션 추가판) |
| 실행 시각 | measure 14:32:34→14:53:57 (21분 23초), validate 5분 12초, trace 5개 1분 20초 | measure 15:36:13→15:57:15 (21분 2초), validate 5분 13초, trace 6개 1분 47초 |

- 공통: RTX 3090 24GiB (cc 8.6), driver 570.211.01, 이미지 `cloudgpufit-bench:cu124-v1` (sha256:457da255…fb032), python 3.11.11, torch 2.6.0+cu124, cuDNN 90100, transformers 4.52.4
- 공통: CUDA_MODULE_LOADING=LAZY(env·드라이버 조회), allocator 설정 없음(native, expandable_segments off), TF32 off, cudnn.benchmark off, FP16, seed 20261006
- 공통 입력: 같은 조건이면 두 실행의 입력 텐서 sha256 동일 (CSV `*_input_sha256`)
- 공통 프로토콜: 조건마다 새 프로세스 → 로드 1회 → warmup 3 → measure 5 (allocator 캐시 유지). measure 직전 시작값 기록 후 peak 통계만 초기화. NVML은 같은 PID, 10ms 간격
- 생성 설정: greedy, beam 1, KV cache, `min_new_tokens = max_new_tokens`, hub generation_config 대신 `GenerationConfig.from_model_config(config)` 사용

## 2. 상태 요약

- measure: A 26/26 ok, B 26/26 ok. 생성 토큰 수 전부 요청값과 일치. OOM 0, alloc retry 0
- 출력 검증(validate, 별도 프로세스, 모든 forward의 원시 logits 유한성): A 24/26 ok, B 26/26 ok
  - A의 실패 2개: Qwen2.5-1.5B-Instruct 두 조건, 첫 forward부터 nonfinite (§5)
- 52개 measure 전부: 5회 피크 reserved가 모두 같고, 1회차 시작 reserved = 1회차 피크 reserved
- trace 12개(A 6, B 6) 전부: 기록 시작 시 allocator 빈 상태, alloc 이벤트 수 = 누적 카운터 (잘림 없음)

## 3. 조건별 GPU 측정값 (MiB, 1회차 = CPU 비교 기준)

- 차이 = 기본값 − eager
- 출력 토큰 동일 = 1회차 출력 token id sha256 일치 여부 (attention 구현이 바뀌면 수치가 달라질 수 있음)
- 전체 필드(5회 최댓값, 시작 allocated, 경과시간 등): `results/pilot_rtx3090_eager_vs_default.csv`

| 조건 | attention (eager/기본) | 1회차 피크 reserved eager | 기본 | 차이 | 1회차 피크 allocated eager | 기본 | 차이 | 로드 후 reserved eager/기본 | NVML 최대 eager/기본 | 검증 eager/기본 | 출력 토큰 동일 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| opt-125m__b8__in128__new128 | eager/sdpa | 418.0 | 418.0 | 0.0 | 388.1 | 388.1 | 0.0 | 316.0/316.0 | 730.0/730.0 | ok/ok | 아니오 |
| opt-125m__b16__in512__new512 | eager/sdpa | 1940.0 | 1524.0 | -416.0 | 1334.3 | 1334.0 | -0.3 | 316.0/316.0 | 2252.0/1836.0 | ok/ok | 예 |
| tinyllama-1.1b-chat__b8__in128__new128 | eager/sdpa | 2348.0 | 2330.0 | -18.0 | 2189.8 | 2182.5 | -7.3 | 2234.0/2234.0 | 2660.0/2642.0 | ok/ok | 예 |
| tinyllama-1.1b-chat__b16__in512__new512 | eager/sdpa | 3986.0 | 2926.0 | -1060.0 | 3510.6 | 2678.6 | -832.0 | 2234.0/2234.0 | 4298.0/3238.0 | ok/ok | 예 |
| qwen2.5-1.5b-instruct__b8__in128__new128 | eager/sdpa | 3512.0 | 3510.0 | -2.0 | 3051.9 | 3048.6 | -3.3 | 3448.0/3448.0 | 3824.0/3822.0 | nonfinite_output/ok | 아니오 |
| qwen2.5-1.5b-instruct__b16__in512__new512 | eager/sdpa | 4200.0 | 3868.0 | -332.0 | 3801.3 | 3697.3 | -104.0 | 3448.0/3448.0 | 4512.0/4180.0 | nonfinite_output/ok | 아니오 |
| gpt2__b8__in128__new128 | eager/sdpa | 476.0 | 476.0 | 0.0 | 418.4 | 417.7 | -0.7 | 474.0/474.0 | 788.0/788.0 | ok/ok | 아니오 |
| gpt2__b16__in512__new512 | eager/sdpa | 1982.0 | 1838.0 | -144.0 | 1489.9 | 1490.8 | 0.9 | 474.0/474.0 | 2294.0/2150.0 | ok/ok | 아니오 |
| gpt2-medium__b8__in128__new128 | eager/sdpa | 1004.0 | 1004.0 | 0.0 | 959.3 | 958.8 | -0.5 | 1002.0/1002.0 | 1316.0/1316.0 | ok/ok | 아니오 |
| gpt2-medium__b16__in512__new512 | eager/sdpa | 3518.0 | 3390.0 | -128.0 | 2663.2 | 2663.2 | 0.0 | 1002.0/1002.0 | 3830.0/3702.0 | ok/ok | 아니오 |
| gpt2-xl__b8__in128__new128 | eager/sdpa | 3784.0 | 3812.0 | 28.0 | 3636.7 | 3636.2 | -0.5 | 3486.0/3486.0 | 4096.0/4124.0 | ok/ok | 예 |
| gpt2-xl__b16__in512__new512 | eager/sdpa | 9374.0 | 9174.0 | -200.0 | 7885.6 | 7885.1 | -0.5 | 3486.0/3486.0 | 9686.0/9486.0 | ok/ok | 예 |
| resnet-18__b8__r224 | eager/eager | 80.0 | 80.0 | 0.0 | 61.2 | 61.2 | 0.0 | 48.0/48.0 | 394.0/394.0 | ok/ok | 예 |
| resnet-18__b16__r512 | eager/eager | 458.0 | 458.0 | 0.0 | 343.7 | 343.7 | 0.0 | 48.0/48.0 | 772.0/772.0 | ok/ok | 예 |
| resnet-50__b8__r224 | eager/eager | 132.0 | 132.0 | 0.0 | 107.4 | 107.4 | 0.0 | 82.0/82.0 | 446.0/446.0 | ok/ok | 예 |
| resnet-50__b16__r512 | eager/eager | 812.0 | 812.0 | 0.0 | 563.1 | 563.1 | 0.0 | 82.0/82.0 | 1128.0/1128.0 | ok/ok | 예 |
| mobilenet-v2-1.0-224__b8__r224 | eager/eager | 88.0 | 88.0 | 0.0 | 62.3 | 62.3 | 0.0 | 26.0/26.0 | 402.0/402.0 | ok/ok | 예 |
| mobilenet-v2-1.0-224__b16__r512 | eager/eager | 848.0 | 848.0 | 0.0 | 505.0 | 505.0 | 0.0 | 26.0/26.0 | 1162.0/1162.0 | ok/ok | 예 |
| vit-large-patch16-224__b8__r224 | eager/sdpa | 682.0 | 656.0 | -26.0 | 649.2 | 632.9 | -16.3 | 620.0/620.0 | 996.0/970.0 | ok/ok | 아니오 |
| vit-large-patch16-224__b16__r224 | eager/sdpa | 788.0 | 738.0 | -50.0 | 707.0 | 674.2 | -32.8 | 620.0/620.0 | 1102.0/1052.0 | ok/ok | 아니오 |
| t5-small__b8__in128__new128 | eager/eager | 304.0 | 304.0 | 0.0 | 154.2 | 154.2 | 0.0 | 272.0/272.0 | 616.0/616.0 | ok/ok | 예 |
| t5-small__b16__in512__new512 | eager/eager | 738.0 | 738.0 | 0.0 | 555.7 | 555.7 | 0.0 | 272.0/272.0 | 1050.0/1050.0 | ok/ok | 예 |
| t5-base__b8__in128__new128 | eager/eager | 710.0 | 710.0 | 0.0 | 515.7 | 515.7 | 0.0 | 678.0/678.0 | 1022.0/1022.0 | ok/ok | 예 |
| t5-base__b16__in512__new512 | eager/eager | 1474.0 | 1474.0 | 0.0 | 1082.3 | 1082.3 | 0.0 | 678.0/678.0 | 1786.0/1786.0 | ok/ok | 예 |
| t5-3b__b8__in128__new128 | eager/eager | 7220.0 | 7220.0 | 0.0 | 6228.5 | 6228.5 | 0.0 | 7166.0/7166.0 | 7532.0/7532.0 | ok/ok | 예 |
| t5-3b__b16__in512__new512 | eager/eager | 12996.0 | 12996.0 | 0.0 | 11678.9 | 11678.9 | 0.0 | 7166.0/7166.0 | 13308.0/13308.0 | ok/ok | 예 |

- 관측 메모
  - sdpa로 바뀐 14조건 중 피크 reserved가 줄어든 조건 10개, 같은 조건 3개, 늘어난 조건 1개 (gpt2-xl b8/128 +28MiB, allocated는 −0.5MiB)
  - 가장 큰 차이: TinyLlama b16/512/512 reserved −1060MiB, allocated −832MiB
  - attention 설정이 같게 풀린 12조건(T5·ResNet·MobileNet)은 두 실행 값이 모두 같음 → 실행 간 재현 확인용
  - 로드 직후 reserved는 26조건 모두 두 실행이 같음

## 4. GPU가 실제로 쓴 attention 커널 (실행 B, 별도 진단 프로세스)

- 방법: `torch.profiler`로 generate 1토큰(prefill만)과 2토큰(prefill+decode 1스텝)을 측정, 차이로 decode 커널 분리. ViT는 forward 1회
- 결과: sdpa 14조건 모두 prefill·decode 둘 다 FlashAttention 커널만 사용, 레이어당 1회 (memory-efficient·math·cudnn sdpa 커널 미관측)
  - 예: Qwen2.5 prefill `Flash_fwd_kernel_traits<128, 64, 64, …>`, decode `Flash_fwd_kernel_traits<128, 128, 32, …>` (head_dim 128). gpt2·opt·TinyLlama·ViT는 `<64, 128, 128, …>`
- 활성화된 sdpa backend 플래그: flash/mem_efficient/math/cudnn 모두 True (선택은 PyTorch가 함)
- 원본: `results/attention_kernels_rtx3090_default.json` (조건별 전체 커널 이름·횟수 포함)

## 5. Qwen2.5-1.5B-Instruct FP16 eager nonfinite (실행 A)

- measure는 정상 완료(VRAM 값 기록됨), validate에서만 실패
- 변형 비교(B8/L128 prefill 1회): FP16+eager만 실패. FP16+sdpa, BF16, FP32(eager/sdpa) 모두 유한. 입력 ID 범위(config vocab 151936 vs tokenizer 151665) 무관
- 첫 비유한 지점: `model.layers.0.self_attn` (o_proj 출력)
- 관측값: transformers 4.52.4 eager는 `matmul(Q, Kᵀ) * scaling` 순서. 0번 레이어 스케일 전 |QKᵀ| 최대 127,295 (B8/L128), 82,706 (B1/L32) > FP16 최대 65,504. 스케일 후 최대 11,251 / 7,310
- 실행 B(sdpa, FlashAttention)에서는 두 조건 모두 validate ok
- 원본: `results/diag_qwen/diag_qwen_nonfinite.json`, `results/diag_qwen/diag_qwen_qk.json`

## 6. GPU allocator 요청의 xMem 재생 (CPU 시뮬레이터 정책 검증)

- 방법: trace 프로세스(로드 → warmup 3 → measure 1)의 GPU alloc(원요청 크기)/free_completed 이벤트를 순서대로 xMem `CachingAllocator`(커밋 1ee4bd0, allocator.py sha256 `7ac16de1…` = lock 파일과 동일)에 재생
- GPU segment 이벤트는 재생하지 않고 xMem이 스스로 segment를 결정 → 개수·reserved를 GPU와 비교
- 경계: worker가 synchronize 후 기록한 누적 카운터로 이벤트 위치를 찾음

| 조건 | 실행 설정(실제 attention) | 경계 위치 확인 | 짝 없는 free | GPU/xMem segment 수 | 로드 후 reserved GPU/xMem | m1 시작 reserved GPU/xMem | m1 피크 reserved GPU/xMem | m1 피크 allocated GPU/xMem | m1 구간 alloc 요청 수 | 전체 alloc 이벤트 |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt2__b8__in128__new128 | eager 고정(eager) | 10/10 | 0 | 11/11 | 474.0/474.0 | 476.0/476.0 | 476.0/476.0 | 418.4/418.4 | 60960 | 244162 |
| gpt2__b8__in128__new128 | 기본값(sdpa) | 10/10 | 0 | 11/11 | 474.0/474.0 | 476.0/476.0 | 476.0/476.0 | 417.7/417.7 | 51103 | 204734 |
| opt-125m__b8__in128__new128 | eager 고정(eager) | 10/10 | 0 | 4/4 | 316.0/316.0 | 418.0/418.0 | 418.0/418.0 | 388.1/388.1 | 53420 | 213879 |
| opt-125m__b8__in128__new128 | 기본값(sdpa) | 10/10 | 0 | 4/4 | 316.0/316.0 | 418.0/418.0 | 418.0/418.0 | 388.1/388.1 | 46635 | 186739 |
| qwen2.5-1.5b-instruct__b8__in128__new128 | eager 고정(eager) | 10/10 | 0 | 62/62 | 3448.0/3448.0 | 3512.0/3512.0 | 3512.0/3512.0 | 3051.9/3051.9 | 188307 | 753908 |
| qwen2.5-1.5b-instruct__b8__in128__new128 | 기본값(sdpa) | 10/10 | 0 | 61/61 | 3448.0/3448.0 | 3510.0/3510.0 | 3510.0/3510.0 | 3048.6/3048.6 | 173331 | 694004 |
| t5-small__b8__in128__new128 | eager 고정(eager) | 10/10 | 0 | 36/36 | 272.0/272.0 | 304.0/304.0 | 304.0/304.0 | 154.2/154.2 | 68215 | 273124 |
| t5-small__b8__in128__new128 | 기본값(eager) | 10/10 | 0 | 36/36 | 272.0/272.0 | 304.0/304.0 | 304.0/304.0 | 154.2/154.2 | 68215 | 273124 |
| resnet-50__b8__r224 | eager 고정(eager) | 10/10 | 0 | 12/12 | 82.0/82.0 | 132.0/132.0 | 132.0/132.0 | 107.4/107.4 | 292 | 1757 |
| resnet-50__b8__r224 | 기본값(eager) | 10/10 | 0 | 12/12 | 82.0/82.0 | 132.0/132.0 | 132.0/132.0 | 107.4/107.4 | 292 | 1757 |
| gpt2__b16__in512__new512 | eager 고정(eager) | 10/10 | 0 | 26/26 | 474.0/474.0 | 1982.0/1982.0 | 1982.0/1982.0 | 1489.9/1489.9 | 243744 | 975298 |
| gpt2__b16__in512__new512 | 기본값(sdpa) | 10/10 | 0 | 24/24 | 474.0/474.0 | 1838.0/1838.0 | 1838.0/1838.0 | 1490.8/1490.8 | 204319 | 817598 |

- 관측 메모
  - 12개 전부: 모든 경계(10/10)에서 xMem의 reserved·allocated(시작·피크·끝)가 GPU와 같음, segment 개수 같음, 짝 없는 free 0
  - 같은 조건에서 sdpa는 eager보다 alloc 요청 수가 적음 (예: gpt2 b16/512/512 measure 1회 243,744 → 204,319)
  - gpt2 b16/512/512: sdpa 피크 reserved 1838 vs eager 1982, segment 24 vs 26
- 원본: `results/xmem_replay_rtx3090/{eager,default}/<condition>.json`, 표 `results/xmem_replay_rtx3090/replay_table.md`
- 재생 도구: `tools/xmem_replay.py` (xmem-gpu 컨테이너, `CUDA_VISIBLE_DEVICES=` 로 GPU 미사용)

## 7. CPU(xMem) 결과와 맞춰 볼 때 확인할 항목

- 같은 조건 ID·입력 sha·생성 설정인지 (`config/conditions.json`, `models.json`, 각 measure.json의 `inputs`, `generation_config`)
- attention 구현이 같은지: CPU 실행의 실제 `_attn_implementation`과, sdpa라면 CPU에서 선택된 aten 연산
  - GPU 실행 B의 sdpa는 FlashAttention만 사용(§4) → L×L attention 행렬이 생기지 않음
  - GPU 실행 A(eager)는 L×L attention 가중치를 생성
- 시작 상태: GPU는 52개 measure 모두 1회차 시작 reserved = 1회차 피크 → 피크는 warmup 3회가 만든 캐시 상태에서 결정됨. CPU도 warmup 3 + measure 1을 캐시 유지로 재생해야 같은 기준
- 요청 단위 비교: trace 12개(§6)는 GPU 요청 크기·순서·해제 시점 전체를 담고 있음 → CPU trace와 같은 구간(`measure_0_begin`~`_end`)끼리 요청 수·크기 분포를 대조하면 GPU 전용 요청(cuBLAS workspace 등)·커널별 중간 텐서 차이를 찾을 수 있음
- xMem allocator 정책 자체는 GPU 요청을 그대로 넣었을 때 12개 모두 GPU와 같은 결과(§6) → CPU 예측과 GPU 실측이 다르면 요청 목록(입력 trace) 쪽 차이부터 확인
- Qwen2.5 FP16 eager는 출력 값이 NaN이지만 요청 크기는 shape로 결정됨. CPU 쪽에서 같은 조건 비교 시 출력 검증 결과를 별도로 표시
- NVML 값은 allocator 밖 메모리(CUDA context 등)를 포함하므로 reserved와 직접 비교하지 않음

## 8. 파일 위치 (모두 `/home/inugpu/cgf-fp16-vram-20261111/` 기준)

| 내용 | 경로 |
|---|---|
| 이 보고서 | `results/REPORT_rtx3090_pilot_eager_vs_default_KO.md` |
| 통합 CSV (조건 × 두 실행) | `results/pilot_rtx3090_eager_vs_default.csv` |
| 실행 A 원본 | `runs/rtx3090/<condition_id>/{measure,validate,trace}.json`, `trace.snapshot.pickle.gz`, `*.launch.json`, 로그 |
| 실행 B 원본 | `runs/rtx3090__attn-default/<condition_id>/…` (같은 구성) |
| 진행 기록 | `runs/<label>/progress.jsonl`, `sessions.jsonl` |
| 실행 로그 | `logs/pilot.txt` (A), `logs/pilot_v2_attn_default.txt` (B) |
| 코드 해시 | `logs/code_sha256_pilot.txt` (A), `logs/code_sha256_pilot_v2.txt` (B) |
| xMem 재생 | `results/xmem_replay_rtx3090/` |
| attention 커널 진단 | `results/attention_kernels_rtx3090_default.json` |
| Qwen 진단 | `results/diag_qwen/` |
| 조건표·설정 | `config/` (680개 조건, 13개 모델 revision, 캐시 manifest) |
| 코드 | `src/vram_worker.py`, `src/run_conditions.py`, `tools/` |
| 실험 설명 | `README_KO.md` |

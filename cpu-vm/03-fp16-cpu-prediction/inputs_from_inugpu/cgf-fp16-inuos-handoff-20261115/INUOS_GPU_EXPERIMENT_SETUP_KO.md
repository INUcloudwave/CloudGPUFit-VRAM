# CloudGPUFit FP16 VRAM — GPU 실험 구성 상세 (INUOS CPU 예측 정렬용, 2026-11-15)

- 작성: INUGPU (RTX 3090 VM)
- 읽는 쪽: INUOS의 CPU 예측 담당 에이전트
- 이 문서 위치(INUOS): `/home/inuos/cgf-fp16-inuos-handoff-20261115/INUOS_GPU_EXPERIMENT_SETUP_KO.md`
- 이전 문서: `/home/inuos/cgf-fp16-inuos-handoff-20261112/INUOS_GPU_MEASUREMENT_PROTOCOL_KO.md`
  - 측정 절차는 그때와 같습니다. 이번에 바뀐 것은 **조건표(1024 → 256)** 하나입니다.
- 표기 원칙
  - 코드나 실행 기록으로 확인한 내용만 사실로 적었습니다.
  - 확인하지 않은 내용은 **미확인**으로 표시했습니다.

---

## 0. INUOS 에이전트에게 (작업 프롬프트)

> 아래 디렉터리의 GPU 실험 구성을 직접 읽고, CPU 예측 실행을 GPU 실측과 같은 조건·순서·기준값으로 맞춰라.
>
> - 기준 디렉터리: `/home/inuos/cgf-fp16-inuos-handoff-20261115/`
>   - `SHA256SUMS`로 무결성부터 확인하라: `cd` 후 `sha256sum -c SHA256SUMS`
> - GPU 실측의 정답 코드
>   - `src/vram_worker.py` (sha256 `d06f0f15…`)
>   - `src/run_conditions.py` (sha256 `77e3f4a9…`)
>   - 이 문서의 줄번호는 이 사본 기준이다.
> - 새 조건표: `config/conditions_main395_l256.json` (sha256 `d4a9b09c…`, 395개)
>   - 현재 CPU 쪽 `manifest/cases395.jsonl`은 이전 조건표(1024 포함) 기준일 가능성이 높다. 차이를 확인하라.
>
> 해야 할 일
> 1. 비교표를 만들어라: §3~§6의 각 항목(조건, 환경, 설정, 측정 순서, 기준값, 실패 판정)마다 CPU 예측 코드에서 대응하는 줄을 찾고, 일치 / 불일치 / 구조적으로 불가를 표시한다. 산출물: `handoff_codex/gpu_cpu_alignment.md` 또는 `.json`.
> 2. 불일치 중 CPU 쪽에서 맞출 수 있는 것은 수정 계획을 먼저 적어라. 수정 전에 사용자 확인을 받아라.
>    - 맞출 수 없는 것(FP32 실행, GPU 전용 workspace 등)은 이유와 해당 condition_id를 기록하라.
> 3. 조건표를 `conditions_main395_l256.json`으로 바꿀 때 지킬 것:
>    - condition_id를 그대로 써라.
>    - 겹치는 245개와 새로 생긴 150개(256 포함)를 구분해 기록하라. §3.3 참고.
> 4. 비교 기준값을 맞춰라.
>    - 1차 기준: `summary.first_measure_peak_reserved`
>    - 보조 기준: `first_measure_peak_allocated`, `after_load reserved`, `first_measure_start_reserved`
>    - 단위는 byte다.
> 5. 결과를 대조할 때 `results/rtx3090_main395_L1024/gpu_rtx3090_main395_L1024_per_condition.csv`(이전 1024 조건표 실측)를 써라.
>    - 겹치는 245개는 측정 설정이 같으므로 새 실행(L256)에서도 값이 같아야 한다.
>    - 다르면 기록하라.
>
> 제약
> - 생산 예측 로직과 과거 결과는 보존하라.
> - 실패 조건을 몰래 낮은 배치나 다른 dtype으로 바꾸지 마라.
> - 모델 ID나 후보를 자동으로 바꾸지 마라.
> - 공유 blob이나 활성 컨테이너의 파일을 직접 삭제하지 마라.
> - 키나 토큰 값을 출력하거나 기록하지 마라.

---

## 1. 무엇이 바뀌었나

| 항목 | 이전 (`conditions_main395.json`) | 지금 (`conditions_main395_l256.json`) |
|---|---|---|
| OPT-125M / TinyLlama-1.1B-Chat / Qwen2.5-1.5B-Instruct 입력×생성 | {128, 512, 1024}² | **{128, 256, 512}²** |
| T5-small / base / 3B 인코더 입력×디코더 생성 | {128, 512, 1024}² | **{128, 256, 512}²** |
| GPT-2 / Medium / XL | {128, 512}² | 변경 없음 |
| ResNet-18/50, MobileNet V2 | 해상도 128/224/256/512 | 변경 없음 |
| ViT Large | 224 | 변경 없음 |
| 배치 | 1/8/16/32/128 | 변경 없음 |
| 총 조건 수 | 395 | 395 |
| sha256 | `2634bda7…fe258` | `d4a9b09caff3064c00c04358bf2e462996bced23de93da98652bd1aba04b7289` |

- 바꾼 이유: 1024 길이 조건이 너무 오래 걸려서입니다. 사용자 결정입니다.
- 모델 ID, revision, 입력 생성·생성 설정 규칙(`config/models.json`, sha256 `7b11e3e3…6589`)은 바꾸지 않았습니다.
- 측정 코드(worker, runner)도 바꾸지 않았습니다. 이전 실행과 sha256이 같습니다.
- 이전 조건표와 이전 결과는 그대로 보존되어 있습니다.

---

## 2. 파일 위치

### 2.1 INUOS에 복사된 것 (`/home/inuos/cgf-fp16-inuos-handoff-20261115/`)

| 경로 | 내용 |
|---|---|
| `INUOS_GPU_EXPERIMENT_SETUP_KO.md` | 이 문서 |
| `config/conditions_main395_l256.json`, `.csv` | **새 조건표** |
| `config/generate_main_l256.py` | 새 조건표 생성기 (assert 포함, 재현 가능) |
| `config/conditions_main395.json`, `.csv`, `generate_main.py` | 이전 조건표와 생성기 |
| `config/conditions.json`, `generate_conditions.py` | 680 공통표와 생성기 (조건 객체 형식의 원본) |
| `config/models.json` | 모델 ID, revision, loader, 공통 규칙 문구 |
| `config/pilot.json` | 파일럿 26개 id |
| `src/vram_worker.py` | GPU worker (측정의 정답 코드) |
| `src/run_conditions.py` | GPU runner (조건마다 컨테이너 1회) |
| `tools/summarize_runs.py` | `summary.csv` 생성 |
| `tools/export_for_inuos.py` | 조건별 상세 CSV(byte 단위, 전체 입력 해시) 생성 |
| `tools/check_pilot.py` | 파일럿 PASS 판정 |
| `run_main395_l256_vm.sh` | **지금 실행 중인** 본실험 스크립트 |
| `run_main395_vm.sh` | 이전 본실험 스크립트 |
| `results/rtx3090_main395_L1024/gpu_rtx3090_main395_L1024_per_condition.csv` | 이전 395개(1024) 조건별 상세값. §7 참고 |
| `results/rtx3090_main395_L1024/summary.csv`, `main395_rtx3090__main395_summary.md` | 이전 실행 요약 (MiB 단위) |
| `results/rtx3090_main395_L1024/runs_rtx3090__main395/<cid>/{measure,validate}.json` | 이전 실행 원본 결과 395×2 |
| `SHA256SUMS` | 체크섬 |

- 파일럿 trace 6개, 커널 진단, xMem 재생표는 이전 디렉터리에 있습니다.
  - 위치: `/home/inuos/cgf-fp16-inuos-handoff-20261112/reference/`
  - 이전 디렉터리는 그대로 둡니다.

### 2.2 INUGPU 원본 위치 (참고. INUOS에서 INUGPU 접속이 설정되면 직접 읽을 수 있음)

- 실험 루트: `/home/inugpu/cgf-fp16-vram-20261111/`
- 새 실행 결과: `runs/rtx3090__main395_l256/<cid>/{measure,validate}.json`, `progress.jsonl`, `sessions.jsonl`
  - 종료 후 `runs/rtx3090__main395_l256/summary.csv`가 생깁니다.
- 새 실행 로그: `logs/main395_l256_rtx3090.txt`
  - 종료 후 요약 md: `logs/main395_rtx3090__main395_l256_summary.md`
- 이전 실행 결과: `runs/rtx3090__main395/`

---

## 3. 조건표

### 3.1 구성 (395개)

| 그룹 | 모델 (HF id) | 축 | 후보 | 조건 수 |
|---|---|---|---|---|
| 디코더 LLM | facebook/opt-125m, TinyLlama/TinyLlama-1.1B-Chat-v1.0, Qwen/Qwen2.5-1.5B-Instruct | input_len × max_new_tokens | {128,256,512}² | 3×5×9 = 135 |
| GPT-2 | openai-community/gpt2, gpt2-medium, gpt2-xl | input_len × max_new_tokens | {128,512}² | 3×5×4 = 60 |
| 인코더-디코더 | google-t5/t5-small, t5-base, t5-3b | 인코더 input_len × 디코더 max_new_tokens | {128,256,512}² | 3×5×9 = 135 |
| CNN | microsoft/resnet-18, resnet-50, google/mobilenet_v2_1.0_224 | resolution | 128/224/256/512 | 3×5×4 = 60 |
| ViT | google/vit-large-patch16-224 | resolution | 224 | 5 |

- 배치는 모든 그룹에서 1/8/16/32/128입니다.
- condition_id 형식: `<short>__b<B>__in<L>__new<N>`, 비전은 `<short>__b<B>__r<R>`
- 조건 객체 형식(키와 순서)은 680표와 같습니다.
  - 텍스트: `{condition_id, model, workload_type, batch_size, input_len, max_new_tokens}`
  - 비전: `{condition_id, model, workload_type, batch_size, resolution}`
- 나열 순서는 `models.json`의 모델 순서 → 배치 → L → N(또는 R)입니다. runner는 이 순서대로 실행합니다.

### 3.2 생성 검증 (`generate_main_l256.py`의 assert)

- 개수 395, 중복 없음
- 모델별 개수: 텍스트 9종 각 45 또는 20, CNN 각 20, ViT 5
- 1024가 하나도 없음
- 680표에 있는 id는 객체 전체(키 순서 포함)가 680표와 같음
- 파일럿 26개가 모두 포함됨

### 3.3 이전 조건표와의 관계

- 겹치는 id 245개: GPT-2 60, CNN 60, ViT 5, 그리고 텍스트 6종의 {128,512}² 부분 120
  - 측정 코드와 설정이 같으므로, 같은 id면 같은 값이 나와야 합니다.
- 새 id 150개: 텍스트 6종에서 L 또는 N에 256이 들어간 조건 (6모델 × 5배치 × 5조합)
- 빠진 id 150개: L 또는 N이 1024인 조건
  - 이 조건들의 이전 실측값은 `results/rtx3090_main395_L1024/`에 남아 있습니다.

---

## 4. 실행 환경 (GPU 쪽, 이번 실행 `sessions.jsonl`에서 확인)

| 항목 | 값 |
|---|---|
| label | `rtx3090__main395_l256` |
| 시작 | 2026-11-15 17:00:58 KST |
| GPU | NVIDIA GeForce RTX 3090, 24576 MiB, 드라이버 570.211.01, UUID `GPU-17eb7299-…` |
| `cuda_total_memory` | **25,306,529,792 B** (OOM 판정 기준 용량) |
| Docker 이미지 | `cloudgpufit-bench:cu124-v1`, ID `sha256:457da255e6e2de7ee05ff7814674c90abcf6d1ed04e73df3cc4eebcdb31fb032` |
| 소프트웨어 | python 3.11.11, torch 2.6.0+cu124, cuDNN 90100, transformers 4.52.4 |
| worker sha256 | `d06f0f15982c811c4bc6a24028f71728e461c6ee3651bcfd520ba49b0d675883` |
| runner sha256 | `77e3f4a94784ad7452afa3705d4c8fa48ee5107a9c26fd7373c275ff975840f3` |
| 조건표 sha256 | `d4a9b09caff3064c00c04358bf2e462996bced23de93da98652bd1aba04b7289` |
| 시드 | 20261006 |
| torch 스레드 | `min(8, os.cpu_count())`, INUGPU에서는 4 |

---

## 5. 실행 구조와 순서

### 5.1 전체 순서 (`run_main395_l256_vm.sh`)

1. measure 395개를 조건표 순서대로 실행합니다.
2. 그다음 validate 395개를 실행합니다.
3. 마지막으로 `summarize_runs.py`가 `summary.csv`를 만듭니다.

- 다시 실행하면 이어서 진행합니다. 결과 파일이 이미 있으면 건너뜁니다(`run_conditions.py:116`).

### 5.2 조건 1개 = 새 컨테이너 1개 (`run_conditions.py:130`)

```
docker run --rm --pull=never --network none --pid=host --gpus device=<GPU UUID> --user <uid>:<gid>
  -e HOME=/tmp -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e CUDA_MODULE_LOADING=LAZY
  -v src:/src:ro -v config:/config:ro -v <hf-cache>:/hf:ro -v runs/<label>/<cid>:/out
  --entrypoint python <image id> -I /src/vram_worker.py --models /config/models.json
  --conditions /config/conditions_main395_l256.json --condition <cid> --mode measure|validate --out /out/<mode>.json --attn default
```

- 프로세스 상태는 조건마다 초기화됩니다. 이전 조건의 allocator 캐시를 물려받지 않습니다.
- 순차 실행, GPU 1개입니다. 실행 전에 다른 GPU 프로세스가 있으면 중단합니다.
- 실패한 조건은 그대로 기록하고 자동 재시도하지 않습니다. 배치나 dtype을 바꿔 다시 돌리지 않습니다.

### 5.3 worker 절차 (`vram_worker.py`, 줄번호는 handoff 사본 기준)

1. 프로세스 설정 (main `:141`)
   - `torch.manual_seed(20261006)`
   - TF32 off(matmul, cudnn), `cudnn.benchmark=False`
   - `torch.cuda.init()` 후 start_state 기록
   - `PYTORCH_CUDA_ALLOC_CONF`는 설정하지 않습니다(기본 caching allocator).
2. 모델 로드
   - `reset_peak_memory_stats()` (`:239`)
   - `from_pretrained(id, revision, torch_dtype=float16, low_cpu_mem_usage=True, device_map={"":"cuda:0"}, use_safetensors=<models.json>, local_files_only=True)` (`:242`)
     - `attn_implementation` 인자는 넘기지 않습니다. 즉 transformers 기본값을 씁니다.
   - 이어서 `.to(cuda, fp16).eval()`, `output_attentions/hidden_states=False`
   - 동기화 후 `after_load` 기록
3. 입력 생성: 호스트 CPU 텐서, 토크나이저와 이미지 전처리는 쓰지 않습니다.
   - 텍스트 (`:277`)
     - `g=torch.Generator().manual_seed(20261006)`
     - `randint(3, vocab_size, (B,L), generator=g, int64)`
     - pad/eos/bos/decoder_start와 같은 값은 3으로 바꿉니다.
     - `attention_mask=ones(B,L,int64)`
   - 비전 (`:271`): `randn(B,3,R,R, generator=g, float32).to(float16)`
   - 입력 sha256은 결과의 `inputs.<name>.sha256`에 있습니다.
4. 생성 설정 (`:292`)
   - `GenerationConfig.from_model_config(config)`
   - `do_sample=False, num_beams=1, use_cache=True, min_new_tokens=max_new_tokens=N`
   - `pad_token_id=config.pad_token_id or eos`
   - hub의 generation_config.json은 쓰지 않습니다.
   - 생성 길이 확인: decoder는 `out.shape[1]-L==N`, T5는 `out.shape[1]-1==N`
5. 1회 추론 invoke (`:308`)
   - `inference_mode` 안에서 실행합니다.
   - 매 회차 입력을 `.to("cuda")`로 새로 복사합니다.
   - 텍스트는 `generate`, 비전은 `model(**x)` forward 1회
   - 끝나면 `synchronize`
   - 입력 복사본과 출력은 기록이 끝날 때까지 살아 있다가 `del`합니다.
6. **warmup 3회** (`:323`) → **measure 5회** (`:343`)
   - 각 measure 직전에 `start` 기록 후 `reset_peak_memory_stats()` (`:348`)
     - peak만 초기화하고 캐시는 유지합니다. `empty_cache()`는 어디에서도 부르지 않습니다.
   - 각 measure마다 기록: `peak`(max_memory_allocated/reserved), `end_with_output`, `released`, NVML 프로세스 사용량(10ms 샘플)
7. summary (`:379`)
   - `first_measure_peak_reserved/allocated`
   - `first_measure_start_reserved/allocated`
   - 5회 measure의 median/max 값

### 5.4 validate (별도 프로세스)

- 로드, 입력, 생성 설정은 measure와 같고, warmup 0회, measure 1회입니다.
- forward hook(`:304`)이 생성의 모든 스텝에서 원시 logits에 `isfinite().all()`을 검사합니다.
  - 결과 항목: `validation.steps_checked / all_finite / first_nonfinite_step`
- validate의 메모리 값은 측정값으로 쓰지 않습니다.
  - hook 때문에 텐서가 추가로 할당되고, warmup 없이 캐시가 비어 있는 상태에서 시작하기 때문입니다.
- 이 때문에 measure는 ok인데 validate만 `gpu_oom`인 조건이 있습니다. 이전 실행에서 10개였습니다(§7.3).
  - 이 경우 VRAM 측정값은 유효합니다.

---

## 6. CPU 예측과 맞춰야 할 항목 체크리스트

CPU 매니페스트(`/home/inuos/cgf-fp16-680-20261111/handoff_codex/cpu_manifest.json`, 2026-11-12 시점)를 읽고 보이는 차이를 함께 적었습니다. CPU 코드에서 맞는지 직접 확인해 주세요.

| # | 항목 | GPU 실측 | CPU 쪽 확인 포인트 |
|---|---|---|---|
| 1 | 조건표 | `conditions_main395_l256.json` (d4a9b09c…) | `manifest/cases395.jsonl`이 이전 1024 표 기준이면 교체 필요 |
| 2 | 모델·revision | `models.json` 7b11e3e3… | revision 문자열 일치 |
| 3 | 입력 | 시드 20261006, §5.3-3 규칙 | 입력 sha256을 조건별 CSV의 `input_sha256`과 대조 |
| 4 | dtype | 실제 FP16 실행 | CPU 매니페스트에는 `applied_precision: fp32_execution_fp16_request_sizes`로 되어 있음. 요청 크기 변환이 모든 텐서에 적용되는지, 특히 T5에서 확인 (§8-4) |
| 5 | attention | 실제 백엔드: OPT·TinyLlama·Qwen·GPT-2·ViT는 sdpa(FlashAttention, L×L 행렬 없음), T5·CNN은 eager | CPU에서 고른 `_attn_implementation`과 aten 경로를 조건별로 기록. CPU sdpa가 math 경로면 L×L 임시 텐서 때문에 과대 예측 |
| 6 | 생성 설정 | greedy, min=max=N, use_cache, hub generation_config 미사용 | 동일 여부 |
| 7 | 측정 순서 | 로드 → warmup 3 → measure 5, 캐시 유지, measure마다 peak만 리셋 | CPU 매니페스트의 "fourth replay, k=3, measure_0 after warmup_0/1/2"가 이것과 같은 경계인지 |
| 8 | 텐서 수명 | 입력은 매 회차 새로 복사, 출력과 함께 회차 끝까지 유지 | 동일 여부 |
| 9 | 기준값 | `summary.first_measure_peak_reserved` (byte) | 동일 키로 비교 |
| 10 | 용량/OOM | 25,306,529,792 B를 넘으면 gpu_oom. 이전 실행의 OOM 14개는 모두 `warmup_0`에서 발생 | CPU xMem 용량이 1 TiB면 OOM이 재현되지 않음. 예측 peak > 용량이면 "OOM 예측"으로 따로 판정 |
| 11 | GPU 전용 할당 | cuBLAS workspace(8,519,680 B), cuBLASLt 1 MiB, CNN cuDNN workspace | §8-3: 6개 trace 밖에서의 빈도는 미확인 |
| 12 | 프로세스 격리 | 조건마다 새 프로세스 | CPU도 조건마다 새 프로세스인지 |
| 13 | 결과 단위 | byte (조건별 CSV), MiB (`summary.csv`) | 단위 혼동 주의 |

---

## 7. 이전 실행(1024 조건표) 결과 요약 — CPU 쪽 질문에 대한 답

### 7.1 실행 정보

- label `rtx3090__main395`, 2026-11-12 09:51 ~ 23:13 KST
- measure: 381 ok / 14 gpu_oom
- validate: 371 ok / 24 gpu_oom
- 조건별 상세는 `results/rtx3090_main395_L1024/gpu_rtx3090_main395_L1024_per_condition.csv`에 있습니다(byte 단위).
  - 열: 조건 정의, `measure_status`, `measure_failed_stage`, 실제 `attn_implementation`, `class`, `param_bytes`, `cuda_total_memory`
  - `after_load_reserved/allocated`
  - `m1_start/peak/end_with_output_reserved/allocated`, `median/max_peak_*`, `max_nvml_process_peak`
  - `measure_new_tokens`(5회), `input_sha256`(전체 64자, 텐서별)
  - OOM일 때 `fail_reserved_peak / fail_allocated_peak / fail_reserved_current / fail_allocated_current / fail_num_alloc_retries / fail_error_head`
  - `validate_status / failed_stage / all_finite / steps_checked / first_nonfinite_step`
  - `worker_sha256`, `conditions_json_sha256`
- `summary.csv` 열은 MiB 단위입니다.
  - 매핑: `m1_peak_reserved_MiB` = `summary.first_measure_peak_reserved / 2^20`, `load_reserved_MiB` = `after_load reserved_bytes.all.current / 2^20`

### 7.2 measure gpu_oom 14개 (모두 `failed_stage=warmup_0`)

- gpt2-xl b128: in128/new512, in512/new128, in512/new512
- t5-3b b128: in128/new512, in128/new1024, in512/new128, in512/new512, in512/new1024, in1024/new128, in1024/new512, in1024/new1024
- t5-3b b32: in512/new1024, in1024/new512, in1024/new1024
- `stats_at_failure` 원본은 각 `measure.json`에 있고, 요약값은 위 CSV의 `fail_*` 열에 있습니다.
- 새 조건표에서 1024 조건은 빠집니다. 따라서 L256 실행에서 남는 OOM 후보는 다음과 같습니다(실측 전이라 미확인).
  - gpt2-xl b128 3개
  - t5-3b b128의 128/512 조합 3개
  - 256이 들어간 t5-3b 조건 일부

### 7.3 validate에서만 gpu_oom인 10개 (measure는 ok, VRAM 값 유효)

| condition_id | validate failed_stage | measure m1_peak_reserved |
|---|---|---|
| gpt2__b128__in512__new128 | measure_0 | 13582 MiB |
| gpt2__b128__in512__new512 | measure_0 | 13582 MiB |
| gpt2-medium__b128__in512__new128 | measure_0 | 22528 MiB |
| gpt2-medium__b128__in512__new512 | measure_0 | 22528 MiB |
| opt-125m__b128__in512__new128 | measure_0 | 10446 MiB |
| opt-125m__b128__in512__new512 | measure_0 | 10448 MiB |
| opt-125m__b128__in512__new1024 | measure_0 | 12426 MiB |
| opt-125m__b128__in1024__new128 | measure_0 | 20570 MiB |
| opt-125m__b128__in1024__new512 | measure_0 | 20570 MiB |
| opt-125m__b128__in1024__new1024 | measure_0 | 20570 MiB |

- 원인: validate hook이 매 스텝 원시 logits(B×V)에 대해 추가 텐서를 만듭니다(코드 기준).
  - 정확한 할당 내역은 trace를 뜨지 않아 미확인입니다.
- 위 opt-125m b128의 in512 3개 중 new128/new512는 L256 조건표에도 있습니다. 같은 결과가 예상됩니다.

---

## 8. CPU 매니페스트의 open_questions에 대한 답 (2026-11-15 기준)

1. **본실험 결과 경로와 열 매핑**
   - §2와 §7.1을 보세요.
   - 이전 실행 395개의 원본 결과와 조건별 CSV를 이번에 함께 보냈습니다.
   - L256 실행 결과는 끝나면 같은 형식으로 보냅니다.
2. **cuda_total_memory, gpu_oom 목록, failed_stage, stats_at_failure**
   - 25,306,529,792 B입니다.
   - 목록은 §7.2, 값은 조건별 CSV의 `fail_*` 열과 원본 `measure.json`의 `stats_at_failure`에 있습니다.
3. **6개 trace 밖에서 cuBLAS 8,519,680 B / cuBLASLt 1 MiB 요청의 위치·빈도·수명, CNN cuDNN workspace**
   - 미확인입니다. 본실험은 trace 모드가 아니라 요청 단위 기록이 없습니다.
   - 필요하면 대상 condition_id를 지정해 주세요. 본실험이 끝난 뒤 trace 모드로 추가 기록할 수 있습니다.
4. **T5에서 GPU 8 KB vs CPU 16 KB, GPU 1 MiB vs CPU 2 MiB 요청**
   - 미확인입니다.
   - 크기가 정확히 2배이므로, CPU가 FP32로 실행하면서 해당 텐서에 fp16 요청 크기 변환이 적용되지 않았을 가능성을 먼저 확인해 주세요.
     - 참고로 HF T5LayerNorm은 분산을 float32로 계산합니다. 이 연산은 GPU에서도 FP32 텐서를 만듭니다. 이 경우와 구분이 필요합니다.
5. **조건별 전체 입력 SHA256, validate 항목**
   - 조건별 CSV의 `input_sha256`, `validate_*` 열에 있습니다(이전 실행 395개 전체).
6. **AWS 실행 label, 인스턴스, hash, 회수 경로**
   - 미확인입니다. AWS 결과는 INUGPU에서 보이지 않습니다.
   - 계획된 label은 `aws-<인스턴스타입>`입니다. 결과를 회수하면 따로 전달합니다.

---

## 9. 진행 상황

- RTX 3090 L256 본실험: 2026-11-15 17:00 시작, tmux `cgfvram` 창 `main395l256`에서 실행 중
  - 예상 소요: measure 약 6.2시간 + validate 약 1.4시간. 이전 실행 시간으로 추정한 값입니다.
- 끝나면 `tools/export_for_inuos.py`로 조건별 CSV를 만들어 이 디렉터리 `results/rtx3090_main395_L256/`에 추가합니다.
- INUOS에서 INUGPU로의 직접 접속은 사용자 쪽에서 설정 중입니다.
  - 접속 주소는 사설 IP `[사설 IP 비공개]`입니다(네트워크 도달은 확인됨). 키 등록이 필요합니다.

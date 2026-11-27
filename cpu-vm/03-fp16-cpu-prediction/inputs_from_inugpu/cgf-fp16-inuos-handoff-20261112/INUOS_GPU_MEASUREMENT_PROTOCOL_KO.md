# CloudGPUFit FP16 VRAM — GPU 실측 방식 정리 (INUOS CPU 예측용)

- 작성: 2026-11-12, INUGPU
- 대상: INUOS에서 CPU로 GPU VRAM을 예측하는 쪽
- 목적: GPU 쪽(INUGPU RTX 3090, AWS)에서 어떻게 측정하는지 그대로 적어, CPU 예측이 같은 절차와 같은 기준값을 최대한 따라 할 수 있게 합니다.
- 범위: 아래는 모두 **실제 코드와 실행 기록에 있는 내용**입니다. 확인하지 않은 추정은 "미확인"으로 표시했습니다.

---

## 1. 한 줄 요약

- 측정 단위: 조건 1개를 **새 프로세스(새 컨테이너) 1개**에서 실행
- 측정 절차
  1. FP16 모델을 한 번 로드
  2. warmup 3회
  3. measure 5회 (회차 사이 allocator 캐시는 유지하고, 각 measure 직전에 peak 통계만 초기화)
- **비교 기준값: 1회차 measure의 peak reserved (`summary.first_measure_peak_reserved`)**
- 출력 유한성 검사는 별도 프로세스(validate)에서 1회 추론으로 수행
- attention: Transformers 기본값 (`attn_implementation` 인자 생략)

---

## 2. 고정 요소 (GPU 쪽 전부 동일)

| 항목 | 값 |
|---|---|
| Docker 이미지 | `cloudgpufit-bench:cu124-v1` |
| 이미지 ID | `sha256:457da255e6e2de7ee05ff7814674c90abcf6d1ed04e73df3cc4eebcdb31fb032` (클래식 저장소에서는 config digest `sha256:e33dd4e4…`) |
| 소프트웨어 | python 3.11.11, torch 2.6.0+cu124, cuDNN 90100, transformers 4.52.4 |
| worker | `src/vram_worker.py`, sha256 `d06f0f15982c811c4bc6a24028f71728e461c6ee3651bcfd520ba49b0d675883` |
| 모델·revision | `config/models.json`, sha256 `7b11e3e36eafd5ec73cb7686469acb21036fb60141903e34381f3eda7bad6589` |
| 조건표 (본실험) | `config/conditions_main395.json`, sha256 `2634bda772c263fa52fd257605caa360f3256cb542d2ed18552ff2341d2fe258` |
| 파일럿 26개 | `config/pilot.json`, sha256 `ea739d8535f7fea531b2c40fb5bd3f6fd51fa2932aa328eb1c68abb71cd02b73` |
| 시드 | 20261006 |

- `models.json`의 `common.attn_implementation: "eager"`는 680표를 만들 때의 설명 문구입니다.
  - worker는 이 값을 읽지 않습니다. 실제 실행은 `--attn default`이며, 인자를 생략해 기본값을 씁니다.
- CPU에서도 **transformers 4.52.4 + torch 2.6.0**을 쓰는 것을 권장합니다.
  - generate 내부 동작(KV cache 방식, logits 계산 범위 등)이 버전에 따라 달라질 수 있기 때문입니다.
  - 같은 버전·같은 호출이면 같은 동작이 나옵니다.

### 2.1 조건표 395개

| 그룹 | 모델 | 축 | 후보 | 조건 수 |
|---|---|---|---|---|
| 디코더 LLM | OPT-125M, TinyLlama-1.1B-Chat, Qwen2.5-1.5B-Instruct | 입력×생성 | {128,512,1024}² | 135 |
| GPT-2 | GPT-2, GPT-2 Medium, GPT-2 XL | 입력×생성 | {128,512}² | 60 |
| 인코더-디코더 | T5-small, T5-base, T5-3B | 인코더 입력×디코더 생성 | {128,512,1024}² | 135 |
| CNN | ResNet-18, ResNet-50, MobileNet V2 | 해상도 | 128, 224, 256, 512 | 60 |
| ViT | ViT Large | 해상도 | 224 고정 | 5 |

- 모든 그룹의 배치: 1, 8, 16, 32, 128
- condition_id 형식: `<short>__b<B>__in<L>__new<N>` 또는 `<short>__b<B>__r<R>`
- 680표의 부분집합이며, 조건 객체는 원본 그대로입니다.

---

## 3. 실행 구조

### 3.1 호스트 runner (`src/run_conditions.py`)

- 조건마다, 모드(measure/validate)마다 새 `docker run --rm` 1회 → 프로세스 상태가 매번 초기화됩니다.
- 조건은 순차 실행이며 GPU 1개만 씁니다.
- 실행 전에 다른 GPU 프로세스가 있으면 중단합니다.
- 재개 방식: 결과 파일이 이미 있으면 건너뜁니다. 실패도 그대로 보존하고 자동 재시도는 하지 않습니다.
- 컨테이너 실행 내용:

```
docker run --rm --pull=never --network none --pid=host --gpus device=<GPU UUID> --user <uid>:<gid>
  -e HOME=/tmp -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e CUDA_MODULE_LOADING=LAZY
  -v src:/src:ro -v config:/config:ro -v <hf-cache>:/hf:ro -v runs/<label>/<cid>:/out
  --entrypoint python <image> -I /src/vram_worker.py --models /config/models.json
  --conditions /config/conditions_main395.json --condition <cid> --mode measure|validate --out /out/<mode>.json --attn default
```

- 결과 위치: `runs/<label>/<condition_id>/{measure,validate}.json`
- 진행 기록: `progress.jsonl`, `sessions.jsonl` (이미지 ID, GPU, 인스턴스 타입, 조건 파일·코드 sha256)

### 3.2 INUGPU(로컬) vs AWS 차이

| 항목 | INUGPU | AWS |
|---|---|---|
| GPU | RTX 3090 24GB, 드라이버 570.211.01 | g6(L4) 우선, 이후 g4dn/g5/g6e/gr6/g7/g7e |
| `cuda_total_memory` | 25,306,529,792 B | 각 결과의 `environment.cuda_total_memory` 참고 |
| 이미지·worker·조건표·revision | 동일 | 동일 (이미지와 체크섬 대조 후 실행) |
| 모델 캐시 | 로컬 HF 캐시 | HF에서 같은 revision으로 받은 뒤 75개 파일 전체 내용 해시 검증 |
| label | `rtx3090__main395` | `aws-<인스턴스타입>` (예: `aws-g6.xlarge`) |
| 순서 | 본실험 395개 바로 실행 (파일럿 26개는 이미 통과) | 파일럿 26개 → PASS일 때만 본실험 395개 |
| 실행 모드 | measure 전체 → validate 전체 | 동일 (validate 생략 옵션 있으나 기본은 실행) |

- AWS 파일럿 PASS 조건
  - 26개 모두 measure와 validate가 ok
  - 텍스트 모델은 생성 토큰 수가 요청값과 같음
  - `attn_requested=default`
  - worker sha가 같음
  - 입력 해시가 RTX 3090과 같음
  - VRAM 값은 GPU가 달라 비교하지 않습니다.

---

## 4. worker 상세 절차 (`vram_worker.py`) — CPU에서 모사할 대상

### 4.1 프로세스 시작 설정

```python
os.environ["CUDA_MODULE_LOADING"] = "LAZY"          # 컨테이너 env로도 지정
torch.set_num_threads(min(8, os.cpu_count()))        # INUGPU는 4
torch.manual_seed(20261006)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False               # conv 알고리즘 자동 탐색 안 함
torch.cuda.init(); torch.cuda.synchronize()
start_state = torch.cuda.memory_stats()              # 시작 시 allocator 비어 있음(reserved 0) 확인용
```

- `PYTORCH_CUDA_ALLOC_CONF`는 설정하지 않습니다(기본 caching allocator).
- 환경변수 값은 결과의 `environment.env`에 기록됩니다.

### 4.2 모델 로드

```python
torch.cuda.reset_peak_memory_stats()
model = AutoModelForXxx.from_pretrained(model_id, revision=<models.json>, torch_dtype=torch.float16,
            low_cpu_mem_usage=True, device_map={"": "cuda:0"}, use_safetensors=<models.json>,
            local_files_only=True)                   # attn_implementation 인자 없음
model = model.to(device="cuda", dtype=torch.float16).eval()
model.config.output_attentions = False
model.config.output_hidden_states = False
torch.cuda.synchronize()
after_load = memory_stats + max_memory_allocated/reserved
```

- loader 클래스
  - decoder_llm: `AutoModelForCausalLM`
  - text_encdec: `AutoModelForSeq2SeqLM`
  - vision_cls: `AutoModelForImageClassification`
- `use_safetensors=False`인 모델은 opt-125m과 vit-large(`.bin`)이고, 나머지는 safetensors입니다.

### 4.3 입력 생성 (호스트 CPU 텐서, 토크나이저·이미지 전처리 없음)

```python
g = torch.Generator().manual_seed(20261006)
# 텍스트 (decoder_llm, text_encdec)
ids = torch.randint(3, config.vocab_size, (B, L), generator=g, dtype=torch.int64)
for tok in [pad, eos, bos, decoder_start]의 값(int 또는 list):  ids[ids == tok] = 3
attention_mask = torch.ones(B, L, dtype=torch.int64)
# 이미지 (vision_cls)
pixel_values = torch.randn(B, 3, R, R, generator=g, dtype=torch.float32).to(torch.float16)
```

- 입력 sha256은 결과의 `inputs.<name>.sha256`에 있습니다(바이트 기준). CPU 쪽 입력이 같은지 이 값으로 확인하세요.
- decoder_llm에서 `L+N > max_position_embeddings`이면 `invalid_input`으로 기록합니다(이번 조건표에서는 아직 미관측).

### 4.4 생성 설정 (텍스트)

```python
gen = GenerationConfig.from_model_config(model.config)
gen.update(do_sample=False, num_beams=1, use_cache=True, min_new_tokens=N, max_new_tokens=N,
           pad_token_id=(config.pad_token_id or eos), return_dict_in_generate=False, output_scores=False,
           output_attentions=False, output_hidden_states=False)
model.generation_config = gen          # hub의 generation_config.json(샘플링·penalty 등)을 쓰지 않음
```

- 생성 길이 확인
  - decoder: `out.shape[1] - L == N`
  - T5: `out.shape[1] - 1 == N` (decoder_start 1개 제외)
  - 다르면 `generation_length_mismatch`로 기록합니다.
- min = max 이므로 EOS로 일찍 끝나지 않고, 항상 정확히 N 토큰을 만듭니다.
- 실제 적용된 설정 전체는 결과의 `generation_config`에 있습니다.

### 4.5 1회 추론(invoke) — warmup/measure 공통

```python
with torch.inference_mode():
    gi = {k: v.to("cuda") for k, v in inputs.items()}     # 매 회차 입력을 새로 GPU로 복사
    if vision: o = model(**gi)                            # logits (B, num_labels) 확인
    else:      o = model.generate(**gi, generation_config=model.generation_config)
torch.cuda.synchronize()
```

- 입력 GPU 복사본(`gi`)과 출력(`o`)은 회차 끝까지 살아 있다가, 기록이 끝난 뒤 `del` 합니다.
- 비전 모델은 `model(...)` forward 1회만 합니다(generate 없음).

### 4.6 warmup 3회 → measure 5회

```
for i in range(3):   # warmup
    invoke(); synchronize(); end_with_output 기록; del gi, o; synchronize()
for i in range(5):   # measure
    synchronize()
    start = {allocated: memory_allocated(), reserved: memory_reserved()}
    torch.cuda.reset_peak_memory_stats()       # peak만 초기화, 캐시는 그대로(empty_cache 없음)
    NVML 샘플러 시작 (10ms 간격, 이 프로세스 PID의 usedGpuMemory)
    invoke(); synchronize()
    peak = {allocated: max_memory_allocated(), reserved: max_memory_reserved()}
    end_with_output = {allocated, reserved}    # 출력·입력 텐서가 아직 살아 있는 상태
    NVML 샘플러 종료
    output_sha256 = sha256(o.cpu() 바이트)
    del gi, o; synchronize()
    released = {allocated, reserved}
```

- `empty_cache()`는 어디에서도 호출하지 않습니다.
- `summary` 항목
  - `first_measure_peak_reserved/allocated`
  - `first_measure_start_reserved/allocated`
  - `median/max_peak_reserved`, `median/max_peak_allocated`
  - `max/median_nvml_process_peak`

### 4.7 validate (별도 프로세스)

- 모델 로드, 입력, 생성 설정은 위와 동일합니다. warmup 0, measure 1회입니다.
- 모델에 forward hook을 걸어, **모든 forward(생성 모든 스텝)의 원시 logits**에 대해 `torch.isfinite(...).all()`을 확인합니다.
  - logits processor 이전 값을 봅니다. min_new_tokens가 EOS를 -inf로 바꾸는 영향을 피하기 위해서입니다.
- 결과: `validation.steps_checked`, `all_finite`, `first_nonfinite_step`. 비유한 값이면 `status=nonfinite_output`
- validate의 메모리 값은 측정값으로 쓰지 않습니다(hook 결과 텐서가 추가로 할당되므로).

### 4.8 trace 모드 (allocator 요청 기록, 파일럿 일부만)

- `torch.cuda.init()` 전에 `_record_memory_history(enabled="all", context=None, stacks="python", max_entries=5e7)`를 켭니다.
- 기록 구간: 로드 → warmup 3 → measure 1
- 경계마다 synchronize 후 `memory_stats` 카운터를 기록합니다(`model_load_begin/end`, `warmup_i_begin/end`, `measure_0_begin/end`).
- 결과 파일: `trace.json` + `trace.snapshot.pickle.gz` (`device_traces[0]`에 alloc/free_completed/segment 이벤트)
- 대상 6개: gpt2 b8/128/128, opt-125m b8/128/128, qwen2.5-1.5b b8/128/128, t5-small b8/128/128, resnet-50 b8/224, gpt2 b16/512/512
  - 기본 attention(v2)과 eager(v1) 각각 있습니다.

### 4.9 실패 분류 (`status`, `failed_stage`에 단계 기록)

- `gpu_oom`, `cpu_oom`, `fp16_unsupported`, `invalid_input`, `generation_length_mismatch`, `nonfinite_output`, `compat_or_runtime_error`
- 컨테이너가 결과 없이 죽은 경우 runner가 아래 중 하나로 기록합니다.
  - `worker_killed_cpu_oom_suspected` (rc 137)
  - `timeout` (3600초)
  - `worker_exit_<rc>`
- 실패한 조건을 더 작은 배치나 다른 dtype으로 바꿔 다시 돌리지 않습니다.

---

## 5. 지금까지 GPU에서 관측된 사실 (RTX 3090 파일럿 26개, 기본 attention)

1. **실제 attention 경로** (별도 진단 프로세스, torch profiler 커널 이름 기준)
   - OPT, TinyLlama, Qwen2.5, GPT-2 3종: `_attn_implementation=sdpa`이며, prefill과 decode 모두 `pytorch_flash::flash_fwd_kernel`(FlashAttention)만 사용
     - 레이어 수만큼 호출되고, L×L attention 행렬을 만들지 않습니다.
   - ViT Large: sdpa → FlashAttention. patch embedding conv는 cuDNN입니다.
   - T5 3종, ResNet, MobileNet: 기본값 자체가 eager이며, sdpa로 바뀌지 않습니다.
     - T5 eager는 attention 가중치(B×H×Lq×Lk)를 만듭니다.
2. **피크 위치**
   - 52개 measure(eager·기본 각 26) 모두에서 1회차 시작 reserved = 1회차 peak reserved이고, measure 5회의 peak가 모두 같습니다.
   - 즉 reserved peak는 **warmup 3회가 만든 캐시 상태**에서 이미 정해지고, measure 중에 더 늘지 않습니다.
3. **eager → 기본(sdpa) 변화**
   - sdpa를 쓰는 14개 조건 중 10개에서 peak reserved가 줄었습니다(최대 TinyLlama b16/512/512 −1060MiB).
   - 3개는 같았고, 1개(gpt2-xl b8/128 +28MiB)는 늘었습니다.
4. **xMem 재생 결과**
   - GPU trace 12개의 요청을 그대로 xMem CachingAllocator에 넣으면, 모든 경계에서 reserved/allocated와 segment 수가 GPU와 **완전히 일치**했습니다.
   - 짝 없는 free는 0이었습니다.
   - 따라서 CPU 예측과 GPU 실측이 다르면, allocator 정책보다 **요청 목록(입력 trace) 차이**를 먼저 보는 것이 맞습니다.
5. **Qwen2.5 + eager(v1)의 nonfinite**
   - FP16 eager에서 출력이 NaN이 되었지만, VRAM 요청 크기는 shape로 정해지므로 영향을 받지 않습니다.
   - 기본(sdpa) 실행에서는 26/26 정상이었습니다.

---

## 6. CPU 예측이 GPU 실측을 최대한 따라 하기 위한 체크리스트

### 6.1 반드시 같게 맞출 것

- [ ] 조건 ID·모델 revision·조건표: `conditions_main395.json`, `models.json`의 sha가 같은지
- [ ] 입력: 시드 20261006, 위 4.3 규칙 그대로. 결과의 입력 sha256과 대조
- [ ] 생성 설정: 4.4 그대로(greedy, min=max=N, use_cache, hub generation_config 무시). transformers 4.52.4
- [ ] 로드 방식: FP16, `attn_implementation` 생략. 로드 후 `.to(fp16).eval()`, output_attentions/hidden_states False
- [ ] 실행 순서: 로드 → warmup 3 → measure 1(이상). **캐시를 유지한 채** 재생하고, 각 measure 직전에 peak만 초기화
- [ ] 회차 내 텐서 수명: 입력을 매 회차 새로 device로 복사, 출력과 입력은 회차 끝까지 유지 후 해제
- [ ] 비교 기준값: `summary.first_measure_peak_reserved` (보조로 `first_measure_peak_allocated`, `after_load.reserved_bytes.all.current`, `first_measure_start_reserved`)

### 6.2 GPU와 CPU가 구조적으로 다를 수 있는 부분 (CPU 쪽에서 기록 필요)

- **attention 커널**
  - GPU sdpa는 FlashAttention(L×L 행렬 없음)입니다.
  - CPU에서 sdpa를 쓰면 CPU가 고르는 aten 연산(flash CPU 커널인지 math 경로인지)이 다를 수 있습니다.
    - math 경로라면 L×L 중간 텐서가 생겨 과대 예측이 됩니다.
  - CPU 실행에서 실제 `_attn_implementation`과 선택된 aten 연산을 조건별로 기록해 주세요.
- **GPU 전용 할당**
  - cuBLAS/cuDNN workspace 등은 CPU 실행에서 생기지 않을 수 있습니다.
  - GPU trace(4.8)의 measure_0 구간 요청 크기 분포와 CPU 쪽 요청을 대조하면 차이가 나는 요청을 찾을 수 있습니다.
  - 미확인: 어떤 요청이 workspace인지 개별 식별은 아직 하지 않았습니다.
- **conv 알고리즘**: GPU는 `cudnn.benchmark=False`, TF32 off입니다. CNN의 workspace 크기는 GPU 종류와 cuDNN 버전에 따라 달라질 수 있습니다.
- **FP16 on CPU**
  - CPU에서 일부 연산이 Half를 지원하지 않으면 `fp16_unsupported`로 기록해 주세요.
  - **dtype을 바꿔 우회하지 마세요.** 바꾸면 GPU 조건과 다른 조건이 됩니다.
- **OOM 판정**
  - GPU 용량은 결과의 `environment.cuda_total_memory`입니다(3090: 25,306,529,792 B).
  - GPU에서 `gpu_oom`인 조건은 `failed_stage`(model_load / warmup_i / measure_i)와 `stats_at_failure`가 남습니다.
- **NVML 값**: CUDA context 등 allocator 밖 메모리를 포함하므로 reserved와 직접 비교하지 마세요(참고용).
- **validate 결과**: 출력이 비유한 값인 조건은 VRAM 값은 유효하지만 "출력 무효"로 따로 표시합니다.

### 6.3 조건 대응표를 만들 때 쓸 키

- `condition_id`(그대로), `model.attn_implementation`, `status`, `failed_stage`
- `summary.first_measure_peak_reserved`, `summary.first_measure_peak_allocated`
- `after_load.reserved_bytes.all.current`
- `inputs.*.sha256`, `measures[0].new_tokens`, validate의 `validation.all_finite`

---

## 7. 파일과 진행 상황

- 이 디렉터리(INUOS 전달용)

  | 경로 | 내용 |
  |---|---|
  | `config/` | `conditions_main395.json/.csv`, `conditions.json`(680), `models.json`, `pilot.json`, `trace_pilot_v2.json` |
  | `src/vram_worker.py` | GPU worker 원본 (참고용) |
  | `src/run_conditions.py` | GPU runner 원본 (참고용) |
  | `reference/pilot_rtx3090_eager_vs_default.csv` | 파일럿 26개 측정값 (eager vs 기본) |
  | `reference/attention_kernels_rtx3090_default.json` | 실제 사용된 GPU 커널 |
  | `reference/replay_table.md` | xMem 재생 결과 |
  | `reference/REPORT_rtx3090_pilot_eager_vs_default_KO.md` | 파일럿 통합 보고서 |
  | `reference/traces_attn_default/<cid>/trace.json` + `trace.snapshot.pickle.gz` | GPU allocator 요청 기록 6개 |
  | `SHA256SUMS` | 파일별 체크섬 |

- 진행 상황
  - RTX 3090 본실험 395개: 2026-11-12 09:51 시작, 진행 중
    - 결과: INUGPU `/home/inugpu/cgf-fp16-vram-20261111/runs/rtx3090__main395/`
    - 끝나면 요약 CSV를 따로 전달합니다.
  - AWS g6: 이관 파일 전송 중. 파일럿 PASS 후 본실험을 실행합니다.

# GPU 실측 구성과 CPU 예측 구성의 동등성 검토

**판정: 같은 395개 FP16 대상 조건을 비교하도록 설계됐고, 입력·호출 설정·첫 측정 회차의 경계는 맞춰져 있습니다. 그러나 실행 절차와 메모리 요청을 만드는 방식까지 동일한 실험이라고 볼 수는 없습니다. CPU는 GPU 실험의 근사 예측 구성입니다. 현재 근거만으로 두 구성의 메모리 요청 동등성이 검증됐다고 판단할 수 없습니다.**

이 판단은 오차 크기나 현재 완료 건수를 기준으로 하지 않습니다. 아래는 실험 구성 자체의 비교입니다. 완료 건수는 구성의 실제 적용을 확인할 수 있는 근거 범위를 뜻하며, 미완료를 구성 불일치로 취급하지 않습니다. CPU snapshot은 2026-11-12 11:28 KST, GPU snapshot은 11:25 KST입니다.

## 무엇이 같은가

| 구성 항목 | GPU 실측 | CPU 예측 | 판단 |
| --- | --- | --- | --- |
| 대상 조건 | GPU main395 조건표 | 동일 조건표에서 생성한 cases395 | **일치**. 395개 모두 모델·revision·가중치 포맷·요청 FP16·generate/forward·배치·입력길이·생성길이·해상도 조합 일치 |
| 모델 로드 설정 | 같은 pinned revision, loader, use_safetensors, low_cpu_mem_usage, local_files_only | 동일 pin과 loader 설정 사용 | **설정 일치**. 실제 checkpoint 파일 내용 hash의 VM 간 일치는 미검증 |
| 소프트웨어 | Python 3.11.11 / torch 2.6.0+cu124 / transformers 4.52.4 | 동일 버전 | **일치**. 이미지 자체는 다름 |
| 입력 생성 | seed 20261006; 텍스트 special IDs→3, int64 mask; 이미지는 FP32 randn→FP16 | 같은 규칙, host 입력 hash도 FP16 기준 | **일치**. 실제 확인 가능한 공통 성공20조건의 shape·dtype·전체 SHA256 동일 |
| 텍스트 호출 | GenerationConfig.from_model_config; greedy, beams=1, min=max=N, use_cache=True | 같은 설정과 generate 호출 | **일치**. 공통 텍스트 성공건은 전체 GenerationConfig 동일, 실제 생성길이도 N과 동일. T5 decoder_start 1개 제외 |
| 비전 호출 | model forward | model forward | **일치** |
| attention 선택 설정 | attn 인자 생략/default | attn 인자 생략/default | **설정 일치**. 실제 backend/kernel 동등성은 아래와 별개 |
| 케이스 격리 | 케이스별 신규 컨테이너/프로세스 | 케이스별 신규 컨테이너/프로세스 | **일치** |
| 회차 내 입력·출력 수명 | invoke 내부 GPU 입력 복사, M4까지 입력·출력 유지, 이후 해제 | invoke 내부 clone, M4까지 유지, 이후 해제 | **의미상 경계 일치**. CPU 이미지 실제 연산 입력은 FP32이며 요청 크기는 환산 |
| 캐시와 비교 경계 | warmup3 이후 measure_0, 캐시 유지, measure 직전 peak만 reset | load→3번 warmup 재생→measure_0 재생, 캐시 유지, local peak baseline 재설정 | **대응 목표 일치**. 실제로 같은 캐시 상태가 만들어지는지는 요청열 동등성에 달림 |

조건표와 models.json의 SHA256 일치는 파일별로 검증했습니다. 모델 revision/가중치 포맷 일치와 checkpoint blob 바이트 일치는 구분합니다. 준비된 AWS 번들에서는 계획 설정만 확인할 수 있으며 실행 구성이 실제 적용됐다고 주장하지 않습니다.

## 무엇이 다르며 구성 동등성에 영향을 주는가

| 구성 항목 | GPU 실측 | CPU 예측 | 판정과 영향 |
| --- | --- | --- | --- |
| 실제 연산 정밀도 | 네이티브 FP16 모델로 추론 | FP32 모델로 추론하고 shadow dtype으로 FP16 요청 크기 환산 | **의도된 차이이지만 동등성 미검증**. 단순히 모든 텐서를 절반으로 줄이는 방식은 아니지만, inferred tag·캐스트·dtype 분기·중간 텐서가 GPU FP16과 같다는 보장은 없음. FP16 출력 유한성도 CPU에서 판정 불가 |
| 모델 로드 이력 | GPU from_pretrained와 to(cuda, fp16)의 실제 할당 | CPU에서 FP16 모델 구조를 읽어 GPU load 요청을 합성한 뒤 FP32 실행 모델을 별도로 로드 | **로드 경로가 다름**. 기존 검증은 13모델의 저장된 load 지표 및 일부 모델의 요청열을 뒷받침하지만, 로드 실행 자체가 동일하다는 뜻은 아님 |
| 워밍업 실제 실행 | 실제 추론 3회 | 측정용 실제 추론 trace 1회를 3번 warmup으로 복제 | **절차 불일치/근사**. 최초 호출 전용 할당·회차별 상태·요청 차이가 있으면 GPU와 다른 캐시 상태를 만들 수 있음 |
| 측정 실제 실행 | 실제 measure 5회 | 같은 trace를 추가 1회 재생해 measure_0 예측 | **절차가 다름**. 첫 회차 비교에는 대응할 수 있으나 GPU 5회 측정 전체를 재현하지 않음 |
| attention 및 연산 커널 | CUDA FlashAttention, CUDA eager 연산, cuDNN 등 | CPU attention/BLAS/conv 커널 | **결과에 영향을 줄 수 있는 차이**. 같은 sdpa/eager 설정도 임시 텐서·workspace·할당 수명이 같다는 뜻이 아님 |
| GPU 전용 메모리 요청 | cuBLAS/cuBLASLt/cuDNN workspace 등 실제 할당 | baseline에는 GPU 전용 요청 없음. 별도 상수 workspace variant만 있음 | **요청 구성 불일치**. variant는 baseline이 아니며 조건별 실제 GPU 요청을 재현했다는 근거가 아님 |
| allocator와 용량 | 실제 CUDA native allocator와 GPU 용량 | pinned xMem allocator, 1 TiB 가상 용량, 현재 재생 stream=0 | **모델링 차이**. GPU OOM·용량 압박·다중 stream 상태까지 같은 실험이 아님. 현재 GPU 요청을 xMem에 넣는 검증과 CPU가 같은 요청을 만드는 검증도 구분해야 함 |
| peak 측정 종료 | invoke 완료 M4에서 peak 읽음 | 해제 완료 M5까지 peak 갱신 | **코드상 구간 차이**. 기존20 NPZ에서는 M4~M5 신규 alloc=0이므로 그 범위에서는 peak 비교 영향 없음. 모든 조건의 일반 동등성으로 확대 불가 |

CPU attention probe는 본 추론 trace 밖에서 B=1, 텍스트 2토큰으로 별도 수행됩니다. 이는 전체 배치·전체 생성길이의 실제 측정 경로를 완전히 관측한 기록은 아닙니다.

## 지금 내릴 수 있는 결론

- **“같은 대상 workload 조건을 쓰는가?” → 예.** 같은 395개 조건과 모델 pin, 입력 규칙, 호출 설정을 사용하도록 구성됐습니다. 실제 적용도 공통 성공20조건에서 확인됐습니다.
- **“같은 warmup·measure를 실제 실행하는가?” → 아니요.** GPU는 실제 3+5회, CPU는 단일 trace를 3+1회 재생합니다. 캐시 유지와 첫 측정 경계는 맞췄지만 회차별 요청을 같다고 가정합니다.
- **“GPU FP16 메모리 요청을 CPU가 동등하게 재현하는가?” → 현재는 입증되지 않았습니다.** 실제 FP32 실행, dtype 추정, CPU/GPU 커널과 GPU 전용 workspace 차이가 남습니다.
- **“그럼 무엇을 비교할 수 있는가?” → 근사 예측값을 같은 의미의 GPU 첫 측정 지표와 비교할 수 있습니다.** 주 대상은 first_measure_peak_reserved, 보조는 allocated와 start/M4/M5·after_load입니다. 이것이 실험 구성 전체의 동등성이나 요청열 동등성을 뜻하지는 않습니다.

구성 동등성을 더 강하게 주장하려면 **기존 산출물만으로** GPU FP16과 CPU 환산 요청의 연산/dtype/수명 대응, warmup 회차 복제의 적용 범위, GPU 전용 workspace의 누락 범위를 확인해야 합니다. checkpoint 내용 hash도 양쪽 기존 기록으로 대조해야 합니다. 이 단계에서는 코드를 맞추거나 실험을 재실행하지 않았습니다. 질문은 review_INUOS.md의 INUOS-Q001~Q007에 유지합니다.

## 근거 위치

- CPU: `/home/inuos/cgf-fp16-680-20261111/src/cgf680/protocol2.py:29`(load), `:51`(입력), `:73`(생성), `:86`(invoke).
- CPU: `src/cgf680/cpu_worker2.py:45`(재생), `:170`(xMem 검사), `:177`(FP16 load 구조), `:186`(FP32 실행 모델), `:215`(추론 경계), `:255`(환산), `:272`(재생 호출), `:291`(workspace variant), `:297`(finite 불가).
- GPU 인계 원본: `/home/inuos/cgf-fp16-inuos-handoff-20261112/src/vram_worker.py:239`(FP16 GPU load), `:268`(입력), `:292`(생성 설정), `:308`(invoke), `:323`(실제 warmup), `:343`(실제 measure/reset), `:356`(peak/M4), `:364`(해제).
- 양쪽 manifest 및 공통 성공건의 실행 근거: `comparison_details.json`의 field_checks와 local_result_snapshot_checks. GPU 원격 근거는 전달된 manifest 인용이며 이 VM에서 원격 파일을 직접 읽은 것은 아닙니다.

---

## 부록: 스냅샷 실행 상태·수치 대조와 교환 상태

아래는 구성 검토를 보조하는 기존 상세 자료입니다. 실행 교집합과 예측 오차 크기를 구성 동등성의 판단 기준으로 사용하지 않습니다.

# CloudGPUFit INUGPU 실측 ↔ INUOS 예측 교차 검토

현재 비교 가능한 범위는 **본실험 성공 교집합 3조건**입니다. 별도로 **기본 attention 파일럿 성공 20조건**도 비교할 수 있습니다. 두 집합을 합산해 23개 독립 조건으로 세지 않습니다. 20조건 모두 요청 조합·전체 입력 shape/dtype/SHA256·실제 attention 설정·텍스트 GenerationConfig 및 실제 생성 길이가 일치합니다. CPU는 FP32 실행으로 얻은 FP16 요청 크기 예측이고 GPU는 네이티브 FP16이므로, 이는 예측 오차 비교이며 동일 정밀도 추론 실측끼리의 비교가 아닙니다. AWS prepared, GPU eager 과거 실행, validate/trace 수치는 baseline 오차 집계에서 제외했습니다.

작성 시각 2026-11-12T11:58:49.531259+09:00. CPU snapshot 2026-11-12T11:28:06.613092+09:00; GPU snapshot 2026-11-12T11:25:38.814873+09:00. 양쪽 실행은 계속될 수 있으며 이 보고서는 두 고정 manifest를 대조합니다. 새 결과를 원본 manifest에 섞지 않았습니다.

## 입력과 근거

- CPU: `/home/inuos/cgf-fp16-680-20261111/handoff_codex/cpu_manifest.json`; SHA256 `57f4cdd6142e5a6d87037a905e53aa92b9a8d9432591a426732429d7108faa1f`
- GPU: `/home/inuos/cloudgpufit-review/gpu_manifest.json`; SHA256 `db3411fe91346bca8ad7bee0efe208df7cde3dd18c011f7911c03948c716bef0`
- CPU 성공20건의 실제 result.json을 재열람하고 manifest에 기록된 result SHA256 20/20 일치를 확인했습니다.
- GPU 원격 파일은 이 VM에 없으므로 GPU manifest의 observed_in_results를 전달 근거로 사용합니다. 현재 GPU worker는 인계 사본 `/home/inuos/cgf-fp16-inuos-handoff-20261112/src/vram_worker.py`로 확인했고 SHA256은 GPU 성공 결과의 d06f0f15…와 일치합니다. 과거 v1 소스는 이 사본으로 확정하지 않습니다.
- 전체 조합/상태: [case_crosswalk.csv](case_crosswalk.csv), 실제 수치/bytes: [paired_metrics.csv](paired_metrics.csv), 근거와 비교값: [comparison_details.json](comparison_details.json).

## 집계 방법과 상태

키는 `(model, revision, use_safetensors, requested_precision, generate/forward, batch, input_len, max_new_tokens, resolution)`입니다. condition_id만으로 합치지 않고 이 키를 전부 대조했습니다. GPU의 mode=measure/validate/trace와 추론 호출 generate/forward를 구분하며, label별 실험 이력도 보존합니다. 실제 applied_precision을 동일 키의 필수 일치 조건으로 삼으면 CPU FP32↔GPU FP16이라 동일 정밀도 실행 교집합은 0건입니다. 요청 FP16 작업의 예측↔실측 비교에는 이 차이를 명시하고 사용합니다.

| 범위 | 공통 | CPU에만 | GPU에만 | 양쪽 결과 없음 |
| --- | --- | --- | --- | --- |
| main395 계획 조합 | 395 | 0 | 0 | — |
| main395 실제 완료 성공 | 3 | 17 | 70 | 305 |
| 모든 GPU label의 계획 조합(고유) | 395 | 0 | 2 | — |


GPU에만 있는 고유 조합 2건: `gpt2__b1__in32__new32`, `t5-small__b1__in32__new32`. 모두 길이32 smoke이며 main395 누락이 아닙니다. CPU snapshot은 성공20/실패0/실행중1/미시작374입니다. GPU main395는 성공 measure73/실패0/최종결과없음322, validate 최종결과없음395입니다. 최종결과없음에는 진행중이 포함될 수 있습니다.

| GPU label | measure 실행 | 그중 CPU 성공 공통 | measure 미완료 | measure 준비만 | 모든 mode 실패 | skip 이벤트 |
| --- | --- | --- | --- | --- | --- | --- |
| rtx3090 | 26 | 20 | 0 | 0 | 2 | 0 |
| rtx3090-smoke | 3 | 0 | 0 | 0 | 0 | — |
| rtx3090-smoke__attn-default | 1 | 0 | 0 | 0 | 0 | — |
| rtx3090__attn-default | 26 | 20 | 0 | 0 | 0 | 5 |
| rtx3090__main395 | 73 | 3 | 322 | 0 | 0 | 0 |
| rtx3090__main395-pathcheck | 3 | 2 | 0 | 0 | 0 | 0 |
| aws-<instance_type> | 0 | 0 | 0 | 395 | 0 | — |


GPU 전체 1710 레코드는 고유 조건이 아니라 label×condition×mode입니다: executed203(성공201/실패2), planned_not_yet_run717, prepared790입니다. AWS는 395조건×2mode의 prepared790이며 실제 실행0입니다. 실패2건은 과거 eager Qwen의 validate nonfinite_output이며 별도입니다. 기본 attention label의 skip5건은 이미 존재한 성공 파일에 대한 재개 로그이며 5개 실패나 5개 미실행으로 세지 않습니다. 정확한 mode는 INUOS-Q006에서 확인 요청했습니다. CPU는 per-case skip 로그가 없어 생략 여부 미확인입니다.

## 실패와 재개 생략의 상세

| label | condition_id | mode | status | failed_stage |
| --- | --- | --- | --- | --- |
| rtx3090 | qwen2.5-1.5b-instruct__b16__in512__new512 | validate | nonfinite_output | None |
| rtx3090 | qwen2.5-1.5b-instruct__b8__in128__new128 | validate | nonfinite_output | None |

위 실패는 과거 eager 실행의 validate 결과입니다. 현재 default 비교 성공건에 실패 상태를 전파하지 않았습니다.

| label | condition_id | 관측된 이벤트 | mode |
| --- | --- | --- | --- |
| rtx3090__attn-default | gpt2__b8__in128__new128 | `[1/6] skip gpt2__b8__in128__new128 (ok)` | 미확인 |
| rtx3090__attn-default | opt-125m__b8__in128__new128 | `[2/6] skip opt-125m__b8__in128__new128 (ok)` | 미확인 |
| rtx3090__attn-default | qwen2.5-1.5b-instruct__b8__in128__new128 | `[3/6] skip qwen2.5-1.5b-instruct__b8__in128__new128 (ok)` | 미확인 |
| rtx3090__attn-default | t5-small__b8__in128__new128 | `[4/6] skip t5-small__b8__in128__new128 (ok)` | 미확인 |
| rtx3090__attn-default | resnet-50__b8__r224 | `[5/6] skip resnet-50__b8__r224 (ok)` | 미확인 |

생략 이벤트 원문은 [skip_events.csv](skip_events.csv)에 있습니다. 같은 ID의 measure/validate/trace에 이 이벤트를 임의 배정하지 않았습니다. 정확한 호출은 INUOS-Q006 답변 대기입니다.

## 항목별 판정

| 항목 | 판정 | 근거와 영향 |
| --- | --- | --- |
| 395 조건/13모델 revision/포맷/배치/입출력 | 일치 | 정규화 요청 키 395/395, config SHA256 동일. checkpoint 실제 blob 바이트 hash는 별도 미확인(Q001). |
| 전체 입력 텐서 | 일치 | 공통20조건 result.inputs 전체 dict 일치: SHA256 64자리·shape·dtype 포함. 이전 prefix-only 판단을 갱신. CPU FP32용 내부 복사본은 host FP16 입력의 변환값. |
| generate/forward 및 실제 길이 | 일치 | 공통 텍스트12조건의 CPU actual length, GPU measure5회·warmup3회 길이가 요청 N과 일치. T5는 decoder_start 1개 제외. EOS 조기종료 관측 없음. 비전8조건은 forward. |
| GenerationConfig/KV cache | 일치 | 공통 텍스트 전체 dict 동일, min=max=N, greedy, use_cache=True, hub generation config 대체. |
| 실제 실행 정밀도 | 의도된 CPU·GPU 차이 | CPU FP32+FP16 size shadow, GPU param float16. FP16 유한성/모든 중간 dtype 동일성은 증명하지 않음. |
| 프로세스/로드/워밍업 | 의도된 CPU·GPU 차이 | 케이스별 신규 컨테이너. CPU FP16 load 합성→1trace 4재생; GPU 실제 FP16 load→warmup3→measure5. 동일 cache prefix 목표지만 요청열 동일성은 Q005. |
| cache 유지와 측정 경계 | 일치 | CPU 하나의 Replayer 유지, GPU empty_cache 없음; measure0 start에서 peak baseline 재설정. CPU M0/M4/M5는 GPU start/end_with_output/released 대응. |
| peak 구간 M0~M5 대 M0~M4 | 의도된 CPU·GPU 차이 | CPU는 해제까지 peak 갱신, GPU는 해제 전 peak 읽음. 기존20 NPZ 모두 M4~M5 신규 alloc0이라 현재 대상에서는 peak 비교 가능. 향후 케이스는 재검사 필요. |
| attention 설정/커널 | 의도된 CPU·GPU 차이 | 공통20 _attn_implementation 일치. CPU flash_attention_for_cpu와 GPU FlashAttention CUDA kernel은 다른 구현. T5 eager, CNN에는 attention 없음. |
| 요청 크기/템포러리/workspace | 결과에 영향을 줄 수 있는 불일치 | GPU-only workspace, CPU temporaries, inferred dtype tags. 동일 입력·설정이어도 수치 차이 존재. Q003/Q004, paired_metrics에 케이스·지표별 잔차. |
| 전체 checkpoint 내용 hash | 확인 불가 | revision/포맷 일치가 blob 내용 검증을 대신하지 않음. Q001. |
| FP16 출력 유한성 | 확인 불가 | default 파일럿20조건은 GPU validate ok로 확인됨. main395 공통3건의 같은 label validate는 아직 없음. CPU는 판정불가(Q002). |
| 전체395 완료 및 AWS 실행 | 확인 불가 | 각 snapshot 시점 미완료/준비만 된 조건은 숫자 비교 불가(Q006/Q007). |

## 직접 대응 가능한 지표

동일 의미/경계의 지표로 오차를 비교할 수 있다는 판정이며, 요청열·커널까지 동일하거나 수치가 일치한다는 뜻은 아닙니다. 모든 값은 정수 bytes이며 MiB 표는 bytes/2^20입니다.

| CPU result 키 | GPU result 키 | 경계/판정 |
| --- | --- | --- |
| summary.first_measure_peak_reserved | summary.first_measure_peak_reserved = measures[0].peak.reserved | 주 비교 지표. warmup3 이후 measure0 peak. |
| summary.first_measure_peak_allocated | summary.first_measure_peak_allocated = measures[0].peak.allocated | 보조 peak. requested bytes와 혼동하지 않음. |
| summary.first_measure_start_{reserved,allocated} | measures[0].start.{reserved,allocated} | 4번째 재생 M0. GPU start 후 reset. |
| summary.first_measure_end_with_output_{reserved,allocated} | measures[0].end_with_output.{reserved,allocated} | M4; 입력복사본·출력 생존. |
| summary.first_measure_released_{reserved,allocated} | measures[0].released.{reserved,allocated} | M5; 입력복사본·출력 해제 후 캐시 유지. |
| summary.after_load_{reserved,allocated} | after_load.{reserved,allocated}_bytes.all.current | FP16 load 합성 종료, warmup 전. |
| summary.load_peak_allocated | after_load.max_allocated | load reset~load 완료 peak. |
| variant_with_cublas_workspace | baseline 대응 없음 | 상수 workspace를 추가한 비교 변형; CPU baseline 열에 섞지 않음. |
| CPU 대응 없음 | NVML/outside, 5회 median/max, validate 메모리 | 직접 오차 비교 제외. NVML max-reserved max 차이는 동시 측정 outside가 아님. |


코드 근거: `/home/inuos/cgf-fp16-680-20261111/src/cgf680/cpu_worker2.py:45`(load→4회 재생), `:67`(피크 baseline), `:215`(M0/M4/M5), `:262`(구간 추출), `:280`(키 매핑); GPU 인계 원본 `/home/inuos/cgf-fp16-inuos-handoff-20261112/src/vram_worker.py:239`(load reset), `:343`(measure reset), `:356`(start/peak/end), `:364`(해제), `:379`(summary). GPU 원격 근거 경로도 comparison_details.json에 보존했습니다.

## 본실험 실제 성공끼리의 수치

| condition_id | CPU reserved B | GPU reserved B | CPU−GPU B | 오차 % |
| --- | --- | --- | --- | --- |
| opt-125m__b16__in512__new512 | 1583349760 | 1598029824 | -14680064 | -0.9186 |
| opt-125m__b8__in128__new128 | 436207616 | 438304768 | -2097152 | -0.4785 |
| tinyllama-1.1b-chat__b8__in128__new128 | 2443182080 | 2443182080 | 0 | 0.0000 |


3건 모두 전체 입력·생성 설정·실제 토큰 수·기본 sdpa 설정은 일치합니다. TinyLlama reserved 동률이 allocated 동률까지 의미하지는 않으므로 paired_metrics의 11지표를 각각 보아야 합니다. 이 3건을 395건 정확도의 추정치로 일반화하지 않습니다.

## 별도 기본 attention 파일럿의 수치

| condition_id | CPU MiB | GPU MiB | 차이 MiB | 오차 % |
| --- | --- | --- | --- | --- |
| gpt2-medium__b8__in128__new128 | 1004.000 | 1004.000 | 0.000 | 0.000 |
| gpt2-xl__b8__in128__new128 | 3778.000 | 3812.000 | -34.000 | -0.892 |
| gpt2__b16__in512__new512 | 1836.000 | 1838.000 | -2.000 | -0.109 |
| gpt2__b8__in128__new128 | 474.000 | 476.000 | -2.000 | -0.420 |
| mobilenet-v2-1.0-224__b16__r512 | 1108.000 | 848.000 | 260.000 | 30.660 |
| mobilenet-v2-1.0-224__b8__r224 | 88.000 | 88.000 | 0.000 | 0.000 |
| opt-125m__b16__in512__new512 | 1510.000 | 1524.000 | -14.000 | -0.919 |
| opt-125m__b8__in128__new128 | 416.000 | 418.000 | -2.000 | -0.478 |
| qwen2.5-1.5b-instruct__b8__in128__new128 | 3512.000 | 3510.000 | 2.000 | 0.057 |
| resnet-18__b16__r512 | 330.000 | 458.000 | -128.000 | -27.948 |
| resnet-18__b8__r224 | 66.000 | 80.000 | -14.000 | -17.500 |
| resnet-50__b16__r512 | 618.000 | 812.000 | -194.000 | -23.892 |
| resnet-50__b8__r224 | 112.000 | 132.000 | -20.000 | -15.152 |
| t5-base__b16__in512__new512 | 1474.000 | 1474.000 | 0.000 | 0.000 |
| t5-base__b8__in128__new128 | 710.000 | 710.000 | 0.000 | 0.000 |
| t5-small__b16__in512__new512 | 738.000 | 738.000 | 0.000 | 0.000 |
| t5-small__b8__in128__new128 | 306.000 | 304.000 | 2.000 | 0.658 |
| tinyllama-1.1b-chat__b8__in128__new128 | 2330.000 | 2330.000 | 0.000 | 0.000 |
| vit-large-patch16-224__b16__r224 | 732.000 | 738.000 | -6.000 | -0.813 |
| vit-large-patch16-224__b8__r224 | 640.000 | 656.000 | -16.000 | -2.439 |


| label/metric | n | 정확히 동률 | 평균 절대 상대오차 % | 최대 절대 상대오차 % |
| --- | --- | --- | --- | --- |
| rtx3090__main395/first_measure_peak_reserved | 3 | 1 | 0.4657 | 0.9186 |
| rtx3090__main395/first_measure_peak_allocated | 3 | 0 | 1.0384 | 2.0937 |
| rtx3090__main395/after_load_reserved | 3 | 3 | 0.0000 | 0.0000 |
| rtx3090__main395/after_load_allocated | 3 | 3 | 0.0000 | 0.0000 |
| rtx3090__main395/load_peak_allocated | 3 | 3 | 0.0000 | 0.0000 |
| rtx3090__attn-default/first_measure_peak_reserved | 20 | 6 | 6.0968 | 30.6604 |
| rtx3090__attn-default/first_measure_peak_allocated | 20 | 0 | 6.0353 | 36.8100 |
| rtx3090__attn-default/after_load_reserved | 20 | 20 | 0.0000 | 0.0000 |
| rtx3090__attn-default/after_load_allocated | 20 | 20 | 0.0000 | 0.0000 |
| rtx3090__attn-default/load_peak_allocated | 20 | 20 | 0.0000 | 0.0000 |


이 수치는 baseline만 사용합니다. 허용 오차 기준을 사용자가 지정하지 않았으므로 임의의 PASS/FAIL을 붙이지 않았습니다. 같은 peak reserved가 나와도 dtype/요청열/allocated까지 같다고 판정하지 않습니다.

## 불일치의 영향과 비교 조건

| 대상 | 영향 | 비교에 필요한 정합 조건 |
| --- | --- | --- |
| CNN 6공통조건(ResNet18/50, MobileNet 각2) | reserved 및 allocated의 start/peak/end/released; 특히 r512 큰 차이 | 현재도 예측오차는 계산 가능. 기존 GPU trace의 cuDNN workspace·수명·알고리즘 및 CPU temp를 분리해 설명해야 동일 요청 모델로 볼 수 있음(Q003). |
| T5 4공통조건 | FP16 dtype 태그·캐스트 요청 및 allocated peak; reserved 동률이 상쇄일 수 있음 | 기존 GPU T5 trace에서 요청크기 차이와 실제 dtype/연산을 식별(Q004). |
| sdpa 모델 공통10조건 | workspace/CPU 내부 임시·커널 차이로 allocated/reserved 잔차 | 동일 sdpa 표기만으로 동등성 주장 금지. 기존 trace로 GPU-only 요청 수명·CPU emulation 차이를 대응(Q003/Q005). |
| 모든 CPU v2 조건 | FP32→FP16 emulation, 1trace×4 근사 | 기존 validation 범위 밖 일반화는 보류. 과거 emu/native 11건 중 전체 action/size 일치0, 8개 prediction byte 지표 모두 동률3건이라는 제한을 유지. |
| 미완료 CPU375 / GPU main measure322 | 수치 없음; 예상값을 실행값으로 대체 불가 | 기존 진행 중 작업이 자연 산출한 finalized result와 업데이트 manifest를 받은 후 교집합 재산출. |
| GPU eager 이전label | sdpa가 적용되는 모델의 중간 tensor·메모리 footprint 변화 | 현재 baseline에는 default label만 대응. 과거 eager는 별도 비교축으로 유지. |

## 상대 매니페스트의 INUOS 질문에 대한 현지 확인

정식 review_INUGPU.md는 아직 없으므로 reply_INUOS.md를 임의 생성하지 않았습니다. 아래는 GPU manifest의 open_questions를 선제 확인한 것이며, 정식 질문 파일이 오면 그 ID를 그대로 이어 사용합니다.

| GPU manifest 질문 위치 | 이 VM에서 확인한 답 | 현지 근거 |
| --- | --- | --- |
| open_questions[0],[1] prefix/cache/lifetime/reset | FP16 load 합성→warmup3→measure0 캐시 유지; invoke clone 및 M4 후 del; local peak baseline reset | cpu_worker2.py:45,67,215; protocol2.py:86 |
| [2] attention | 공통20 설정 같음, CPU kernel probe는 B1/텍스트2토큰 별도 진단 | result.attention_ops_cpu; cpu_worker2.py:92 |
| [3] fp16 미지원 | 현재 v2는 FP32 실행+shadow FP16 requests. 네이티브 FP16 성공으로 주장하지 않음 | cpu_worker2.py:186,255,297 |
| [4] 입력 | GPU 전체 SHA256이 새로 제공되어 공통20 shape/dtype/sha256 전체 일치로 갱신 | comparison_details.json field_checks |
| [5] 지표 | primary reserved, auxiliary allocated; measure0 start/peak/M4/M5 대응 | 위 지표표; cpu_worker2.py:280 |
| [6] workspace | baseline에 GPU workspace 없음. 별도8.125MiB variant만 존재 | cpu_worker2.py:291 |
| [7] 조건표 | 2634bda772c263fa52fd257605caa360f3256cb542d2ed18552ff2341d2fe258 동일 | manifest/conditions_main395.json |
| [8] 생성/revision/loader/포맷 | 395 요청 키 일치, 성공20 실제 loader class 대응 및 설정·길이 확인 | protocol2.py:21,29,73,86; result.json |
| [10] common eager 문구 | 현재 worker 인자 생략/default 및 결과 attn을 사용 | protocol2.py:29; result.model_info |
| [14] xMem 검증 범위 | GPU manifest가 명시한 기록된 필드만 검증 범위로 채택. 최초 CPU manifest의 모든 경계 일치로 읽힐 수 있는 인계 문구는 검증 범위 확대의 근거로 사용하지 않음 | gpu_manifest.run.xmem, open_questions[14] |

## 검토·답변 교환 상태와 다음 단계

- `review_INUOS.md`: 작성 완료. INUOS-Q001~Q007, 상대 관측값·현지값·근거·상대 확인 경로 포함.
- `review_INUGPU.md`: 미도착. 받으면 질문 ID별로 이 VM 코드·기존 로그를 재확인해 `reply_INUOS.md` 작성.
- `reply_INUGPU.md`: 미도착. 받으면 INUOS-Q001~Q007별 답변 근거를 대조하고 위 판정과 이 문서를 갱신.
- `comparison.md`: 현재는 양쪽 manifest만 반영한 잠정 종합본. 양쪽 review/reply 완료로 표시하지 않음.
- 진행 후 새 결과를 반영하려면 갱신한 `gpu_manifest.json` 및 CPU의 새 스냅샷도 필요. 기존 스냅샷 날짜와 섞지 않고 버전을 기록할 것.
- 실험 재실행, 기존 코드/설정 변경, 컨테이너/tmux 종료, 네트워크 전송 없음. 전달은 사용자가 수행.

# INUOS → INUGPU 교차 검토 질문

작성: 2026-11-12T11:58:49.531259+09:00. 내 역할 이름은 INUOS, hostname은 INUOSVM-1입니다. 파일명에는 역할 이름을 사용합니다.

입력: `/home/inuos/cgf-fp16-680-20261111/handoff_codex/cpu_manifest.json` SHA256 `57f4cdd6142e5a6d87037a905e53aa92b9a8d9432591a426732429d7108faa1f`; `/home/inuos/cloudgpufit-review/gpu_manifest.json` SHA256 `db3411fe91346bca8ad7bee0efe208df7cde3dd18c011f7911c03948c716bef0`. 원본 매니페스트는 수정하지 않았습니다. GPU 원격 경로는 전달 매니페스트가 인용한 근거이며 이 VM에서 직접 읽었다는 뜻이 아닙니다.

현재 범위: 본실험 actual-success 교집합 3건, 별도 default-attention 파일럿 actual-success 교집합 20건. 입력 전체 해시/shape/dtype·생성 설정·실제 생성 길이의 일치는 확인했습니다. EOS 조기 종료는 공통 텍스트 성공건에서 실제 길이가 요청과 같으므로 관측되지 않았습니다. 이미 확인된 내용을 재질문하지 않습니다.

상대 답변은 `reply_INUGPU.md`에 아래 ID를 그대로 사용해 주세요. 각 ID별 결론·코드/로그 경로·관측값·확인 시각·미확인 범위를 적어 주세요. 새 실험을 실행하라는 요청이 아닙니다.

## INUOS-Q001 — 체크포인트 실제 내용 해시

- 상태: 답변 대기
- INUOS 확인값: 모델 revision/use_safetensors 및 loader와 395개 요청 조합이 일치. INUOS 캐시 snapshot 경로·존재만 검사했고 checkpoint blob 내용 hash는 미검증.
- INUGPU 매니페스트 확인값: GPU models.json hash 7b11e3e…와 revision은 일치. GPU source inventory에는 model_cache_manifest.json hash가 있으나 파일별 checkpoint 내용 hash는 전달 manifest에 없음.
- 질문: 같은 revision이라는 사실 외에 실제 가중치 바이트의 일치를 확인할 기존 model_cache_manifest.json/model_cache_filelist.txt를 보내 주세요. SHA256이 아닌 HF ETag라면 그 의미도 구분해 주세요. 새 다운로드·실행은 요청하지 않습니다.
- 영향 범위: 13모델 전체; 실제 수치 비교 20조건의 모델 내용 provenance
- 현재 근거: `/home/inuos/cgf-fp16-680-20261111/manifest/models.json`; `/home/inuos/cgf-fp16-680-20261111/handoff_codex/cpu_manifest.json#/run/models`; `/home/inuos/cloudgpufit-review/gpu_manifest.json#/run/source_inventory`
- 상대가 확인할 코드/로그: `/home/inugpu/cgf-fp16-vram-20261111/config/model_cache_manifest.json`; `/home/inugpu/cgf-fp16-vram-20261111/config/model_cache_filelist.txt`; `/home/inugpu/cgf-fp16-vram-20261111/tools/prepare_cache.py`

## INUOS-Q002 — 본실험 validate와 FP16 유한성

- 상태: 답변 대기
- INUOS 확인값: CPU 20건 finite_check=not_assessable (FP32 execution). 실제 텍스트 생성 길이는 요청과 같음.
- INUGPU 매니페스트 확인값: 기본 attention 파일럿 validate 26/26 ok. main395 measure 73건, validate finalized 0/395건. 과거 eager Qwen validate 실패 2건은 현재 baseline과 분리됨.
- 질문: 진행 중인 작업이 자연스럽게 산출한 main395 validate 결과를 같은 label·condition_id로 전달해 주세요. raw logits hook과 all_finite/steps_checked/first_nonfinite_step를 포함하고, 기존 파일럿 검증을 본실험 검증으로 대체하지 말아 주세요.
- 영향 범위: 본실험 공통 3건 및 향후 finalized 조건; VRAM 비교와 출력 유효성 판정 분리
- 현재 근거: `/home/inuos/cgf-fp16-680-20261111/handoff_codex/cpu_manifest.json#/cases/*/finite_check`; `/home/inuos/cloudgpufit-review/gpu_manifest.json#/cases[label=rtx3090__main395,mode=validate]`; `/home/inuos/cloudgpufit-review/gpu_manifest.json#/cases[label=rtx3090__attn-default,mode=validate]`
- 상대가 확인할 코드/로그: `/home/inugpu/cgf-fp16-vram-20261111/runs/rtx3090__main395/*/validate.json`; `/home/inugpu/cgf-fp16-vram-20261111/src/vram_worker.py:301`

## INUOS-Q003 — GPU 전용 workspace와 실제 수치 잔차

- 상태: 답변 대기
- INUOS 확인값: 기본 예측에 GPU workspace 없음. 별도 variant는 warmup_0 시작에 8,519,680 B 영구 요청을 추가. 예: mobilenet b16/r512 reserved CPU=1161822208, resnet50 b16/r512 CPU=648019968 B.
- INUGPU 매니페스트 확인값: 동일 조건 GPU reserved=889192448 및 851443712 B. main OPT b16/512/512는 CPU=1583349760, GPU=1598029824 B. 입력·생성 설정·attention 값은 같음.
- 질문: 기존 trace에서 cuBLAS/cuBLASLt/cuDNN 할당을 연산·요청크기·수명·경계로 식별 가능한 근거를 보내 주세요. 모델별 workspace를 일괄 상수로 귀속할 근거는 아직 없습니다. trace 없는 조건은 미확인으로 답해 주세요.
- 영향 범위: 20개 성공 공통조건의 start/peak/end/released reserved/allocated; 특히 CNN 6조건 및 transformer 잔차
- 현재 근거: `/home/inuos/cgf-fp16-680-20261111/src/cgf680/cpu_worker2.py:42`; `/home/inuos/cgf-fp16-680-20261111/src/cgf680/cpu_worker2.py:291`; `/home/inuos/cloudgpufit-review/paired_metrics.csv`
- 상대가 확인할 코드/로그: `/home/inugpu/cgf-fp16-vram-20261111/results/attention_kernels_rtx3090_default.json`; `/home/inugpu/cgf-fp16-vram-20261111/runs/rtx3090__attn-default/*/trace.json`; `/home/inugpu/cgf-fp16-vram-20261111/runs/rtx3090__attn-default/*/trace.snapshot.pickle.gz`; `/home/inugpu/cgf-fp16-vram-20261111/runs/rtx3090__main395/*/measure.json`

## INUOS-Q004 — T5 요청 dtype 식별

- 상태: 답변 대기
- INUOS 확인값: 기존 t5-small 요청 차이: CPU 16KB 2304회 및 2MiB 12회; h/f/x inferred 태그·T5 branch patch 적용.
- INUGPU 매니페스트 확인값: 대응 GPU 크기 multiset에 8KB 2304회 및 1MiB 12회. 개별 할당의 연산·dtype는 multiset만으로 미확정.
- 질문: 이미 저장된 T5 trace·커널·코드에서 해당 요청의 실제 dtype과 생성 연산을 확인해 주세요. 서로 다른 크기가 같은 논리 텐서라는 추정도 검증해 주세요.
- 영향 범위: T5 공통4조건; 특히 allocated와 FP16 dtype 추적의 정합성
- 현재 근거: `/home/inuos/cgf-fp16-680-20261111/results/main395/gpu_trace_request_diff.jsonl:3`; `/home/inuos/cgf-fp16-680-20261111/src/cgf680/shadow.py:206`; `/home/inuos/cgf-fp16-680-20261111/src/cgf680/shadow.py:244`
- 상대가 확인할 코드/로그: `/home/inugpu/cgf-fp16-vram-20261111/runs/rtx3090__attn-default/t5-small__b8__in128__new128/trace.snapshot.pickle.gz`; `/home/inugpu/cgf-fp16-vram-20261111/src/vram_worker.py`

## INUOS-Q005 — 4회 trace 복제의 GPU 범위

- 상태: 답변 대기
- INUOS 확인값: 실제 CPU 1회 추론 trace를 warmup3+measure1로 복제. 기존 eager/v1 37건 action/size/lifetime 동일; v2 전체와 GPU 요청열 동일성은 미입증.
- INUGPU 매니페스트 확인값: GPU 실제 warmup3+measure5. 기본 attention 파일럿 성공20 공통조건의 measure 5회 reserved peak는 기록에서 검사 가능하지만 요청열 동일성은 별개.
- 질문: 기존 기본 attention trace 6개에서 warmup_0/1/2/measure_0의 요청 순서·크기·수명 비교가 이미 가능한지 확인해 주세요. 로드 이후 생성 캐시나 1회성 GPU 요청은 따로 분리해 주세요. 기존 xMem 검증은 기록된 필드만 답해 주세요.
- 영향 범위: 20공통조건의 warmed measure_0 예측; 전체395 일반화 제한
- 현재 근거: `/home/inuos/cgf-fp16-680-20261111/handoff_codex/cpu_manifest.json#/metrics/trace_replication_evidence`; `/home/inuos/cloudgpufit-review/gpu_manifest.json#/run/xmem`; `/home/inuos/cloudgpufit-review/gpu_manifest.json#/open_questions/14`
- 상대가 확인할 코드/로그: `/home/inugpu/cgf-fp16-vram-20261111/runs/rtx3090__attn-default/*/trace.json`; `/home/inugpu/cgf-fp16-vram-20261111/runs/rtx3090__attn-default/*/trace.snapshot.pickle.gz`; `/home/inugpu/cgf-fp16-vram-20261111/tools/xmem_replay.py`

## INUOS-Q006 — 스냅샷 진행 상태·생략 이벤트의 의미

- 상태: 답변 대기
- INUOS 확인값: 11:28:06 KST CPU snapshot: ok20/running1/not_started374; CPU per-case resume skip 로그 없음. 원본 manifest 고정.
- INUGPU 매니페스트 확인값: 11:25:38 KST GPU snapshot: main measure ok73/no-final-result322, validate no-final-result395. 기본 attention label skip 로그5건은 [1/6] 형식이며 mode 미명시.
- 질문: 기존 sessions/progress/log에서 GPU skip5건이 trace 재개인지 mode·호출을 확인해 주세요. no-final-result 중 in-flight는 기존 실행 기록으로 구분 가능하면 알려 주시고, 진행 이후 자연 생성 결과는 갱신 manifest로 전달해 주세요.
- 영향 범위: 상태 집계; skip 이벤트는 성공 결과를 무효화하거나 미실행으로 만들지 않음
- 현재 근거: `/home/inuos/cgf-fp16-680-20261111/handoff_codex/cpu_manifest.json#/run/status_snapshot`; `/home/inuos/cloudgpufit-review/gpu_manifest.json#/run/labels[label=rtx3090__attn-default]/skipped_resume_events`; `/home/inuos/cloudgpufit-review/gpu_manifest.json#/run/snapshot`
- 상대가 확인할 코드/로그: `/home/inugpu/cgf-fp16-vram-20261111/logs/*`; `/home/inugpu/cgf-fp16-vram-20261111/runs/rtx3090__attn-default/sessions.jsonl`; `/home/inugpu/cgf-fp16-vram-20261111/runs/rtx3090__attn-default/progress.jsonl`; `/home/inugpu/cgf-fp16-vram-20261111/runs/rtx3090__main395/sessions.jsonl`; `/home/inugpu/cgf-fp16-vram-20261111/runs/rtx3090__main395/progress.jsonl`

## INUOS-Q007 — AWS 실제 실행 증거

- 상태: 답변 대기
- INUOS 확인값: INUOS는 AWS 실행/결과를 보유하지 않음.
- INUGPU 매니페스트 확인값: GPU manifest aws-<instance_type>는 395조건×measure/validate=790 prepared 레코드, executed=0.
- 질문: 실제 AWS label·instance/GPU·cuda_total_memory·worker/config hash 및 기존 결과 회수 경로가 생기면 전달해 주세요. 준비 번들 정보와 실제 실행 관측을 분리해 주세요.
- 영향 범위: AWS 전체; 현재 수치 비교 범위에서 제외
- 현재 근거: `/home/inuos/cloudgpufit-review/gpu_manifest.json#/cases[label=aws-<instance_type>]`
- 상대가 확인할 코드/로그: `/home/inugpu/cgf-fp16-vram-20261111/aws/`; `/home/inugpu/cgf-fp16-vram-20261111/runs/aws-*/sessions.jsonl`; `/home/inugpu/cgf-fp16-vram-20261111/runs/aws-*/*/measure.json`

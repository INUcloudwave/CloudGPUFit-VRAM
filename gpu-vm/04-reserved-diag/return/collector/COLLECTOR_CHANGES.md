# inugpu 수집기 수정 내역 (gpu_collector_reference.py → gpu_collector.py)

- 기준: 받은 참조 sha256 513cbd80…66b6 → 실제 사용 gpu_collector.py (sha256는 collector/SHA256)
- 기록 의미(이벤트·boundaries·snapshot 형식)는 변경 없음. 추가/보강만
- mod 1: `--run-label-suffix` 추가 → 선택 run(stacks=all)을 별도 run_id 디렉터리로 분리. meta에 `manifest_run_id` 추가
- mod 2: run_meta 보강 — record_start_check, tf32 실제값, host(cpu_count·torch threads), container(image id·docker 명령), weights(repo+revision), 생성 설정, bench_worker_v3 대비 차이 목록
- mod 3: attention_mask sha도 manifest 기대값과 비교, 불일치 시 실패. input sha 일치 여부를 meta에 기록
- mod 4: finite_check에서 nonfinite면 실패 처리(bench_worker_v3와 동일하게 중단). 참조 코드는 계속 진행했음
- mod 5: 생성 토큰 수 불일치 시 관측값을 invocations.jsonl에 남긴 뒤 실패
- mod 6: 실패 invocation도 invocations.jsonl에 status와 함께 기록
- mod 7: 종료 시 trace 잘림 검사 — alloc/segment_alloc 이벤트 수 vs 누적 카운터 증가분, entries < max_entries → meta.trace_check, meta.truncated
- 런처: Docker(이미지 cloudgpufit-bench:cu124-v1, id sha256:457da255…, 이전 AWS image id와 동일), `--network none`, HF cache read-only, `python -I`, GPU UUID 지정, 실행 전 다른 GPU 프로세스 없음 확인
- run_one.sh.v1: 12개 기준·진단 run + 첫 stacks=all 시도에 사용
- run_one.sh: `TORCH_SYMBOLIZE_MODE` 전달만 추가 → `R1_controlled__stacks_all_dladdr` 재시도에만 사용(TORCH_SYMBOLIZE_MODE=dladdr)

# bench_worker_v3 대비 알려진 차이
- NVML 샘플러 스레드·idle/온도 대기 생략(allocator 미사용)
- bench_worker_v3는 두 번째(memory) 모드의 model_reuse 직전에도 reset_peak_memory_stats() 호출. 본 수집기는 요청서대로 model load 전·measure 블록 직전에만 호출 → peak 통계에만 영향
- tokens/components 모드 생략

# 선택 run 관련
- `R1_controlled__stacks_all`(실패, 보존): stacks="all" 심볼화가 addr2line을 fork/execvp로 호출, 이미지에 addr2line 없음
  - fork된 자식 프로세스 8개가 각각 예외 경로를 실행 → collector.log에 "failed execvp" 8회, invocations.jsonl 중복 9줄
  - 부모 프로세스는 `RuntimeError: stoi`로 memory_snapshot 저장 실패 → memory_snapshot.pickle 없음
- `R1_controlled__stacks_all_dladdr`(성공): TORCH_SYMBOLIZE_MODE=dladdr. 내보낸 심볼만 이름이 나오고 내부 심볼은 "??" (라이브러리 경로는 기록됨)
- `_precheck/`: 본 실행 전 점검용 fp16_01 R2 1회. 본 실행 R2와 별도

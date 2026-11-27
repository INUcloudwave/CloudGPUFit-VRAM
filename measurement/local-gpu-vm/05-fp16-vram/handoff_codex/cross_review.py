#!/usr/bin/env python3
"""Offline manifest comparison. Never imports torch or changes experiment inputs."""
from pathlib import Path
import json,hashlib,collections,csv,datetime
from zoneinfo import ZoneInfo
R=Path('/home/inugpu/cgf-fp16-vram-20261111');D=Path('/home/inugpu/cloudgpufit-review')
GP=R/'handoff_codex/gpu_manifest.json';CP=D/'cpu_manifest.json'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def val(x):return x.get('value') if isinstance(x,dict) and 'value' in x else x
G=load(GP);C=load(CP);now=datetime.datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
assert G['vm']['hostname']=='gpuvm-1'
# Call mode and measurement mode are distinct axes. FP16 is requested target, not CPU execution dtype.
def key(x,cpu=False):return (x['model'],x['revision'],x['requested_precision'] if cpu else x['precision_requested'],'measure',x['mode'] if cpu else x['call'],x['batch'],x.get('input_len'),x.get('max_new_tokens'),x.get('resolution'),x.get('use_safetensors'))
def keydict(k):return dict(zip(['model','revision','precision_target','measurement_role','call','batch','input_len','max_new_tokens','resolution','use_safetensors'],k))
CE={key(x,True):x for x in C['cases'] if x['prepared_or_executed']=='executed' and x['result_exists']}
CPENDING=[x for x in C['cases'] if not x['result_exists']]
ALLCPU={key(x,True):x for x in C['cases']}
GE=[x for x in G['cases'] if x['state']=='executed'];GM=[x for x in GE if x['mode']=='measure'];GSET={key(x) for x in GM}
primary='rtx3090__attn-default';P={key(x):x for x in GM if x['label']==primary}
strict_dtype_pairs=0
labelsets=[]
for l in G['run']['labels']:
 rows=[x for x in G['cases'] if x['label']==l['label']]
 executed=[x for x in rows if x['state']=='executed' and x['mode']=='measure'];ks={key(x) for x in executed}
 labelsets.append({'label':l['label'],'kind':l['kind'],'manifest_counts':l['counts'],'actual_measure_common':len(ks&CE.keys()),'actual_measure_gpu_only':len(ks-CE.keys()),'actual_measure_cpu_only':len(CE.keys()-ks),'common_keys':[keydict(k) for k in sorted(ks&CE.keys(),key=str)],'gpu_only_keys':[keydict(k) for k in sorted(ks-CE.keys(),key=str)],'cpu_only_keys':[keydict(k) for k in sorted(CE.keys()-ks,key=str)]})
metrics_map={'first_measure_start_reserved':('summary','first_measure_start_reserved'),'first_measure_start_allocated':('summary','first_measure_start_allocated'),'first_measure_peak_reserved':('summary','first_measure_peak_reserved'),'first_measure_peak_allocated':('summary','first_measure_peak_allocated'),'first_measure_end_with_output_reserved':('measures',0,'end_with_output','reserved'),'first_measure_end_with_output_allocated':('measures',0,'end_with_output','allocated'),'first_measure_released_reserved':('measures',0,'released','reserved'),'first_measure_released_allocated':('measures',0,'released','allocated'),'after_load_reserved':('after_load','reserved_bytes.all.current'),'after_load_allocated':('after_load','allocated_bytes.all.current'),'load_peak_allocated':('after_load','max_allocated')}
def get(x,path):
 try:
  for p in path:x=x[p]
  return x
 except (KeyError,IndexError,TypeError):return None
m4={z['condition_id']:z for z in val(C['metrics']['M4_to_M5_allocations_audit'])}
pairs=[]
for k in sorted(CE.keys()&P.keys(),key=str):
 c,g=CE[k],P[k];cpu_inputs=c['input_sha256'];gh={n:v['sha256'] for n,v in g['inputs'].items()};cg=val(c['generation_config']);gg=g['generation_config'];cfgdiff={n:{'cpu':(cg or {}).get(n),'gpu':(gg or {}).get(n)} for n in set(cg or {})|set(gg or {}) if (cg or {}).get(n)!=(gg or {}).get(n)}
 # Vision has no new token count; text length observation is per invocation output width.
 lens=[z.get('new_tokens') for z in g['warmup']+g['measures']]
 row={'condition_id':c['condition_id'],'key':keydict(k),'cpu_result':c['value_evidence']['status'],'gpu_result':g['evidence'],'cpu_precision_actual':c['applied_precision'],'gpu_precision_actual':g['precision_actual'],'inputs_equal_full_sha256':cpu_inputs==gh,'cpu_inputs_sha256':cpu_inputs,'gpu_inputs_sha256':gh,'generation_config_equal':cg==gg if c['mode']=='generate' else None,'generation_config_differences':cfgdiff,'cpu_actual_new_tokens':c['actual_new_tokens'],'gpu_actual_new_tokens_warmup_and_measure':lens,'requested_length_met':(c['actual_new_tokens']==c['max_new_tokens'] and all(z==c['max_new_tokens'] for z in lens)) if c['mode']=='generate' else None,'cpu_attention_config':c['attn_implementation'],'gpu_attention_config':g['attn_actual'],'cpu_attention_probe':val(c['attention_ops_cpu']),'cpu_trace_stats':val(c['trace_stats']),'cpu_fp16_emulation':val(c['fp16_emulation']),'M4_to_M5_audit':m4.get(c['condition_id']),'gpu_protocol':g['protocol'],'metrics':{},'gpu_validation':[]}
 for name,path in metrics_map.items():
  cv=(val(c['summary']) or {}).get(name);gv=get(g,path)
  row['metrics'][name]={'cpu':cv,'gpu':gv,'cpu_minus_gpu_bytes':cv-gv if cv is not None and gv is not None else None,'relative_error_percent':100*(cv-gv)/gv if cv is not None and gv else None,'equal':cv==gv if cv is not None and gv is not None else None,'mapping':'corresponding allocator metric; peak window equivalent only if M4→M5 has no allocs','evidence_level':'observed_in_results (CPU: imported manifest log_or_result, not independently read remote log)'}
 for v in GE:
  if v['label']==primary and v['mode']=='validate' and key(v)==k:row['gpu_validation'].append({'status':v['result_status'],'validation':v['validation'],'source':v['evidence']})
 pairs.append(row)
# Full state union retains all GPU labels/modes; prepared artifacts never become executions.
state_rows=[]
for c in C['cases']:
 state_rows.append({'side':'INUOS','label':val(C['run']['run_id']),'key':keydict(key(c,True)),'condition_id':c['condition_id'],'measurement_mode':'prediction_of_measure_0','raw_mode':c['mode'],'state':c['execution_state'],'prepared_or_executed':c['prepared_or_executed'],'result_status':c['status'],'failed_stage':c['failed_stage'],'skip_confirmed':c['skipped_on_resume'],'evidence':c['evidence_paths']})
for g in G['cases']:
 state_rows.append({'side':'INUGPU','label':g['label'],'key':{**keydict(key(g)),'measurement_role':g['mode']},'condition_id':g['condition_id'],'measurement_mode':g['mode'],'raw_mode':g['mode'],'state':g['state'],'result_status':g['result_status'],'failed_stage':g.get('failed_stage'),'error':g.get('error'),'evidence':g['evidence'],'attempts':g.get('attempts',[])})
files=sorted(p for p in D.glob('*.md') if p.name!='review_gpuvm-1.md')
peer_reviews=[p for p in files if p.name.startswith('review_')];peer_replies=[p for p in files if p.name.startswith('reply_') and p.name!='reply_gpuvm-1.md']
summary={'cpu':{'planned_workloads':len(C['cases']),'finalized':len(CE),'success':sum(c['status']=='ok' for c in CE.values()),'failed':sum(c['status']!='ok' for c in CE.values()),'running_without_result':sum(c['status']=='running' for c in CPENDING),'pending_not_running':sum(c['status']=='pending' for c in CPENDING),'prepared_only':sum(c['prepared_or_executed']=='prepared' for c in C['cases']),'skip_confirmed':sum(c['skipped_on_resume'] is True for c in C['cases']),'skip_unknown':sum(c['skipped_on_resume'] is None for c in C['cases'])},'gpu':{'executed_slots_all_modes':len(GE),'successful_slots':sum(x['result_status']=='ok' for x in GE),'failed_slots':sum(x['result_status']!='ok' for x in GE),'measure_slots':len(GM),'unique_measure_workloads':len(GSET),'prepared_slots':sum(x['state']=='prepared' for x in G['cases']),'unfinished_slots':sum(x['state']=='planned_not_yet_run' for x in G['cases'])},'unique_actual_workloads':{'common':len(GSET&CE.keys()),'gpu_only':len(GSET-CE.keys()),'cpu_only':len(CE.keys()-GSET)},'primary_pairs':len(pairs),'primary_full_input_sha_equal':sum(x['inputs_equal_full_sha256'] for x in pairs),'primary_generation_config_equal':sum(x['generation_config_equal'] is True for x in pairs),'primary_vision_not_applicable':sum(x['key']['call']=='forward' for x in pairs),'primary_text_cases':sum(x['key']['call']=='generate' for x in pairs),'primary_recorded_lengths_met':sum(x['requested_length_met'] is True for x in pairs),'strict_native_fp16_execution_pairs':strict_dtype_pairs,'scope':'Target precision is FP16; CPU physical execution is FP32 plus FP16 request-size emulation. Strict native execution dtype equality has zero pairs. GPU labels and modes remain separate.'}
questions=[]
def q(id,title,gpu,cpu,evidence,check,impact):questions.append({'id':id,'title':title,'gpu_value':gpu,'cpu_value':cpu,'evidence':evidence,'remote_code_or_log_to_check':check,'impact':impact,'status':'awaiting_INUOS_reply'})
root='/home/inuos/cgf-fp16-680-20261111'
q('INUGPU-Q01','단일 profiled invocation을 네 번 재생하는 것과 실제 warmup 3회의 동등성','각 프로세스에서 실제 load→warmup 3→measure 5; measure_0만 비교','FP32 invocation trace를 FP16 요청으로 바꾸어 cached Replayer에 4회 적용; 현재 main395의 iteration별 원본 실행 여부는 직접 확인 못함',['gpu_manifest.run.workload_protocol','cpu_manifest.metrics.trace_replication_evidence'],[root+'/src/cgf680/cpu_worker2.py:262',root+'/tools/check_replicated_replay.py',root+'/runs/main395/cases/<ID>/invocation_fp16_requests.npz'],'20개 공통 조건 전체: 첫 invocation과 warmed invocation의 요청·수명이 같아야 warmed peak 비교 의미가 성립. 과거 audit 37건을 현재 20건의 직접 증명으로 사용하지 말고 적용 범위를 명시해 달라.')
q('INUGPU-Q02','실제 생성 길이 기록 범위와 EOS 종료','공통 텍스트는 warmup 3+measure 5 모두 요청 N; min=max=N','공통 텍스트 actual_new_tokens=N, config 일치; 단일 scalar 길이의 관측 위치는 protocol2 근거',['crosscheck_gpuvm-1.json#/pairs/*/cpu_actual_new_tokens','crosscheck_gpuvm-1.json#/pairs/*/gpu_actual_new_tokens_warmup_and_measure'],[root+'/src/cgf680/protocol2.py:86',root+'/src/cgf680/protocol2.py:90',root+'/runs/main395/cases/<ID>/result.json'],'공통 텍스트 12건. scalar가 실제 output.shape에서 계산된 값인지, profiler invocation과 attention probe 중 어디서 나온 값인지 확인. batch별 padding/EOS 종료가 메모리에 영향을 줄 수 있다면 per-sequence 종료 기록 유무도 답해 달라. 현재 scalar 기준 조기 길이 축소는 관측되지 않음.')
q('INUGPU-Q03','FP16 변환 태그와 고정 FP32 임시 텐서 크기','native torch.float16 파라미터·CUDA 실행','FP32 실행 + h/f/x/b 및 inferred 태그로 요청 크기 변환; 과거 native 비교 11건 중 전체 event sequence 일치 0건',['cpu_manifest.metrics.dtype_emulation','cpu_manifest.metrics.emulated_vs_native_summary'],[root+'/src/cgf680/shadow.py:123',root+'/src/cgf680/shadow.py:206',root+'/src/cgf680/shadow.py:244',root+'/runs/main395/cases/<ID>/result.json#/fp16_emulation'],'20개 전체의 peak allocated/reserved. inferred_h/f/x의 실제 dtype을 증명하는 저장 근거와 T5 8KiB↔16KiB, 1MiB↔2MiB 차이의 연산 대응을 확인. 미지원 연산 처리 정책이 dtype 승격/대체/에뮬레이션 중 무엇인지 명시.')
q('INUGPU-Q04','실제 측정 shape의 CPU attention 경로','파일럿 GPU SDPA 진단에서 CUDA FlashAttention; T5 eager; CNN attention 없음','CPU 별도 B=1/텍스트 N=2 probe는 aten::_scaled_dot_product_flash_attention_for_cpu, T5 bmm/softmax; 전체 B/N probe 아님',['gpu_manifest.run.attention_diagnostics','cpu_manifest.metrics.attention.cpu_probe_scope'],[root+'/src/cgf680/cpu_worker2.py:92',root+'/runs/main395/cases/<ID>/result.json#/attention_ops_cpu'],'SDPA 텍스트·ViT 및 T5 peak 지표. 기존 로그에서 실제 profiled invocation의 aten 경로를 식별 가능한지, probe를 전체 실행 경로로 확장 해석하지 않는지 답해 달라. 새 프로파일링은 요구하지 않음.')
q('INUGPU-Q05','measure peak의 M4/M5 경계와 입력 복사·출력 해제','start 후 reset; M4에 peak/end 읽고 CPU 출력 복사 후 del gi,o→released','CPU peak M0~M5; 공통20개 M4→M5 allocations=0 기록, input clone은 profiled region 안',['gpu worker src/vram_worker.py:343-367','cpu_manifest.metrics.M4_to_M5_allocations_audit'],[root+'/src/cgf680/cpu_worker2.py:45',root+'/src/cgf680/cpu_worker2.py:67',root+'/src/cgf680/protocol2.py',root+'/runs/main395/cases/<ID>/invocation_fp16_requests.npz'],'현재20개는 free-only 구간 확장으로 peak 비교 가능. M4/M5 marker 자체 제외 여부, start model residency와 input clone 포함 여부, 이후 조건에도 0 alloc 검사를 적용하는지 확인.')
q('INUGPU-Q06','load 합성과 checkpoint 바이트 동일성','실제 pinned checkpoint from_pretrained; 로컬 model_cache_filelist.txt 및 manifest 있음','동일 models/conditions SHA; FP16 load 합성; 원격 checkpoint bytes 동일성은 unverified',['gpu_manifest.run.source_inventory','cpu_manifest.run.condition_config_checks','cpu_manifest.run.models'],[root+'/src/cgf680/load_emu.py:60',root+'/hf_cache/hub/models--*/snapshots/<revision>/*',root+'/runs/main395/cases/<ID>/result.json#/load_synthesis'],'모든 조건 after_load 및 warmed cache. 체크포인트 내용 SHA 대조 자료가 이미 있으면 제공하고, 합성 load 순서·tied 중복·keep_fp32 처리의 기존 근거를 확인. revision 동일만으로 바이트 동일을 확정하지 않음.')
q('INUGPU-Q07','GPU workspace 보정 변형의 사용 여부','CUDA 요청 포함; NVML은 allocator 외부 메모리도 포함하므로 별도','전달 summary는 emulated allocator 수치. GPU request multiset 네 조건 모두 불일치; workspace 보정 변형이 primary summary에 포함되는지 확인 필요',['cpu_manifest.metrics.gpu_request_differences','crosscheck_gpuvm-1.json#/pairs/*/metrics'],[root+'/src/cgf680/cpu_worker2.py',root+'/results/main395/gpu_trace_request_diff.jsonl',root+'/runs/main395/cases/<ID>/result.json'],'20개 first_measure_peak allocated/reserved. cuBLAS/cuBLASLt/cuDNN 요청 주입량·시점·수명과 baseline/보정값 열 이름을 구별해 달라. 차이를 모두 workspace로 귀속할 근거는 아직 없음.')
q('INUGPU-Q08','진행·실패·재개 skip 및 이후 결과 범위','GPU snapshot 11:25, main395 measure 73 완료; validate395 미완료','CPU snapshot 11:28, 20 ok/1 running/374 pending; skipped_on_resume 395개 unknown',['gpu_manifest.run.labels','cpu_manifest.run.status_snapshot'],[root+'/runs/main395/suite_invocations.jsonl',root+'/runs/main395/attempts.jsonl',root+'/runs/main395/cases/*/result.json'],'아직 완료하지 않은 조건을 실패/skip으로 세지 않음. 기존 기록에서 실제 skip 확인이 가능한지, 최신 manifest를 교환할 때 snapshot과 superseded run을 구분해 달라.')
q('INUGPU-Q09','CPU validate 의미와 FP16 출력 유한성','default 파일럿 validate26 ok; eager Qwen2개 nonfinite_output','finite_check=not_assessable (FP32 execution); 별도 validate 모드 없음',['gpu_manifest.cases[mode=validate]','cpu_manifest.metrics.validation_limitations'],[root+'/src/cgf680/cpu_worker2.py:297',root+'/runs/main395/cases/<ID>/result.json#/finite_check'],'CPU status ok는 native FP16 출력 정상의 증거가 아님을 최종 comparison.md에 유지할지 확인. GPU eager 실패는 default 예측 비교와 분리.')
artifact={'created_kst':now,'inputs':{'gpu':{'path':str(GP),'sha256':sha(GP),'snapshot':G['run']['snapshot']},'cpu':{'path':str(CP),'sha256':sha(CP),'snapshot':C['vm']}},'normalization':{'key_axes':list(keydict(next(iter(CE)))),'cpu_mode':'CPU mode forward/generate maps to call; prediction_of_measure_0 maps semantically to GPU measure. GPU validate and trace are independent, not extra CPU prediction matches.','precision':'Match requested FP16 target only; always retain physical dtype difference. Native fp16 execution strict intersection=0.','checkpoint':'model+revision+use_safetensors matched; checkpoint byte identity unverified.','attention':'retain label-specific eager/default; do not mix eager with default primary comparisons.','source_scope':'Only imported manifests are used for cross-VM numerical counts; no live later GPU results silently added. CPU remote paths are citations, not locally opened files.'},'summary':summary,'per_label':labelsets,'pairs':pairs,'case_state_inventory':state_rows,'questions':questions,'received_peer_reviews':[{'name':p.name,'sha256':sha(p)} for p in peer_reviews],'received_peer_replies':[{'name':p.name,'sha256':sha(p)} for p in peer_replies],'expected_files':['review_INUOSVM-1.md','reply_INUOSVM-1.md'],'comparison_owner':'INUOS Codex; comparison.md intentionally not authored by INUGPU','resume':'Read same inputs and reviews/replies next turn. Preserve INUGPU-Q IDs. reply_gpuvm-1.md must answer incoming peer IDs after local source/log verification, not invent answers before review arrives.'}
# Validate pairing and every numerical result before writing.
assert len(pairs)==20 and len(CE)==20
assert len({x['condition_id'] for x in pairs})==20
assert all(x['requested_length_met'] for x in pairs if x['key']['call']=='generate')
assert {key(g) for g in G['cases'] if g['label']=='rtx3090__main395' and g['mode']=='measure'}==ALLCPU.keys()
assert all(x['M4_to_M5_audit']['allocations_M4_to_M5']==0 for x in pairs)
(D/'crosscheck_gpuvm-1.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2)+'\n')
with (D/'case_pairs_gpuvm-1.tsv').open('w',newline='') as f:
 w=csv.writer(f,delimiter='\t');w.writerow(['condition_id','gpu_label','metric','cpu_bytes','gpu_bytes','cpu_minus_gpu_bytes','relative_error_percent','input_sha256_equal','generation_config_equal','requested_length_met','cpu_evidence','gpu_evidence'])
 for p in pairs:
  for n,v in p['metrics'].items():w.writerow([p['condition_id'],primary,n,v['cpu'],v['gpu'],v['cpu_minus_gpu_bytes'],v['relative_error_percent'],p['inputs_equal_full_sha256'],p['generation_config_equal'],p['requested_length_met'],';'.join(p['cpu_result']),p['gpu_result']])
md=[]
def add(s=''):md.append(s)
add('# INUGPU 교차 검토 — gpuvm-1')
add();add('현재 비교 가능한 범위: GPU default 파일럿의 실제 measure 결과와 CPU main395의 완료 예측 20조건입니다. 요청 FP16·모델 revision·checkpoint 형식·배치·입출력·호출 조건을 맞춘 allocator 예측 오차 비교이며, CPU native FP16 실행과의 동등성 검증은 아닙니다. 상대 검토·답변이 없어 판단은 잠정입니다. INUOS Codex가 최종 comparison.md를 작성합니다.')
add();add(f'작성: {now}. GPU 매니페스트: `{GP}` (SHA256 `{sha(GP)}`), CPU: `{CP}` (SHA256 `{sha(CP)}`). GPU snapshot 11:25 KST / CPU snapshot 11:28 KST; 실행 중인 실험의 이후 결과를 혼합하지 않았습니다. 코드 변경·실험 재실행·VM 간 전송 없음.')
add();add('## 집계 기준과 범위');add();add('키 = 모델 ID × revision × checkpoint 형식(use_safetensors) × 요청 정밀도 × 측정 역할 × 호출 종류 × 배치 × 입력 길이 × 새 토큰 길이 × 해상도. CPU mode는 forward/generate, GPU mode는 measure/validate/trace여서 별도 축으로 정규화했습니다. 실제 실행 dtype을 엄격히 같게 요구하면 공통 native FP16 실행은 0건입니다. checkpoint 파일 내용 동일성은 미확인입니다.')
add();add('```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```')
add();add('| GPU label | 종류 | 실제 measure 공통 | GPU만 실행 | CPU만 실행 |');add('|---|---|---:|---:|---:|')
for l in labelsets:
 if l['kind']=='prepared':continue
 add(f"| {l['label']} | {l['kind']} | {l['actual_measure_common']} | {l['actual_measure_gpu_only']} | {l['actual_measure_cpu_only']} |")
add();add('공통/단독은 완료 파일이 있는 실행끼리만 계산합니다. GPU만 실행은 CPU 계획에 없는 경우와 CPU 아직 미완료인 경우를 모두 포함하며, 전체 키 목록·상태는 `crosscheck_gpuvm-1.json`의 per_label 및 case_state_inventory에 있습니다. GPU main395와 CPU main395 계획 자체는 395조건 동일합니다. AWS 790슬롯은 준비분이며 실행 비교에서 제외합니다. CPU pending 375개 중 1개는 실행 중이며 prepared로 바꾸어 세지 않았습니다. 이전 emu-validate/fp32-calib 등의 결과는 현재 cases 실행집합에 섞지 않고 보조 근거로만 사용합니다.')
add();add('GPU 실패는 eager Qwen validate 2건(nonfinite_output)이며 measure 실패가 아닙니다. GPU smoke에는 보존된 과거 재시도 1건이 있고 최신 결과는 ok입니다. default 파일럿 skip 로그 5회는 재개 이벤트이며 고유 조건 실패/미실행과 중복 합산하지 않습니다. CPU skip은 전 케이스 unknown이므로 확인된 skip 0건을 “skip이 없었다”로 해석하지 않습니다.')
add();add('## 항목별 잠정 판단');add();add('| 항목 | 분류 | 근거와 영향 |');add('|---|---|---|')
for r in [
('모델·revision·설정표','일치','공통20조건의 키가 동일; models.json 및 conditions_main395.json SHA 동일. weight blob 내용 검증은 별도 미확인.'),
('입력 생성·전체 SHA256','일치',f"공통 {len(pairs)}조건 중 전체 input SHA 일치 {sum(p['inputs_equal_full_sha256'] for p in pairs)}건. seed20261006/특수토큰→3/이미지 FP32→FP16 반올림 규칙도 문서상 일치."),
('GenerationConfig·실제 길이','일치',f"텍스트 {summary['primary_text_cases']}건의 전체 generation_config 비교 및 관측 길이를 case_pairs/JSON에 저장. GPU warmup3+measure5의 N 확인. CPU scalar의 관측 범위는 Q02 확인 대기."),
('정밀도','의도된 CPU·GPU 차이','GPU native FP16; CPU FP32 execution + FP16 request-size emulation. inferred dtype/고정 FP32 임시 텐서의 차이는 결과에 영향을 줄 수 있는 불일치로 추가 검토(Q03).'),
('Attention','의도된 CPU·GPU 차이','SDPA config는 일치하나 CPU FlashAttention-for-CPU와 CUDA FlashAttention은 다른 구현. probe는 전체 배치·생성 길이를 증명하지 않음(Q04).'),
('Warmup/cache','확인 불가','GPU는 실제 warmup3, CPU는 요청 trace 반복 재생. 캐시 유지·k=3 비교 의도는 일치; 요청 반복 동등성의 현재 케이스별 근거 범위 Q01 확인 대기.'),
('측정 시작/종료·peak','일치','CPU start M0, end_with_output M4, released M5 대응. peak는 CPU M0~M5/GPU M0~M4로 경계 차이가 있으나 현재20건 모두 M4→M5 alloc=0이므로 peak 구간은 비교 가능(Q05).'),
('Load 모델링','확인 불가','GPU 실제 loader vs CPU 합성 load. bytes 값은 비교 가능하나 요청 순서·checkpoint blob 동일성의 전체 증명은 부족(Q06).'),
('요청 trace/워크스페이스','결과에 영향을 줄 수 있는 불일치','CPU 매니페스트의 OPT/GPT2/T5-small/ResNet50 4조건 request multiset 모두 다름. 이를 모두 workspace 차이로 귀속 불가. 두 peak 지표와 warmed cache에 영향(Q03/Q07).'),
('FP16 출력 validation','의도된 CPU·GPU 차이','GPU 별도 validate; CPU FP32 finite로 FP16 정상성 입증 불가(Q09).'),
('아직 미완료 조건·AWS 실행','확인 불가','실행된 결과끼리만 비교. 이후 완료 결과·AWS 계획을 성공/실패로 추정하지 않음(Q08).')]:add('| '+' | '.join(r)+' |')
add();add('## 지표 대응과 실제 수치');add();add('CPU summary.first_measure_peak_reserved ↔ GPU summary.first_measure_peak_reserved, allocated도 동명 대응이 직접적인 예측 오차 비교 대상입니다. start/end/released는 동일 경계의 allocator 상태로 대응하고 load 지표는 합성 load 정확도 비교로 분리합니다. GPU median/max over 5, NVML process peak, CPU container RSS/mem_peak는 이 CPU first_measure 값과 직접 대응하지 않습니다. 대응 가능한 지표라는 사실은 값의 일치나 요청 trace 동등성을 뜻하지 않습니다. 수치는 전부 bytes, delta=CPU−GPU입니다.')
add();add('| 조건 | CPU peak reserved | GPU peak reserved | Δ reserved | CPU peak allocated | GPU peak allocated | Δ allocated |');add('|---|---:|---:|---:|---:|---:|---:|')
for p in pairs:
 r=p['metrics']['first_measure_peak_reserved'];a=p['metrics']['first_measure_peak_allocated'];add(f"| {p['condition_id']} | {r['cpu']} | {r['gpu']} | {r['cpu_minus_gpu_bytes']:+d} | {a['cpu']} | {a['gpu']} | {a['cpu_minus_gpu_bytes']:+d} |")
add();add('결과에 영향을 줄 수 있는 불일치: 공통20건에서 peak reserved는 6건 일치/14건 불일치, peak allocated는 20건 모두 불일치합니다. 합성 after_load reserved/allocated 및 load peak allocated는 20건 모두 수치 일치합니다. 따라서 load 값이 같아도 warmed invocation 요청 동등성이 입증되는 것은 아닙니다. 대표적으로 MobileNet b16/r512는 reserved +272,629,760 B, allocated +194,906,112 B; ResNet50 b16/r512는 reserved −203,423,744 B입니다. 이 차이의 원인 귀속은 Q03/Q04/Q07 확인 전까지 유보합니다.')
add();add('11개 대응 지표의 수치·오차·원본 경로는 `case_pairs_gpuvm-1.tsv`(220행)에, 전체 상태 2,105행은 `crosscheck_gpuvm-1.json`에 보존했습니다. 차이가 있는 케이스와 지표는 각 metric의 equal=false 및 delta로 식별합니다. 비교를 native-FP16 동등성 주장으로 확대하려면 Q01/Q03/Q04/Q06/Q07의 기존 근거를 먼저 확인하고, workload 요청·수명·dtype 보정과 kernel/workspace 모델의 차이를 명시해야 합니다. 이번 단계에서는 구현 변경이나 재실험을 요구하지 않습니다.')
add();add('## INUOS 확인 질문');add();add('다음 ID를 reply_INUOSVM-1.md에서 그대로 사용해 답해 주세요. CPU 경로는 상대 manifest에 실린 인용이며 이 VM에서 원격 파일을 직접 읽었다는 뜻이 아닙니다.')
for x in questions:
 add();add('### '+x['id']+' — '+x['title']);add();add('- GPU 확인값: '+x['gpu_value']);add('- CPU 확인값: '+x['cpu_value']);add('- 현재 근거: '+ '; '.join(x['evidence']));add('- 상대 확인 코드·로그: '+ '; '.join('`'+p+'`' for p in x['remote_code_or_log_to_check']));add('- 질문·영향: '+x['impact']);add('- 상태: 답변 대기')
add();add('## 다음 교환과 재개');add();add('현재 상대 review/reply 파일이 없습니다. 다음에 받을 파일은 `review_INUOSVM-1.md`(상대가 GPU에 묻는 질문)와 `reply_INUOSVM-1.md`(위 INUGPU-Q01~Q09에 대한 답변)입니다. 다른 파일명이라도 VM 식별자가 명확하면 인식할 수 있습니다. 상대 review 도착 후 로컬 코드·로그를 재확인해 상대 질문 ID 그대로 `reply_gpuvm-1.md`를 작성합니다. 상대 reply 도착 후 이 문서의 잠정 판단과 crosscheck 상태를 갱신합니다. 최종 `comparison.md`는 INUOS Codex가 양쪽 review/reply를 모아 작성합니다. 받지 않은 질문에 대한 reply 파일은 만들지 않았습니다.')
(D/'review_gpuvm-1.md').write_text('\n'.join(md)+'\n')
state={'input_hashes':{'gpu':sha(GP),'cpu':sha(CP)},'created_kst':now,'own_review':'review_gpuvm-1.md','own_reply':None,'question_ids':[q['id'] for q in questions],'pending_peer_files':artifact['expected_files'],'received_peer_reviews':artifact['received_peer_reviews'],'received_peer_replies':artifact['received_peer_replies'],'comparison_owner':'INUOS','phase':'awaiting_peer_review_and_reply'}
(D/'review_state_gpuvm-1.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2));print('PAIR_DIFFS',[(n,sum(p['metrics'][n]['equal'] for p in pairs)) for n in metrics_map]);print('FILES',[p.name for p in D.iterdir()])

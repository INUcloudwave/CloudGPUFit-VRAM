#!/usr/bin/env python3
"""Read-only audit of experiment inputs; writes only beside this script. No torch import or GPU work."""
import ast, collections, datetime, hashlib, json, re, subprocess
from pathlib import Path
from zoneinfo import ZoneInfo
R=Path('/home/inugpu/cgf-fp16-vram-20261111'); OUT=R/'handoff_codex'
START=datetime.datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
OBS='observed_in_results'; CODE='code_only'; UN='unverified'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def ev(value, source, level=OBS): return {'value':value,'source':str(source),'evidence_level':level}
def lines(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
worker=R/'src/vram_worker.py'; wt=worker.read_text(); wl=wt.splitlines()
def loc(s,p=worker):
 for i,line in enumerate(p.read_text().splitlines(),1):
  if s in line:return f'{p}:{i}'
 raise ValueError(s)
models=read(R/'config/models.json'); full=read(R/'config/conditions.json')['conditions']; main=read(R/'config/conditions_main395.json')['conditions']; cmap={c['condition_id']:c for c in full}
inventory={}
def inventory_file(p):
 if p.is_file(): inventory[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size,'evidence_level': CODE if any(x in p.parts for x in ['src','config','aws']) else OBS}
for sub in ['src','config','aws']:
 for p in (R/sub).glob('*'):
  if p.is_file():inventory_file(p)
for p in (R/'logs').glob('*sha*'):inventory_file(p)
inventory_file(R/'logs/SHA256SUMS_code_config_results.txt')
for p in R.glob('run*.sh'):inventory_file(p)
def selected(s):
 a=s['argv']; arg=a[a.index('--list')+1]
 cf=s.get('conditions_file','conditions.json')
 if arg=='all':return [c['condition_id'] for c in read(R/'config'/cf)['conditions']]
 if arg=='pilot':return read(R/'config/pilot.json')['condition_ids']
 if arg.endswith('.json'):return read(R/arg)['condition_ids']
 return arg.split(',')
# Freeze result membership before parsing; active experiment continues independently.
label_dirs=sorted(p for p in (R/'runs').iterdir() if p.is_dir())
snapshot={d.name:sorted(p for p in d.glob('*/*.json') if p.name in ['measure.json','validate.json','trace.json']) for d in label_dirs}
raw={str(p):read(p) for ps in snapshot.values() for p in ps}
manifest={'vm':{},'run':{},'environment':{},'cases':[],'metrics':[],'open_questions':[]}
qs=manifest['open_questions']
def question(q,why,to='INUOS',evidence=None,level=UN): qs.append({'to':to,'question':q,'why':why,'evidence':evidence or [],'evidence_level':level})
label_notes={'rtx3090':('run_pilot.sh','logs/pilot.txt'),'rtx3090__attn-default':('run_pilot_attn_default.sh','logs/pilot_v2_attn_default.txt'),'rtx3090__main395':('run_main395_vm.sh','logs/main395_rtx3090.txt'),'rtx3090__main395-pathcheck':(None,'logs/main395_pathcheck.txt')}
labels=[]
for d in label_dirs:
 label=d.name;sessions=lines(d/'sessions.jsonl'); plan={}; selection_evidence={}
 for i,s in enumerate(sessions):
  for cid in selected(s):plan[(cid,s['mode'])]=True;selection_evidence[(cid,s['mode'])]=f'{d}/sessions.jsonl:{i+1}'
 if label=='rtx3090__main395':
  for c in main:
   for mode in ['measure','validate']:
    key=(c['condition_id'],mode);plan[key]=True;selection_evidence.setdefault(key,loc('for M in measure validate',R/'run_main395_vm.sh'))
 results={(p.parent.name,p.stem):(p,raw[str(p)]) for p in snapshot[label]}
 plan.update({k:True for k in results})
 script,log=label_notes.get(label,(None,None)); cs=[]
 for (cid,mode) in sorted(plan):
  pair=results.get((cid,mode)); p,a=pair if pair else (d/cid/f'{mode}.json',{})
  c=a.get('condition',cmap[cid]);pin=models['models'][c['model']];m=a.get('model',{});measures=a.get('measures',[])
  src=str(p) if pair else selection_evidence[(cid,mode)]
  level=OBS if pair else CODE
  x={'label':label,'condition_id':cid,'state':'executed' if pair else 'planned_not_yet_run','model':c['model'],'revision':m.get('revision',pin['revision']), 'precision_requested':'fp16','precision_actual':m.get('param_dtypes'),'attn_actual':m.get('attn_implementation'),'mode':mode,'batch':c['batch_size'],'input_len':c.get('input_len'),'max_new_tokens':c.get('max_new_tokens'),'actual_new_tokens':[z.get('new_tokens') for z in measures] if measures else None,'resolution':c.get('resolution'),'call':'forward' if c['workload_type']=='vision_cls' else 'generate','result_status':a.get('status') if pair else 'not_yet_run','failed_stage':a.get('failed_stage'),'error':a.get('error'),'evidence':src,'evidence_level':level,
  'loader':m.get('loader',pin['loader']),'use_safetensors':m.get('use_safetensors',pin['use_safetensors']),'attn_requested':m.get('attn_requested'),'sdpa_backends_enabled':m.get('sdpa_backends_enabled'),'inputs':a.get('inputs'),'generation_config':a.get('generation_config'),'special_tokens_replaced_by_3':a.get('special_tokens_replaced_by_3'),'protocol':a.get('protocol'),'validation':a.get('validation'),'summary':a.get('summary'),'after_load':a.get('after_load'),'warmup':a.get('warmup'),'measures':measures if pair else None,'history':a.get('history'),'code_sha256':{k:a.get(k) for k in ['worker_sha256','models_json_sha256','conditions_json_sha256']},'attempts':[]}
  # Every case field has a distinct evidence grade, including missing observed values.
  x['field_evidence']={k:{'source':src,'evidence_level':level if v is not None else UN} for k,v in x.items() if k not in ['evidence','evidence_level']}
  for k in ['revision','loader','use_safetensors']:
   if k not in m:x['field_evidence'][k]={'source':str(R/'config/models.json'),'evidence_level':CODE}
  x['field_evidence']['call']={'source':loc('def invoke():'),'evidence_level':CODE}
  x['field_evidence']['precision_requested']={'source':str(p)+'#/argv' if pair else loc('model = cls.from_pretrained'),'evidence_level':OBS if a.get('model') else CODE}
  for ap in sorted((d/cid/'attempts').glob(f'*__{mode}.json')):
   ar=read(ap);inventory_file(ap);x['attempts'].append({'path':str(ap),'status':ar.get('status'),'failed_stage':ar.get('failed_stage'),'error':ar.get('error'),'worker_sha256':ar.get('worker_sha256'),'evidence_level':OBS})
  lp=d/cid/f'{mode}.launch.json'
  if lp.exists():x['launch']=read(lp);x['launch']['evidence_level']=OBS;inventory_file(lp)
  if pair:inventory_file(p)
  cs.append(x)
 progress=lines(d/'progress.jsonl')
 logpath=R/log if log else None
 skipped=[]
 if logpath and logpath.exists():
  for line in logpath.read_text().splitlines():
   if re.search(r'\[\d+/\d+\] skip ',line):skipped.append(line)
 counts={'planned_unique_condition_mode':len(cs),'planned_conditions':len({x['condition_id'] for x in cs}),'executed':sum(x['state']=='executed' for x in cs),'success':sum(x['result_status']=='ok' for x in cs),'failed':sum(x['state']=='executed' and x['result_status']!='ok' for x in cs),'planned_not_yet_run':sum(x['state']=='planned_not_yet_run' for x in cs),'skipped_resume_log_events':len(skipped),'historical_attempt_files':sum(len(x['attempts']) for x in cs),'by_mode':{mode:dict(collections.Counter(x['result_status'] for x in cs if x['mode']==mode)) for mode in ['measure','validate','trace']},'evidence_level':OBS,'note':'Unique condition × mode totals; retries and resume events are separate. Planned membership may be code_only; completed files are observed.'}
 byhash={k:sorted({a[k] for p in snapshot[label] if (a:=raw[str(p)]).get(k)}) for k in ['worker_sha256','models_json_sha256','conditions_json_sha256']}
 info={'label':label,'kind':'executed','sessions':sessions,'script':str(R/script) if script else None,'log':str(logpath) if logpath else None,'code_sha256':byhash,'session_code_sha256':[s.get('code_sha256') for s in sessions],'conditions_file':sorted({s.get('conditions_file','conditions.json') for s in sessions}),'attn':sorted({x['attn_requested'] for x in cs if x['attn_requested']}),'counts':counts,'progress_events':progress,'skipped_resume_events':skipped,'evidence':[str(d/'sessions.jsonl'),str(d/'progress.jsonl')],'evidence_level':OBS,'field_evidence':{'script':CODE if script else UN,'conditions_file':OBS if any(s.get('conditions_file') for s in sessions) else CODE,'attn':OBS if any(x['attn_requested'] for x in cs) else UN}}
 labels.append(info);manifest['cases'].extend(cs)
 for p in [d/'sessions.jsonl',d/'progress.jsonl']:inventory_file(p)
# AWS is one template deployment; no instance type or remote execution is asserted.
bundle=Path('/home/inugpu/cgf-fp16-aws-bundle-20261111/cgf-fp16-vram-20261111')
awslabel='aws-<instance_type>'
for c in main:
 pin=models['models'][c['model']]
 for mode in ['measure','validate']:
  x={'label':awslabel,'condition_id':c['condition_id'],'state':'prepared','model':c['model'],'revision':pin['revision'],'precision_requested':'fp16','precision_actual':None,'attn_actual':None,'mode':mode,'batch':c['batch_size'],'input_len':c.get('input_len'),'max_new_tokens':c.get('max_new_tokens'),'actual_new_tokens':None,'resolution':c.get('resolution'),'call':'forward' if c['workload_type']=='vision_cls' else 'generate','result_status':'execution_unverified','failed_stage':None,'error':None,'evidence':str(bundle/'config/conditions_main395.json'),'evidence_level':CODE,'attn_requested':'default','loader':pin['loader'],'use_safetensors':pin['use_safetensors']}
  x['field_evidence']={k:{'source':x['evidence'],'evidence_level':CODE if v is not None else UN} for k,v in x.items()};manifest['cases'].append(x)
labels.append({'label':awslabel,'kind':'prepared','sessions':[],'script':str(bundle/'aws/RUN_ON_AWS.sh'),'log':None,'code_sha256':{n:sha(bundle/'src'/n) for n in ['vram_worker.py','run_conditions.py']},'conditions_file':str(bundle/'config/conditions_main395.json'),'attn':'default','counts':{'planned_conditions':395,'planned_unique_condition_mode':790,'prepared':790,'executed':0,'success':0,'failed':0,'evidence_level':CODE},'evidence':[str(bundle/'aws/aws_03_pilot.sh'),str(bundle/'aws/aws_04_main.sh')],'evidence_level':CODE,'notes':['Deployment template; actual instance type and AWS results unverified.','Pilot 26 × 2 then main 395 × 2; pilot results reused within same label, giving 790 unique slots.','CGF_MAIN_VALIDATE=0 optionally omits main validation.']})
# Environment variations are captured across every completed result, not from the host Python.
envs=[(p,a['environment']) for p,a in raw.items() if a.get('environment')]
def variants(key,nested=False):
 vals={}
 for p,e in envs:
  v=e.get('env',{}).get(key) if nested else e.get(key)
  ident=json.dumps(v,sort_keys=True);vals.setdefault(ident,{'value':v,'count':0,'source':p,'evidence_level':OBS})['count']+=1
 return list(vals.values())
manifest['environment']={k:variants(v) for k,v in {'torch':'torch','transformers':'transformers','cuda':'torch_cuda','driver':'driver','python':'python','cudnn':'cudnn','allocator_backend':'allocator_backend','tf32_matmul':'tf32_matmul','tf32_cudnn':'tf32_cudnn','cudnn_benchmark':'cudnn_benchmark','threads':'torch_threads','cuda_module_loading':'cuda_module_loading_driver'}.items()}
manifest['environment'].update(allocator_conf=variants('PYTORCH_CUDA_ALLOC_CONF',True),tf32={'matmul':variants('tf32_matmul'),'cudnn':variants('tf32_cudnn')},workspace_environment={k:variants(k,True) for k in ['CUBLAS_WORKSPACE_CONFIG','CUBLASLT_WORKSPACE_SIZE','PYTORCH_NO_CUDA_MEMORY_CACHING','OMP_NUM_THREADS']},evidence=[p for p,e in envs],evidence_level=OBS)
host=subprocess.check_output(['hostname'],text=True).strip();gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip()
assert host=='gpuvm-1' and 'GPU-17eb7299-8b39-914d-7c5b-26908502904f' in gpu
manifest['vm']={'role':'INUGPU','hostname':host,'gpu':gpu,'evidence':['hostname','nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader'],'evidence_level':OBS,'observed_at':START}
# Definitions state exact scopes; count-valued memory_stats fields do not pretend to be bytes.
def metric(name,unit,start,end,reset,needle,cpu,level=CODE,**kw):manifest['metrics'].append(dict(name=name,unit=unit,window_start=start,window_end=end,reset=reset,source=loc(needle),maps_to_cpu=cpu,evidence_level=level,**kw))
metric('measures[].start.{allocated,reserved}','bytes','after previous output/input release and synchronize, after warmup for i=0','before peak reset and before GPU input copies','none','st = {','CPU live allocations and cached segments after load + warmup 3')
metric('measures[].peak.{allocated,reserved}','bytes','after start/begin boundary, immediately at reset_peak_memory_stats','after invoke() and synchronize; GPU inputs/output remain live','each measure, after start recorded','"peak": {','max xMem usage / max segment total in matching measure window')
metric('measures[].end_with_output.{allocated,reserved}','bytes','invoke complete and synchronized','before sampler stop, output CPU copy and del gi,o','none','"end_with_output": {','current usage/segment bytes while GPU inputs/output are referenced')
metric('measures[].released.{allocated,reserved}','bytes','after output copied to CPU and del gi,o','after synchronize, at measure_i_end boundary','none','row["released"]','current allocator state after input/output release; allocator cache retained')
metric('after_load','mixed: *_bytes and max_* bytes; allocation/segment/num_* counts','reset immediately before from_pretrained','after .to(cuda,float16).eval and synchronize','before loader, after model_load_begin boundary','R["after_load"]','load-only current/peak state, before any invocation')
statkeys=ast.literal_eval(next(n.value for n in ast.parse(wt).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='STAT_KEYS' for t in n.targets)))
metric('measures[].stats_end','mixed: byte keys bytes, other keys counts','peak stats since current reset; cumulative counters since process start','with output/input still live','peak counters reset per measurement; allocation/free counters not explicitly reset','"stats_end": stats()','compare matching counters/peaks; do not treat every key as bytes',keys=statkeys)
metric('measures[].nvml_process_peak_bytes','bytes','sampler.start after peak reset before invoke','sampler.stop after peak/end/stats_end reads; includes final synchronous sample','new sample list per measured invocation','class NvmlSampler','not directly allocator reserved/allocated; process memory includes allocator-external overhead',interval_s=0.01,note='10ms sampling is not an exact continuous maximum; only measure mode')
for name,definition in {'first_measure_peak_reserved':'measures[0].peak.reserved','first_measure_peak_allocated':'measures[0].peak.allocated','first_measure_start_reserved':'measures[0].start.reserved','first_measure_start_allocated':'measures[0].start.allocated','median_peak_reserved':'median of 5 reserved peaks','max_peak_reserved':'max of 5 reserved peaks','median_peak_allocated':'median of 5 allocated peaks','max_peak_allocated':'max of 5 allocated peaks','max_nvml_process_peak':'maximum non-null sampled process peaks','median_nvml_process_peak':'median non-null sampled process peaks'}.items():
 metric('summary.'+name,'bytes','measure_0 after warmup 3','measure_0 end' if name.startswith('first') else 'measure_4 completion','per-measure peak reset','R["summary"]','primary candidate: cached measure_0 reserved' if name=='first_measure_peak_reserved' else definition,definition=definition)
metric('outside (NVML minus reserved)','bytes',None,None,'none','class NvmlSampler','worker does not calculate an outside metric; NVML and reserved maxima are not necessarily simultaneous',exists=False)
metric('history.boundaries','mixed bytes/counters','history enabled before torch.cuda.init; model_load_begin before load reset','measure_0_end after GPU input/output deletion and synchronize','trace load reset and measure reset only','def boundary(name):','replay full event prefix load → warmup_0,1,2 → measure_0; compare at named boundaries',boundary_names=['model_load_begin','model_load_end']+[f'warmup_{i}_{b}' for i in range(3) for b in ['begin','end']]+['measure_0_begin','measure_0_end'])
# Direct numeric checks retained case by case.
checks={}
for label in ['rtx3090','rtx3090__attn-default']:
 rows=[]
 for x in manifest['cases']:
  if x['label']!=label or x['mode']!='measure' or not x.get('measures'):continue
  ms=x['measures'];rows.append({'condition_id':x['condition_id'],'first_start_reserved_equals_peak':ms[0]['start']['reserved']==ms[0]['peak']['reserved'],'all_five_peak_reserved_equal':len(ms)==5 and len({m['peak']['reserved'] for m in ms})==1,'all_five_peak_allocated_equal':len(ms)==5 and len({m['peak']['allocated'] for m in ms})==1,'generated_lengths_match':all(m['new_tokens']==x['max_new_tokens'] for m in ms),'evidence':x['evidence'],'evidence_level':OBS})
 checks[label]={'rows':rows,'n':len(rows),'first_start_equals_peak_count':sum(x['first_start_reserved_equals_peak'] for x in rows),'five_reserved_equal_count':sum(x['all_five_peak_reserved_equal'] for x in rows),'five_allocated_equal_count':sum(x['all_five_peak_allocated_equal'] for x in rows),'evidence_level':OBS}
replays=[]
for p in sorted((R/'results/xmem_replay_rtx3090').rglob('*.json')):
 a=read(p); comparisons=[]
 for w in a['windows']:
  for suffix in ['start_reserved','end_reserved','start_allocated','peak_reserved','peak_allocated']:
   g='gpu_'+suffix;x='xmem_'+suffix
   if g in w and x in w:comparisons.append({'window':w['window'],'metric':suffix,'gpu':w[g],'xmem':w[x],'equal':w[g]==w[x]})
 replays.append({'source':str(p),'condition_id':a['condition_id'],'comparisons':comparisons,'all_available_comparisons_equal':all(z['equal'] for z in comparisons),'missing_frees':a['missing_frees'],'boundaries':[a['boundaries_located'],a['boundaries_total']],'segment_count_equal':a['gpu_segment_alloc_events']==a['xmem_segment_alloc_count'],'evidence_level':OBS});inventory_file(p)
attention_path=R/'results/attention_kernels_rtx3090_default.json';attention=read(attention_path);inventory_file(attention_path)
attention_rows=[{k:v for k,v in row.items() if k in ['condition_id','model','attn_implementation','prefill_attention_kernels','decode_step_attention_kernels','error']} for row in attention['rows']]
# Input identity across both pilots.
left={x['condition_id']:x for x in manifest['cases'] if x['label']=='rtx3090' and x['mode']=='measure'}
input_comparison=[{'condition_id':x['condition_id'],'equal':x['inputs']==left[x['condition_id']]['inputs'],'source':[x['evidence'],left[x['condition_id']]['evidence']],'evidence_level':OBS} for x in manifest['cases'] if x['label']=='rtx3090__attn-default' and x['mode']=='measure']
xrepo=Path('/home/inugpu/xmem-work/xMem'); xstatus=subprocess.check_output(['git','-C',str(xrepo),'status','--short'],text=True).splitlines(); xcommit=subprocess.check_output(['git','-C',str(xrepo),'rev-parse','HEAD'],text=True).strip()
for p in [R/'tools/xmem_replay.py',Path('/home/inugpu/xmem-work/cgf-fp16-replay-20261111/xmem_replay.py'),xrepo/'perf_estimator/allocator/allocator.py',bundle/'MANIFEST_SHA256SUMS']:inventory_file(p)
manifest['run']={'experiment_root':str(R),'version_control':'none (sha256)','version_control_evidence':ev({'root_dot_git_exists':(R/'.git').exists()},R,CODE),'image':{'metadata':read(R/'aws/IMAGE_INFO.json'),'evidence':str(R/'aws/IMAGE_INFO.json'),'evidence_level':CODE,'observed_image_ids':sorted({e['image_id'] for p,e in envs})},'labels':labels,'snapshot':{'started_kst':START,'completed_kst':datetime.datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),'note':'Frozen membership of finalized result JSON files at audit start; per-file SHA256 retained. Live run continues; this is not a run-completion assertion.','evidence_level':OBS},'evidence_convention':{'observed_in_results':'Result JSON/log or directly observed read-only system output','code_only':'Current code/config/artifact only; not proof of past execution','unverified':'No evidence or absent value. field_evidence applies individually; parent evidence_level supplies default for nested fields unless overridden.'},'container_mapping':[{'host':str(R/'src'),'container':'/src','access':'ro'},{'host':str(R/'config'),'container':'/config','access':'ro'},{'host':'/home/inugpu/cloudgpufit-bench/hf-cache','container':'/hf','access':'ro'},{'host':str(R/'runs/<label>/<condition_id>'),'container':'/out','access':'rw'}], 'container_options':ev(['--rm','--pull=never','--network none','--pid=host','--gpus device=<verified GPU UUID>','--user uid:gid','--entrypoint python','-I /src/vram_worker.py'],loc('cmd = ["docker"',R/'src/run_conditions.py'),CODE),'container_mapping_evidence':ev('Exact executed command is retained per case in launch.cmd',loc('cmd = ["docker"',R/'src/run_conditions.py'),CODE),'container_environment':ev(['HOME=/tmp','HF_HOME=/hf','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','CUDA_MODULE_LOADING=LAZY','CGF_IMAGE_ID','CGF_GPU_LABEL','CGF_INSTANCE_TYPE'],loc('"-e", "HOME=/tmp"',R/'src/run_conditions.py'),CODE),'workload_protocol':{'source':str(worker),'evidence_level':CODE,'seed':models['common']['seed'],'input_text':models['common']['text_input_rule'],'input_image':models['common']['image_input_rule'],'input_hash':'sha_tensor: contiguous CPU tensor, view(uint8), numpy().tobytes(), SHA256; per-case hashes retained','generation':'GenerationConfig.from_model_config; full actual dict in cases; greedy num_beams=1, do_sample=false, use_cache=true, min=max=N; successful cases have exact requested new tokens','new_token_count':'decoder output.shape[1]-input_len; T5 output.shape[1]-1','mode_iterations':{'measure':[3,5],'validate':[0,1],'trace':[3,1]},'invocation':'CPU source inputs retained, .to(cuda) each invoke inside measured interval; inference_mode; vision model forward once, text generate','release':'warmup/measure GPU inputs and output deleted after end_with_output; synchronize; no explicit empty_cache call in current worker','peak_reset':'before from_pretrained; before each measure after start and trace begin boundary','finite_check':'validate only, hook raw logits for each forward; stored GPU boolean tensors remain referenced through validation measurement','attn_config_warning':'models.json common.attn_implementation=eager is descriptive; worker reads only common.seed and actual --attn; default omits loader attention kwarg','historical_code_limit':'v1 worker source is absent here. v1 result.protocol confirms iterations/cache/reset; current source is not proof of every detail of old worker.'},'pilot_comparison_checks':checks,'pilot_inputs_match':input_comparison,'attention_diagnostics':{'source':str(attention_path),'rows':attention_rows,'note':attention['note'],'scope':'Separate profiler diagnostic for listed pilot shapes only, not all main395 conditions. Enabled SDPA flags alone do not prove selected kernel. CNN has no attention operation; eager is config value.','evidence_level':OBS},'xmem':{'repository':str(xrepo),'commit':xcommit,'git_status_short':xstatus,'replay_checks':replays,'evidence_level':OBS,'limitation':'GPU allocation-event replay validates allocator behavior for these traces; it does not validate CPU-generated operation/request trace equivalence.'},'transfer_paths':[{'path':'/home/inugpu/cgf-fp16-aws-bundle-20261111/cgf-fp16-aws-full-20261112.tar','exists':True,'state':'prepared','evidence_level':CODE},{'path':str(bundle),'state':'prepared','evidence_level':CODE},{'path':'/home/inugpu/cgf-fp16-inuos-handoff-20261112','state':'local_handoff_present','evidence_level':CODE},{'path':'/home/inuos/cgf-fp16-inuos-handoff-20261112','state':'remote_copy_not_independently_verified','evidence_level':UN}],'source_inventory':inventory,'no_new_experiments':True,'no_file_transfer_performed':True}
question('CPU 예측이 load → warmup 3 → measure_0 전체 prefix를 캐시 유지하며 재생합니까?','파일럿의 first start reserved=peak 여부를 run.pilot_comparison_checks에 직접 검산했습니다. cold 측정과 warmed measure_0는 다른 구간입니다.')
question('캐시 유지, 입력 GPU 복사와 출력 해제 수명, start 기록 후 peak 리셋 시점이 GPU와 같습니까?','입력 복사도 invoke 안에 포함되고 end_with_output에는 GPU 입력과 출력이 살아 있습니다.')
question('CPU에서 실제 attention 구현과 선택된 aten 연산은 무엇입니까?','GPU 진단의 sdpa 모델은 FlashAttention 커널이 관측됩니다. T5는 eager, CNN은 attention 없는 모델로 config eager입니다. CPU math SDPA/eager의 중간 텐서는 다를 수 있습니다.',evidence=[str(attention_path)])
question('CPU fp16 미지원 연산은 dtype 승격, 대체 구현, 실패 중 어떻게 처리합니까?','GPU 실제 param_dtypes는 케이스에 보존했습니다. param dtype만으로 모든 중간 연산 정밀도를 보장하지 않습니다.')
question('조건별 input tensor shape/dtype/sha256이 gpu_manifest cases.inputs와 일치합니까?','seed와 특수 토큰 치환, 이미지 float32 생성 후 fp16 변환까지 일치해야 합니다.')
question('CPU 예측값은 GPU reserved와 allocated 중 무엇이며 어떤 측정 구간의 최대입니까?','primary candidate는 summary.first_measure_peak_reserved입니다. NVML은 별도 프로세스 메모리 지표입니다.')
question('cuBLAS/cuDNN workspace와 CUDA 전용 요청을 모델링합니까?','GPU trace 재생 일치만으로 CPU 요청 목록의 일치를 입증하지 못합니다.')
question('동일 conditions_main395.json SHA256을 사용합니까?','기대 SHA256='+sha(R/'config/conditions_main395.json'))
question('GenerationConfig 전체, 모델 revision/loader/use_safetensors 및 생성 길이 검사가 동일합니까?','요청한 토큰 길이와 실제 생성 길이를 케이스별로 보존했습니다.')
question('이전 v1 worker 원본을 보관하고 있습니까?','v1 결과 hash와 현재 worker hash가 다릅니다. 현재 코드로 과거의 모든 세부 동작을 확정할 수 없습니다.',to='INUGPU',evidence=[str(R/'logs/code_sha256_pilot.txt')])
question('models.json의 common eager 설명 대신 CLI와 결과 attn_requested를 비교 기준으로 사용합니까?','현재 runner 기본 default와 common eager 문구가 다르며 worker는 common attention 값을 읽지 않습니다.',evidence=[str(R/'config/models.json'),loc('attn_kw =')],level=CODE)
question('conditions_main395.json의 rule 문구에 남은 generate_main355.py 이름을 문서상 오기로 처리해도 됩니까?','실제 생성 스크립트는 generate_main.py이며 실제 조건은 395개입니다. 원본은 수정하지 않았습니다.',to='INUGPU',evidence=[str(R/'config/conditions_main395.json'),str(R/'config/generate_main.py')],level=CODE)
question('별도 reserved 진단 실험은 이번 비교에서 제외하는 것이 맞습니까?','/home/inugpu/cgf-reserved-diag-20261110/는 별도 실험으로 manifest 케이스에 포함하지 않았습니다.',to='INUGPU')
# Archive record checksum lists are history, not automatic assertions about current files.
hash_audits=[]
for lp in [R/'logs/code_sha256_pilot.txt',R/'logs/code_sha256_pilot_v2.txt',R/'logs/code_sha256_main395_bundle.txt',R/'logs/SHA256SUMS_code_config_results.txt',bundle/'MANIFEST_SHA256SUMS']:
 rows=[]
 for line in lp.read_text().splitlines():
  match=re.match(r'^([a-f0-9]{64})\s+\*?(.+)$',line)
  if not match:continue
  expected,rel=match.groups();p=(bundle if lp.parent==bundle else R)/rel
  # avoid hashing multi-GB archives or cache weights while experiments run
  current=sha(p) if p.is_file() and p.stat().st_size<20_000_000 else None
  rows.append({'path':str(p),'recorded_sha256':expected,'current_sha256':current,'matches':current==expected if current else None,'evidence_level':OBS})
 hash_audits.append({'path':str(lp),'rows':rows,'evidence_level':OBS})
manifest['run']['historical_hash_audits']=hash_audits
mismatches=[z for a in hash_audits for z in a['rows'] if z['matches'] is False]
if mismatches:question('과거 체크섬과 현재 파일의 차이를 버전 차이로 유지해도 됩니까?','과거 파일럿 이후 수정된 파일이 있습니다. run.historical_hash_audits에 기록값과 현재 해시를 병기했습니다.',to='INUGPU',evidence=[z['path'] for z in mismatches],level=OBS)
# Additional provenance and audit limits.
manifest['run']['xmem']['container_mapping']=ev({'container':'xmem-gpu','host':'/home/inugpu/xmem-work','destination':'/work','rw':True},'docker inspect xmem-gpu --format {{json .Mounts}}',OBS)
manifest['run']['xmem']['container_mapping']['checked_against_live_inspect']=True
manifest['run']['snapshot']['planned_not_yet_run_definition']='No finalized result in this snapshot; may include an invocation currently in flight. It does not mean the runner never started it.'
manifest['run']['version_control_evidence']['note']='Experiment root has no .git; source identity is per-file SHA256, not any enclosing home-directory repository.'
for label in labels:
 if label['kind']=='executed':
  if not label.get('log'):
   label['counts']['skipped_resume_log_events']=None
   label['counts']['skipped_resume_evidence_level']=UN
  else:
   label['counts']['skipped_resume_evidence_level']=OBS
  label['attn_actual_values']=sorted({x['attn_actual'] for x in manifest['cases'] if x['label']==label['label'] and x.get('attn_actual')})
  label['attn_missing_requested_count']=sum(x['state']=='executed' and x.get('attn_requested') is None for x in manifest['cases'] if x['label']==label['label'])
  label['code_hash_records']=[str(z['path']) for z in hash_audits if label['label'] in ['rtx3090','rtx3090__attn-default']]
for x in manifest['cases']:
 if x.get('validation') and x['result_status']=='nonfinite_output':
  x['failure_detail']=ev('Result reports nonfinite logits, first_nonfinite_step='+str(x['validation']['first_nonfinite_step'])+'; failed_stage and error fields are absent in original JSON, retained as null without inventing an exception.',x['evidence'])
 if x['state']=='prepared':
  for k in ['revision','loader','use_safetensors','precision_requested']:
   x['field_evidence'][k]={'source':str(bundle/'config/models.json'),'evidence_level':CODE}
  for k in ['attn_requested','call']:
   x['field_evidence'][k]={'source':str(bundle/('aws/aws_04_main.sh' if k=='attn_requested' else 'src/vram_worker.py')),'evidence_level':CODE}
 else:
  x['field_evidence']['precision_requested']={'source':loc('model = cls.from_pretrained'),'evidence_level':CODE}
  if 'launch' in x:x['field_evidence']['launch']={'source':x['evidence'].replace('.json','.launch.json'),'evidence_level':OBS}
  x['field_evidence']['attempts']={'source':str(Path(x['evidence']).parent/'attempts'),'evidence_level':OBS}
for f in ['models.json','conditions_main395.json','pilot.json']:
 assert sha(bundle/'config'/f)==sha(R/'config'/f)
 manifest['run'].setdefault('prepared_bundle_config_checks',[]).append(ev({'name':f,'local_sha256':sha(R/'config'/f),'bundle_sha256':sha(bundle/'config'/f),'equal':True},str(bundle/'config'/f),CODE))
# Verify local handoff checksums against experiment; remote contents remain unverified.
handoff=Path('/home/inugpu/cgf-fp16-inuos-handoff-20261112')
for f in ['models.json','conditions_main395.json']:
 manifest['run'].setdefault('local_inuos_handoff_config_checks',[]).append(ev({'name':f,'local_sha256':sha(R/'config'/f),'handoff_sha256':sha(handoff/'config'/f),'equal':sha(R/'config'/f)==sha(handoff/'config'/f)},str(handoff/'config'/f),CODE))
# Historical report claims broader boundary agreement than replay JSON explicitly exposes.
question('xMem 검증의 범위를 기록된 비교 필드로 한정해 해석합니까?','재생 JSON은 모든 warmup 구간의 GPU peak allocated나 end allocated를 담지 않습니다. 기록된 start/end reserved, start allocated, measure_0 peak만 직접 검산했습니다. 보고서의 모든 경계 시작·피크·끝 전체 일치 문장은 더 넓은 표현입니다.',evidence=[str(R/'results/REPORT_rtx3090_pilot_eager_vs_default_KO.md'),str(R/'tools/xmem_replay.py')],level=OBS)

# Verify each case is unique and planned cardinality is explicit.
assert len({(x['label'],x['condition_id'],x['mode']) for x in manifest['cases']})==len(manifest['cases'])
assert len([x for x in manifest['cases'] if x['label']=='rtx3090__main395'])==790
assert set(manifest)=={'vm','run','environment','cases','metrics','open_questions'}
for x in manifest['cases']:
 if x['state']=='executed' and x.get('model') and x.get('revision'):
  assert x['revision']==models['models'][x['model']]['revision']
# Preserve UTC/KST client date; never infer completion from elapsed wall time.
manifest['run']['snapshot']['completed_kst']=datetime.datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
out=OUT/'gpu_manifest.json';out.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('PATH',out);print('SHA256',sha(out));print('CASES',len(manifest['cases']))
for label in labels:print(label['label'],json.dumps(label['counts'],ensure_ascii=False))
print('PILOT_CHECKS',json.dumps({k:{a:b for a,b in v.items() if a!='rows'} for k,v in checks.items()}))
print('REPLAY',len(replays),'all_equal',all(x['all_available_comparisons_equal'] for x in replays))

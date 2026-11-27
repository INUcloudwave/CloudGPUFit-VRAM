#!/usr/bin/env python3
"""CPU-only final local audit: all model raw plus today's five sentinels."""
import collections,csv,hashlib,json
from pathlib import Path
from statistics import median
from datetime import datetime,timezone,timedelta
import audit_qwen_bf16 as raw
ROOT=Path(__file__).resolve().parent.parent
NEW='local-closeout-20261110'
OUT=ROOT/'results/local-closeout-20261110-analysis'
GPU='GPU-17eb7299-8b39-914d-7c5b-26908502904f'
PRIMARY=['after_first_forward','after_first_inference','after_warmup3','after_performance5','after_repeat_block0','after_repeat_block1','after_repeat_block2','final_idle']
CONSTANTS={'LAZY':312.0,'EAGER':1040.0}
M=2**20

def write(name,rows):
 if not rows:return
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def quantile(v,p):
 v=sorted(v);pos=(len(v)-1)*p;i=int(pos);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(pos-i)
def main():
 orch=json.loads((ROOT/'results'/NEW/'orchestration.json').read_text());assert orch['status']=='complete' and len(orch['results'])==5 and all(x['exit_code']==0 for x in orch['results'])
 assert not OUT.exists(),'Preserve existing analysis'
 for n,h in {**orch['source_sha256'],**orch['input_sha256']}.items():assert raw.sha(ROOT/'outside-lab'/n)==h
 plan={c['case_id']:c for c in json.loads((ROOT/'outside-lab/local-closeout-plan.json').read_text())['cases']}
 provenance=[];stages=[];quality=[];skipped=[];hashes=[];new_counts=collections.Counter();processes=[]
 for mp in sorted((ROOT/'results').rglob('manifest.json')):
  m=json.loads(mp.read_text());args=m.get('args',{});path=str(mp.relative_to(ROOT));suite=mp.relative_to(ROOT/'results').parts[0]
  reason=None
  if args.get('scenario')!='model':reason='non-model scenario'
  elif mp.parent.name=='smoke' or 'smoke' in suite:reason='smoke'
  elif m.get('status')!='ok':reason='unsuccessful worker'
  elif m.get('gpu',{}).get('uuid')!=GPU:reason='different GPU environment'
  if reason:skipped.append(dict(path=path,reason=reason,status=m.get('status')));continue
  mode=m.get('late_cuda_diagnostics',{}).get('module_loading',{}).get('mode')
  assert mode in CONSTANTS,(path,mode)
  assert m['source_sha256']=='7df9a02ce9ca938b9702ee2c9fd7d0414d0a4cfc0cfe5ab87c97d7a069243d88'
  identity=path;groups=collections.defaultdict(list);maxres=[];maxnv=[]
  base=dict(identity=identity,suite=suite,run_id=m['run_id'],pid=m['pid'],model=m['case']['model'],dtype=m['case']['dtype'],params=json.dumps(m['case']['params'],sort_keys=True),attention=args['attention'],history=args['history'],module_mode=mode,driver=m['gpu']['driver'],image_id=m['image_id'],torch=m['torch'],cuda_build=m['cuda_build'],constant_mib=CONSTANTS[mode])
  if suite==NEW:
   job=next(j for j in orch['jobs'] if j['run_id']==m['run_id']);assert m['case']==plan[job['case_id']] and args['attention']==job['attention'] and mode=='LAZY'
   assert m['uuid_check']['match'] and m['case']['dtype']=='bf16' and m['revision']['revision']=='272aa6d30271028cdf113345b1e82a660437edaa'
   sm=json.loads(mp.parent.parent.joinpath('suite-manifest.json').read_text());assert sm['status']=='complete' and sm['results'][0]['exit_code']==0
   base.update(cond=job['cond'],block=job['block'])
  sample=mp.with_name('samples.jsonl')
  for line in sample.read_text().splitlines():
   r=json.loads(line);n=r.get('nvml_process_bytes');v=r.get('reserved_bytes');o=None if n is None or v is None else n-v
   assert o==r.get('outside_bytes') and r['pid']==m['pid']
   flags=raw.flags(r);groups[r['stage'],r['kind']].append((r,flags))
   if r.get('reserved_bytes') is not None:maxres.append(r['reserved_bytes']/M)
   if r.get('nvml_process_bytes') is not None:maxnv.append(r['nvml_process_bytes']/M)
   if suite==NEW:
    new_counts[r['kind']]+=1
    for flag in flags:new_counts[flag]+=1
    expected=o is None and r['stage'] in ['before_torch_import','after_torch_import'] and not r['torch']['initialized'] and r['nvml']['own_match_status']=='absent'
    new_counts['expected_preinit_missing']+=expected
    if flags:quality.append(dict(identity=identity,stage=r['stage'],kind=r['kind'],flags=';'.join(flags),expected_preinit=expected,query_ms=(r['query_end_ns']-r['query_begin_ns'])/1e6,outside_bytes=o))
  worker_stages=[]
  for (stage,kind),pairs in groups.items():
   good=[r for r,f in pairs if not f];vals=[r['outside_bytes']/M for r in good]
   if kind!='stable':continue
   row=dict(**base,stage=stage,n_raw=len(pairs),n_clean=len(good),usable=len(good)>=10,median_outside_mib=median(vals) if vals else None,last10_span_mib=max(vals[-10:])-min(vals[-10:]) if len(vals)>=10 else None)
   if row['usable']:
    row.update(error_mib=row['median_outside_mib']-CONSTANTS[mode],abs_error_mib=abs(row['median_outside_mib']-CONSTANTS[mode]))
    row['underestimate_mib']=max(row['error_mib'],0)
    for k in ['nvml_process_bytes','allocated_bytes','reserved_bytes']:row[k+'_paired_median_mib']=median(r[k]/M for r in good)
    if stage in PRIMARY:worker_stages.append(row)
   stages.append(row)
  assert worker_stages and all(r['last10_span_mib']<=8 for r in worker_stages),path
  # Report process-level medians separately from stage rows, without treating time samples as independent repeats.
  observed=median(r['median_outside_mib'] for r in worker_stages);error=observed-CONSTANTS[mode]
  processes.append(dict(**base,n_primary_stages=len(worker_stages),observed_process_median_mib=observed,error_mib=error,abs_error_mib=abs(error),max_stage_abs_error_mib=max(r['abs_error_mib'] for r in worker_stages),max_stage_underestimate_mib=max(r['underestimate_mib'] for r in worker_stages),outside_stage_range_mib=max(r['median_outside_mib'] for r in worker_stages)-min(r['median_outside_mib'] for r in worker_stages),sampled_reserved_max_mib=max(maxres),sampled_nvml_max_mib=max(maxnv)))
  provenance.append(dict(identity=identity,probe_sha256=m['source_sha256'],criteria_sha256=m['criteria_sha256'],input_sha256=json.dumps(m['input_sha256'],sort_keys=True),revision=m['revision']['revision'],gpu_uuid=m['gpu']['uuid'],started_kst=datetime.fromtimestamp(m['started_wall_ns']/1e9,timezone(timedelta(hours=9))).isoformat()))
  hashes += [dict(path=str(p.relative_to(ROOT)),sha256=raw.sha(p)) for p in [mp,sample]]
 today=[r for r in stages if r['suite']==NEW and r['usable']];assert len({r['identity'] for r in today})==5
 sdpa_base={r['stage']:r for r in today if r['attention']=='sdpa' and r['cond']=='base'};comparisons=[]
 for r in today:
  b=sdpa_base[r['stage']];comparisons.append(dict(attention=r['attention'],cond=r['cond'],stage=r['stage'],today_mib=r['median_outside_mib'],sdpa_base_mib=b['median_outside_mib'],difference_mib=r['median_outside_mib']-b['median_outside_mib'],within_100_mib=abs(r['median_outside_mib']-b['median_outside_mib'])<=100))
 dates=[]
 for r in today:
  if r['cond']!='base':continue
  prev=[p for p in stages if p['suite']=='qwen-attention-01' and p['attention']==r['attention'] and p['stage']==r['stage'] and p['usable']]
  assert len(prev)==3
  vs=[p['median_outside_mib'] for p in prev];dates.append(dict(attention=r['attention'],stage=r['stage'],today_mib=r['median_outside_mib'],previous_median_mib=median(vs),previous_range_mib=max(vs)-min(vs),difference_mib=r['median_outside_mib']-median(vs),previous_suite='qwen-attention-01',previous_n_processes=3,today_n_processes=1))
 aggregates=[]
 for mode in CONSTANTS:
  pr=[r for r in processes if r['module_mode']==mode];v=[r['abs_error_mib'] for r in pr]
  aggregates.append(dict(module_mode=mode,constant_mib=CONSTANTS[mode],n_processes=len(pr),n_gpu_environments=1,process_median_mae_mib=sum(v)/len(v),process_median_abs_p95_mib=quantile(v,.95),max_process_stage_abs_error_mib=max(r['max_stage_abs_error_mib'] for r in pr),max_underestimate_mib=max(r['max_stage_underestimate_mib'] for r in pr),n_processes_exceeding_100=sum(r['max_stage_abs_error_mib']>100 for r in pr)))
 env=collections.defaultdict(list)
 for p in processes:env[p['driver'],p['image_id'],p['torch'],p['cuda_build'],p['module_mode']].append(p)
 envrows=[]
 for (driver,img,torch,cuda,mode),pr in env.items():envrows.append(dict(driver=driver,image_id=img,torch=torch,cuda_build=cuda,module_mode=mode,n_processes=len(pr),observed_min_mib=min(p['observed_process_median_mib'] for p in pr),observed_max_mib=max(p['observed_process_median_mib'] for p in pr),max_abs_error_mib=max(p['max_stage_abs_error_mib'] for p in pr)))
 OUT.mkdir()
 write('all_local_process.csv',processes);write('all_local_stable_stage.csv',stages);write('constant_error_summary.csv',aggregates);write('environment_summary.csv',envrows);write('new_quality_audit.csv',quality);write('today_vs_sdpa_base.csv',comparisons);write('cross_day_comparison.csv',dates);write('run_provenance.csv',provenance);write('excluded_manifests.csv',skipped);write('evidence_sha256.csv',hashes)
 summary=dict(n_successful_local_model_workers=len(processes),n_primary_process_stage=sum(p['n_primary_stages'] for p in processes),n_gpu_environments=1,new_workers=5,new_counts=dict(new_counts),new_primary_max_abs_error_mib=max(r['abs_error_mib'] for r in today if r['stage'] in PRIMARY),new_primary_range_mib=max(r['median_outside_mib'] for r in today if r['stage'] in PRIMARY)-min(r['median_outside_mib'] for r in today if r['stage'] in PRIMARY),cross_day_max_abs_diff_mib=max(abs(r['difference_mib']) for r in dates),constant_error_summary=aggregates,retrospective_descriptive_not_heldout_prediction=True)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

#!/usr/bin/env python3
"""CPU-only paired-block LAZY/EAGER analysis, preserving raw suite identity."""
import collections,csv,hashlib,json
from pathlib import Path
from statistics import median
import audit_qwen_bf16 as raw
ROOT=raw.ROOT
OUT=ROOT/'results/loading-confirmation-01-analysis'
def write(name,rows):
 if not rows:return
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def main():
 orch_path=ROOT/'results/loading-confirmation-01/orchestration.json';s=json.loads(orch_path.read_text())
 assert s['status']=='complete' and len(s['results'])==20 and all(r['exit_code']==0 for r in s['results'])
 assert not OUT.exists(),'Do not overwrite analysis'
 stages=[];runs=[];quality=[];hashes=[];counts=collections.Counter();lat=[];manifests=[];checks=[]
 for name,h in s['source_sha256'].items():
  p=ROOT/'outside-lab'/name;checks.append(dict(path=str(p.relative_to(ROOT)),sha256=h,current=raw.sha(p),matches=h==raw.sha(p)))
 assert all(r['matches'] for r in checks)
 for name,h in s['input_sha256'].items():
  assert raw.sha(ROOT/'outside-lab'/name)==h
 for job in s['jobs']:
  d=ROOT/'results/loading-confirmation-01'/job['child'];suite=json.loads((d/'session/suite-manifest.json').read_text())
  assert suite['status']=='complete' and len(suite['results'])==len(suite['jobs'])==1 and all(r['exit_code']==0 for r in suite['results'])
  st,ru,qu,co,la,ha=raw.audit(str(d.relative_to(ROOT/'results')))
  assert len(ru)==1 and {r['history'] for r in ru}=={'direct'}
  for rows in [st,ru,qu]:
   for r in rows:r.update(block=job['block'],mode=job['mode'])
  stages+=st;runs+=ru;quality+=qu;counts.update(co);lat+=la;hashes+=ha
  hashes += [dict(path=str(p.relative_to(ROOT)),sha256=raw.sha(p)) for p in [d/'session/suite-manifest.json',d/'container-inspect.json']]
  for mp in (d/'session').glob('*/manifest.json'):
   m=json.loads(mp.read_text());manifests.append(m)
   assert m['status']=='ok' and m['uuid_check']['match'] and m['args']['threads']==4
   assert m['case']['dtype']=='bf16' and m['case']['params']=={'batch_size':1,'seq_len':256,'max_new_tokens':64}
   assert m['environment']['CUDA_MODULE_LOADING']==job['mode']==m['late_cuda_diagnostics']['module_loading']['mode']
   assert m['environment']['CUDA_MODULE_DATA_LOADING'] is None
   assert not m['initial_nvml']['errors'] and not m['initial_nvml']['other_gpu_pids']
   for name,h in suite['source_sha256'].items():assert raw.sha(ROOT/'outside-lab'/name)==h
   for name,h in suite['input_sha256'].items():assert raw.sha(ROOT/'outside-lab'/Path(name).name)==h
 for key in ['gpu_uuid','driver','image_id','probe_sha256','criteria_sha256','revision','input_sha256','torch','cuda_build','transformers','python']:
  assert len({r[key] for r in runs})==1,key
 assert len({r['pid'] for r in runs})==20
 # Compare full recorded environments and args, allowing only intended mode and output/run identity differences.
 for field in ['environment','environment_after_torch_import','environment_after_cuda_init','environment_at_end']:
  norm=[{k:v for k,v in m[field].items() if k!='CUDA_MODULE_LOADING'} for m in manifests]
  assert all(x==norm[0] for x in norm),field
 for k in ['attention','sampling','query_order','stable_samples','interval_ms','threads','dtype','tf32']:
  assert len({str(m['args'].get(k)) for m in manifests})==1,k
 stable=[r for r in stages if r['kind']=='stable' and r['n_clean']>=10]
 assert all(r['last10_span_mib']<=8 for r in stable)
 groups=collections.defaultdict(list)
 for r in stable:groups[r['mode'],r['history'],r['stage']].append(r)
 reps=[]
 for (mode,history,stage),rs in groups.items():
  vs=[r['outside_median_mib'] for r in rs];assert len(rs)==10
  reps.append(dict(mode=mode,history=history,stage=stage,n_processes=len(rs),outside_median_mib=median(vs),min_mib=min(vs),max_mib=max(vs),range_mib=max(vs)-min(vs)))
 index={(r['block'],r['mode'],r['history'],r['stage']):r for r in stable};diff=[]
 for r in stable:
  if r['mode']!='LAZY':continue
  q=index[r['block'],'EAGER',r['history'],r['stage']]
  diff.append(dict(block=r['block'],history=r['history'],stage=r['stage'],lazy_mib=r['outside_median_mib'],eager_mib=q['outside_median_mib'],eager_minus_lazy_mib=q['outside_median_mib']-r['outside_median_mib']))
 dg=collections.defaultdict(list)
 for r in diff:dg[r['history'],r['stage']].append(r)
 effects=[]
 for (history,stage),rs in dg.items():
  vs=[r['eager_minus_lazy_mib'] for r in rs]
  effects.append(dict(history=history,stage=stage,n_blocks=len(rs),median_difference_mib=median(vs),min_difference_mib=min(vs),max_difference_mib=max(vs),same_direction_ge16=all(v>=16 for v in vs) or all(v<=-16 for v in vs)))
 OUT.mkdir()
 write('paired_process_stage.csv',stages);write('run_provenance.csv',runs);write('quality_audit.csv',quality);write('repeat_summary.csv',reps);write('block_mode_comparison.csv',diff);write('mode_effect_summary.csv',effects);write('source_verification.csv',checks)
 hashes.append(dict(path=str(orch_path.relative_to(ROOT)),sha256=raw.sha(orch_path)));write('evidence_sha256.csv',hashes)
 summary=dict(n_runs=len(runs),n_unique_pids=len({r['pid'] for r in runs}),n_gpu_environments=1,counts=dict(counts),stable_process_stage=len(stable),repeat_groups=len(reps),max_repeat_range_mib=max(r['range_mib'] for r in reps),min_stable_clean=min(r['n_clean'] for r in stable),max_plateau_span_mib=max(r['last10_span_mib'] for r in stable),block_mode_pairs=len(diff),query_ms_median=median(lat),query_ms_p95=sorted(lat)[int(.95*(len(lat)-1))],query_ms_max=max(lat))
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
 print('PRIMARY EFFECTS')
 for r in effects:
  if r['stage'] in ['after_cuda_init','after_model_load','after_first_forward','final_idle']:print(r)
if __name__=='__main__':main()

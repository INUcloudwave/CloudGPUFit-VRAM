#!/usr/bin/env python3
"""CPU-only analysis of qwen-shape-01: each case vs base within block; 100 MiB practical criterion."""
import collections,csv,json
from pathlib import Path
from statistics import median
import audit_qwen_bf16 as raw
ROOT=raw.ROOT
SUITE='qwen-shape-01'
OUT=ROOT/'results/qwen-shape-01-analysis'
M=2**20
PRIMARY=['after_first_forward','after_first_inference','after_warmup3','after_repeat_block0','after_repeat_block1','after_repeat_block2','final_idle']
def write(name,rows):
 if not rows:return
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def device_gap(samples):
 g=[]
 for l in samples.read_text().splitlines():
  r=json.loads(l)
  try:d=r['device_memory_info']['value']['used_bytes'];p=r['nvml_process_bytes']
  except Exception:continue
  if r.get('kind')=='stable' and p:g.append((d-p)/M)
 return g
def main():
 orch_path=ROOT/'results'/SUITE/'orchestration.json';s=json.loads(orch_path.read_text())
 assert s['status']=='complete' and len(s['results'])==15 and all(r['exit_code']==0 for r in s['results'])
 assert not OUT.exists(),'Do not overwrite analysis'
 checks=[]
 for name,h in s['source_sha256'].items():
  p=ROOT/'outside-lab'/name;checks.append(dict(path=str(p.relative_to(ROOT)),sha256=h,current=raw.sha(p),matches=h==raw.sha(p)))
 assert all(r['matches'] for r in checks)
 for name,h in s['input_sha256'].items():assert raw.sha(ROOT/'outside-lab'/name)==h
 plan={c['case_id']:c for c in json.loads((ROOT/'outside-lab/qwen-shape-plan.json').read_text())['cases']}
 stages=[];runs=[];quality=[];hashes=[];counts=collections.Counter();lat=[];manifests=[];gaps=[]
 for job in s['jobs']:
  d=ROOT/'results'/SUITE/job['child'];suite=json.loads((d/'session/suite-manifest.json').read_text())
  assert suite['status']=='complete' and len(suite['results'])==1 and suite['results'][0]['exit_code']==0
  st,ru,qu,co,la,ha=raw.audit(str(d.relative_to(ROOT/'results')))
  assert len(ru)==1 and ru[0]['history']=='direct'
  for rows in [st,ru,qu]:
   for r in rows:r.update(block=job['block'],cond=job['cond'])
  stages+=st;runs+=ru;quality+=qu;counts.update(co);lat+=la;hashes+=ha
  for mp in (d/'session').glob('*/manifest.json'):
   m=json.loads(mp.read_text());manifests.append(m)
   assert m['status']=='ok' and m['uuid_check']['match'] and m['args']['threads']==4 and m['args']['attention']=='eager'
   assert m['case']==plan[job['case_id']],'case mismatch'
   assert m['revision']['revision']=='272aa6d30271028cdf113345b1e82a660437edaa'
   assert m['environment']['CUDA_MODULE_LOADING']==m['late_cuda_diagnostics']['module_loading']['mode']=='LAZY'
   assert m['environment']['CUDA_MODULE_DATA_LOADING'] is None
   assert not m['initial_nvml']['errors'] and not m['initial_nvml']['other_gpu_pids']
   g=device_gap(mp.with_name('samples.jsonl'))
   gaps.append(dict(block=job['block'],cond=job['cond'],n=len(g),gap_min_mib=min(g),gap_median_mib=median(g),gap_max_mib=max(g)))
 for key in ['gpu_uuid','driver','image_id','probe_sha256','criteria_sha256','revision','torch','cuda_build','transformers','python']:
  assert len({r[key] for r in runs})==1,key
 assert len({r['pid'] for r in runs})==15
 for field in ['environment','environment_after_torch_import','environment_after_cuda_init','environment_at_end']:
  assert all(m[field]==manifests[0][field] for m in manifests),field
 for k in ['sampling','query_order','stable_samples','interval_ms','threads','dtype','tf32','attention','history']:
  assert len({str(m['args'].get(k)) for m in manifests})==1,k
 stable=[r for r in stages if r['kind']=='stable' and r['n_clean']>=10]
 unstable=[r for r in stable if r['last10_span_mib']>8]
 groups=collections.defaultdict(list)
 for r in stable:groups[r['cond'],r['stage']].append(r)
 reps=[]
 for (cond,stage),rs in sorted(groups.items()):
  vs=[r['outside_median_mib'] for r in rs]
  reps.append(dict(cond=cond,stage=stage,n_processes=len(rs),outside_median_mib=median(vs),min_mib=min(vs),max_mib=max(vs),range_mib=max(vs)-min(vs)))
 idx={(r['block'],r['cond'],r['stage']):r for r in stable};diff=[]
 for r in stable:
  if r['cond']=='base':continue
  b=idx.get((r['block'],'base',r['stage']))
  if b:diff.append(dict(block=r['block'],cond=r['cond'],stage=r['stage'],base_mib=b['outside_median_mib'],cond_mib=r['outside_median_mib'],cond_minus_base_mib=r['outside_median_mib']-b['outside_median_mib']))
 dg=collections.defaultdict(list)
 for r in diff:dg[r['cond'],r['stage']].append(r['cond_minus_base_mib'])
 effects=[dict(cond=c,stage=st,n_blocks=len(vs),median_diff_mib=median(vs),min_diff_mib=min(vs),max_diff_mib=max(vs),all_within_16=all(abs(v)<16 for v in vs),all_within_100=all(abs(v)<=100 for v in vs)) for (c,st),vs in sorted(dg.items())]
 prim=[r for r in stable if r['stage'] in PRIMARY]
 pooled=[r['outside_median_mib'] for r in prim]
 resv=collections.defaultdict(float)
 for r in stages:
  v=r.get('reserved_bytes_sampled_max_mib')
  if v not in (None,''):resv[r['cond']]=max(resv[r['cond']],float(v))
 OUT.mkdir()
 write('process_stage.csv',stages);write('run_provenance.csv',runs);write('quality_audit.csv',quality);write('repeat_summary.csv',reps);write('block_case_vs_base.csv',diff);write('case_effect_summary.csv',effects);write('source_verification.csv',checks);write('device_gap.csv',gaps)
 hashes.append(dict(path=str(orch_path.relative_to(ROOT)),sha256=raw.sha(orch_path)));write('evidence_sha256.csv',hashes)
 summary=dict(n_runs=len(runs),n_unique_pids=len({r['pid'] for r in runs}),n_gpu_environments=1,counts=dict(counts),stable_process_stage=len(stable),unstable_plateau=len(unstable),max_repeat_range_mib=max(r['range_mib'] for r in reps),primary_pooled_min_mib=min(pooled),primary_pooled_max_mib=max(pooled),primary_pooled_range_mib=max(pooled)-min(pooled),primary_max_abs_diff_vs_base_mib=max(abs(r['cond_minus_base_mib']) for r in diff if r['stage'] in PRIMARY),reserved_sampled_max_mib_by_cond=dict(resv),device_gap_mib_range=[min(g['gap_min_mib'] for g in gaps),max(g['gap_max_mib'] for g in gaps)],query_ms_median=median(lat),query_ms_max=max(lat))
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
 print('STAGE MEDIANS BY CASE');[print(r) for r in reps if r['stage'] in ['after_cuda_init','after_model_load']+PRIMARY]
if __name__=='__main__':main()

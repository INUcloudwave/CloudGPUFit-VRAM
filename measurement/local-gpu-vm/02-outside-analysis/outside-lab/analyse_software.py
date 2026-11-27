#!/usr/bin/env python3
"""CPU-only analysis of software-sX-01: (Sx - S0) per block/model/stage; 100 MiB practical criterion."""
import collections,csv,json,sys
from statistics import median
import audit_qwen_bf16 as raw
ROOT=raw.ROOT;M=2**20
PRIMARY=['after_first_forward','after_first_inference','after_warmup3','after_repeat_block0','after_repeat_block1','after_repeat_block2','final_idle']
SHOW=['after_cuda_init','after_model_load','after_first_forward','final_idle']
def write(out,name,rows):
 if not rows:return
 with (out/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def main(variant):
 suite=f'software-{variant.lower()}-01';base=ROOT/'results'/suite;out=ROOT/'results'/(suite+'-analysis')
 s=json.loads((base/'orchestration.json').read_text())
 assert s['status']=='complete' and len(s['results'])==12 and all(r['exit_code']==0 for r in s['results'])
 assert not out.exists(),'Do not overwrite analysis'
 for name,h in s['source_sha256'].items():assert raw.sha(ROOT/'outside-lab'/name)==h,name
 for name,h in s['input_sha256'].items():assert raw.sha(ROOT/'outside-lab'/name)==h,name
 stages=[];runs=[];quality=[];hashes=[];counts=collections.Counter();lat=[];gaps=[];env=[]
 for job in s['jobs']:
  d=base/job['child'];st,ru,qu,co,la,ha=raw.audit(str(d.relative_to(ROOT/'results')))
  assert len(ru)==1
  for rows in [st,ru,qu]:
   for r in rows:r.update(block=job['block'],image=job['image'],mkey=job['model'])
  stages+=st;runs+=ru;quality+=qu;counts.update(co);lat+=la;hashes+=ha
  mp=next((d/'session').glob('*/manifest.json'));m=json.loads(mp.read_text())
  assert m['status']=='ok' and m['uuid_check']['match'] and m['args']['attention']=='eager' and m['args']['history']=='direct'
  assert m['case']['case_id']==job['case_id'] and m['case']['dtype']=='bf16'
  assert m['environment']['CUDA_MODULE_LOADING']==m['late_cuda_diagnostics']['module_loading']['mode']=='LAZY'
  assert m['image_id']==s['image_ids'][job['image']]
  assert not m['initial_nvml']['errors'] and not m['initial_nvml']['other_gpu_pids']
  g=[]
  for l in mp.with_name('samples.jsonl').read_text().splitlines():
   r=json.loads(l)
   try:dv=r['device_memory_info']['value']['used_bytes'];pv=r['nvml_process_bytes']
   except Exception:continue
   if r.get('kind')=='stable' and pv:g.append((dv-pv)/M)
  gaps.append(dict(block=job['block'],image=job['image'],model=job['model'],gap_min_mib=min(g),gap_max_mib=max(g)))
  env.append(dict(image=job['image'],model=job['model'],block=job['block'],torch=m.get('torch'),cuda_build=m.get('cuda_build'),python=m.get('python'),transformers=m.get('transformers'),cuda_late=json.dumps(m.get('late_cuda_diagnostics',{}).get('runtime_api')),versions=json.dumps({k:v for k,v in (m.get('versions') or {}).items() if 'nvidia' in k or k in('torch','numpy','transformers')},sort_keys=True)))
 assert len({r['pid'] for r in runs})==12 and len({r['gpu_uuid'] for r in runs})==1 and len({r['driver'] for r in runs})==1
 for img in ('S0',variant):assert len({e['torch'] for e in env if e['image']==img})==1,img
 stable=[r for r in stages if r['kind']=='stable' and r['n_clean']>=10]
 groups=collections.defaultdict(list)
 for r in stable:groups[r['image'],r['mkey'],r['stage']].append(r['outside_median_mib'])
 reps=[dict(image=i,model=mk,stage=st,n=len(v),median_mib=median(v),min_mib=min(v),max_mib=max(v),range_mib=max(v)-min(v)) for (i,mk,st),v in sorted(groups.items())]
 idx={(r['block'],r['image'],r['mkey'],r['stage']):r for r in stable};diff=[]
 for r in stable:
  if r['image']!=variant:continue
  b=idx.get((r['block'],'S0',r['mkey'],r['stage']))
  if b:diff.append(dict(block=r['block'],model=r['mkey'],stage=r['stage'],s0_mib=b['outside_median_mib'],variant_mib=r['outside_median_mib'],variant_minus_s0_mib=r['outside_median_mib']-b['outside_median_mib']))
 dg=collections.defaultdict(list)
 for r in diff:dg[r['model'],r['stage']].append(r['variant_minus_s0_mib'])
 eff=[dict(model=mk,stage=st,n_blocks=len(v),median_diff_mib=median(v),min_diff_mib=min(v),max_diff_mib=max(v),all_within_16=all(abs(x)<16 for x in v),all_within_100=all(abs(x)<=100 for x in v)) for (mk,st),v in sorted(dg.items())]
 prim=[r for r in diff if r['stage'] in PRIMARY]
 out.mkdir()
 for n,rows in [('process_stage.csv',stages),('run_provenance.csv',runs),('quality_audit.csv',quality),('repeat_summary.csv',reps),('block_variant_vs_s0.csv',diff),('effect_summary.csv',eff),('device_gap.csv',gaps),('software_versions.csv',env),('evidence_sha256.csv',hashes)]:write(out,n,rows)
 summary=dict(variant=variant,spec=s['variant_spec'],image_ids=s['image_ids'],torch={i:sorted({e['torch'] for e in env if e['image']==i}) for i in ('S0',variant)},n_runs=len(runs),counts=dict(counts),stable_process_stage=len(stable),unstable_plateau=sum(1 for r in stable if r['last10_span_mib']>8),max_repeat_range_mib=max(r['range_mib'] for r in reps),primary_pairs=len(prim),primary_diff_range_mib=[min(r['variant_minus_s0_mib'] for r in prim),max(r['variant_minus_s0_mib'] for r in prim)],primary_all_within_100=all(abs(r['variant_minus_s0_mib'])<=100 for r in prim),primary_all_within_16=all(abs(r['variant_minus_s0_mib'])<16 for r in prim),device_gap_range_mib=[min(g['gap_min_mib'] for g in gaps),max(g['gap_max_mib'] for g in gaps)],query_ms_median=median(lat),query_ms_max=max(lat))
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
 for r in reps:
  if r['stage'] in SHOW:print(r)
if __name__=='__main__':main(sys.argv[1])

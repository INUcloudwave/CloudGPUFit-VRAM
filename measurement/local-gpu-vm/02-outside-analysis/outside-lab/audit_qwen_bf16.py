#!/usr/bin/env python3
"""CPU-only independent raw audit. Writes only to a new output directory."""
import collections,csv,hashlib,json,sys
from pathlib import Path
from statistics import median
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'results/qwen-bf16-01-independent-audit'
M=2**20

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,rows):
 if not rows:return
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def flags(r):
 f=[];n=r.get('nvml_process_bytes');v=r.get('reserved_bytes')
 if n is None or v is None:f.append('missing_pair')
 elif n-v<0:f.append('negative_outside')
 if r['query_end_ns']-r['query_begin_ns']>5_000_000:f.append('query_over_5ms')
 if r.get('reserved_bracket_changed') or r.get('nvml_bracket_changed'):f.append('bracket_changed')
 if any(r.get(k,{}).get('other_gpu_pids') for k in ['nvml','nvml_after']):f.append('other_process')
 if r.get('errors') or any(r.get(k,{}).get('errors') for k in ['nvml','nvml_after']):f.append('query_error')
 return f

def audit(suite,select=lambda m:True):
 stages=[];runs=[];quality=[];lat=[];counts=collections.Counter();hashes=[]
 for mp in sorted((ROOT/'results'/suite/'session').glob('*/manifest.json')):
  m=json.loads(mp.read_text())
  if not select(m):continue
  base=dict(suite=suite,run_id=m['run_id'],model=m['case']['model'],dtype=m['case']['dtype'],history=m['args']['history'],repeat=m['independent_repeat'],pid=m['pid'])
  rows=[json.loads(l) for l in mp.with_name('samples.jsonl').read_text().splitlines()];groups=collections.defaultdict(list)
  for r in rows:
   assert r['pid']==m['pid'] and r['run_id']==m['run_id']
   n=r.get('nvml_process_bytes');v=r.get('reserved_bytes');outside=None if n is None or v is None else n-v
   assert outside==r.get('outside_bytes'),'paired arithmetic mismatch'
   for k in ['reserved','nvml']:
    key='reserved_bracket_changed' if k=='reserved' else 'nvml_bracket_changed'
    if key in r:
     a,b=('torch','torch_after') if k=='reserved' else ('nvml','nvml_after');metric='reserved_bytes' if k=='reserved' else 'nvml_process_bytes'
     assert r[key]==(r[a].get(metric)!=r[b].get(metric))
   f=flags(r);expected=outside is None and r['stage'] in ['before_torch_import','after_torch_import'] and r['torch']['initialized']==False and r['nvml']['own_match_status']=='absent'
   counts['samples']+=1;counts[r['kind']]+=1;counts['expected_preinit_missing']+=expected
   for x in f:counts[x]+=1
   groups[r['stage'],r['kind']].append((r,f))
   q=(r['query_end_ns']-r['query_begin_ns'])/1e6;lat.append(q)
   if f:quality.append(dict(**base,stage=r['stage'],kind=r['kind'],query_begin_ns=r['query_begin_ns'],query_ms=q,flags=';'.join(f),expected_preinit=expected,outside_mib=outside/M if outside is not None else None))
   if r.get('device_memory_info',{}).get('error'):counts['device_memory_info_error']+=1
  for (stage,kind),pairs in groups.items():
   good=[r for r,f in pairs if not f];vals=[(r['nvml_process_bytes']-r['reserved_bytes'])/M for r in good]
   row=dict(**base,stage=stage,kind=kind,n_raw=len(pairs),n_clean=len(good),outside_median_mib=median(vals) if vals else None,outside_min_mib=min(vals) if vals else None,outside_max_mib=max(vals) if vals else None,last10_span_mib=max(vals[-10:])-min(vals[-10:]) if len(vals)>=10 else None)
   for k in ['nvml_process_bytes','allocated_bytes','reserved_bytes']:
    row[k+'_paired_median_mib']=median(r[k]/M for r in good) if good else None
    vv=[r[k]/M for r,f in pairs if r.get(k) is not None];row[k+'_sampled_max_mib']=max(vv) if vv else None
   stages.append(row)
  runs.append(dict(**base,status=m['status'],gpu_uuid=m['gpu']['uuid'],driver=m['gpu']['driver'],image_id=m['image_id'],probe_sha256=m['source_sha256'],criteria_sha256=m['criteria_sha256'],revision=m['revision']['revision'],input_sha256=json.dumps(m['input_sha256'],sort_keys=True),module_loading=m['environment']['CUDA_MODULE_LOADING'],runtime_mode=m['late_cuda_diagnostics']['module_loading']['mode'],torch=m['torch'],cuda_build=m['cuda_build'],transformers=m['transformers'],python=m['python'],started_kst=datetime.fromtimestamp(m['started_wall_ns']/1e9,timezone(timedelta(hours=9))).isoformat(),attention=m['args']['attention'],uuid_match=m['uuid_check']['match']))
  for p in [mp,mp.with_name('samples.jsonl')]:hashes.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
 return stages,runs,quality,dict(counts),lat,hashes

def main():
 OUT.mkdir(exist_ok=False)
 stages,runs,quality,counts,lat,hashes=audit('qwen-bf16-01')
 suite=json.loads((ROOT/'results/qwen-bf16-01/session/suite-manifest.json').read_text())
 assert suite['status']=='complete' and len(suite['jobs'])==len(suite['results'])==len(runs)==12
 assert all(r['exit_code']==0 for r in suite['results']) and all(r['status']=='ok' for r in runs)
 assert len({r['pid'] for r in runs})==12 and len({r['gpu_uuid'] for r in runs})==1
 assert {r['run_id'] for r in runs}=={r['run_id'] for r in suite['results']}
 checks=[]
 for name,h in suite['source_sha256'].items():
  p=ROOT/'outside-lab'/name;checks.append(dict(path=str(p.relative_to(ROOT)),recorded=h,current=sha(p),matches=sha(p)==h))
 for name,h in suite['input_sha256'].items():
  p=ROOT/'outside-lab'/Path(name).name;checks.append(dict(path=str(p.relative_to(ROOT)),recorded=h,current=sha(p),matches=sha(p)==h))
 assert all(c['matches'] for c in checks)
 plan=json.loads((ROOT/'outside-lab/qwen-bf16-plan.json').read_text());pins=json.loads((ROOT/'outside-lab/qwen-revisions.json').read_text())
 for mp in (ROOT/'results/qwen-bf16-01/session').glob('*/manifest.json'):
  m=json.loads(mp.read_text());assert m['case']==next(c for c in plan['cases'] if c['case_id']==m['case']['case_id']);assert m['revision']==pins[m['case']['model']]
  assert m['initial_nvml']['errors']==[] and m['initial_nvml']['other_gpu_pids']==[]
 # Independently reproduce every analyser process-stage clean median/count.
 orig={(r['run_id'],r['stage'],r['kind']):r for r in csv.DictReader((ROOT/'results/qwen-bf16-01-analysis/process_stage.csv').open())}
 for r in stages:
  a=orig[r['run_id'],r['stage'],r['kind']];assert int(a['n_clean'])==r['n_clean'];assert (float(a['median']) if a['median'] else None)==r['outside_median_mib']
 stable=[r for r in stages if r['kind']=='stable' and r['n_clean']>=10]
 groups=collections.defaultdict(list)
 for r in stable:groups[r['model'],r['history'],r['stage']].append(r)
 repeats=[]
 for (model,history,stage),rs in groups.items():
  vs=[r['outside_median_mib'] for r in rs];repeats.append(dict(model=model,history=history,stage=stage,n_processes=len(rs),median_mib=median(vs),min_mib=min(vs),max_mib=max(vs),range_mib=max(vs)-min(vs)))
 indexed={(r['model'],r['history'],r['repeat'],r['stage']):r for r in stable};hist=[]
 for r in stable:
  if r['history']!='direct':continue
  p=indexed[r['model'],'performance_first',r['repeat'],r['stage']]
  hist.append(dict(model=r['model'],repeat=r['repeat'],stage=r['stage'],performance_minus_direct_mib=p['outside_median_mib']-r['outside_median_mib']))
 prior,pruns,_,pc,_,ph=audit('large-01',lambda m:m['case']['model']=='openai-community/gpt2');hashes+=ph
 pi={(r['history'],r['repeat'],r['stage']):r for r in prior if r['kind']=='stable' and r['n_clean']>=10};comparison=[]
 for r in stable:
  if r['model']!='openai-community/gpt2':continue
  a=pi[r['history'],r['repeat'],r['stage']];comparison.append(dict(history=r['history'],repeat=r['repeat'],stage=r['stage'],bf16_mib=r['outside_median_mib'],fp16_mib=a['outside_median_mib'],difference_mib=r['outside_median_mib']-a['outside_median_mib'],bf16_suite='qwen-bf16-01',fp16_suite='large-01'))
 write('paired_process_stage.csv',stages);write('repeat_summary.csv',repeats);write('history_comparison.csv',hist);write('gpt2_precision_cross_suite.csv',comparison);write('quality_audit.csv',quality);write('run_provenance.csv',runs+pruns);write('source_input_verification.csv',checks)
 for p in [ROOT/'results/qwen-bf16-01/session/suite-manifest.json',ROOT/'results/qwen-bf16-01/container-inspect.json',ROOT/'qwen-bf16-schedule.json',ROOT/'results/qwen-smoke-01/smoke/manifest.json',ROOT/'results/qwen-bf16-smoke-01/smoke/manifest.json']:
  hashes.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
 write('evidence_sha256.csv',hashes)
 summary=dict(counts=counts,n_process_stage=len(stages),stable_usable=len(stable),repeat_groups=len(repeats),max_repeat_range_mib=max(r['range_mib'] for r in repeats),max_last10_span_mib=max(r['last10_span_mib'] for r in stable),stable_min_clean=min(r['n_clean'] for r in stable),history_pairs=len(hist),history_max_abs_diff=max(abs(r['performance_minus_direct_mib']) for r in hist),cross_suite_pairs=len(comparison),cross_suite_max_abs_diff=max(abs(r['difference_mib']) for r in comparison),query_ms_min=min(lat),query_ms_median=median(lat),query_ms_p95=sorted(lat)[int(.95*(len(lat)-1))],query_ms_max=max(lat),original_analysis_agrees=True)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
 for model in sorted({r['model'] for r in stages}):
  rs=[r for r in stages if r['model']==model];print(model,'max reserved',max(r['reserved_bytes_sampled_max_mib'] or 0 for r in rs),'max nvml',max(r['nvml_process_bytes_sampled_max_mib'] or 0 for r in rs))
 print('DATES',runs[0]['started_kst'],pruns[0]['started_kst'])
if __name__=='__main__':main()

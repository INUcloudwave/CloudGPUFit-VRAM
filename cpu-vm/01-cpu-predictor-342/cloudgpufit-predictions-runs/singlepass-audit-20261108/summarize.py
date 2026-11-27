import json
from pathlib import Path
BASE=Path('/home/inuos/cloudgpufit-predictions')
old={r['case']:r for r in json.loads((BASE/'recovery-audit-20261108/summary.json').read_text()) if r['status']=='ok'}
rows=[]
for d in sorted(BASE.glob('singlepass-check-*')):
 for f in d.glob('*.prediction.json'):
  p=json.loads(f.read_text());events=[]
  for line in f.with_name(f.name.replace('.prediction.json','.log')).read_text(errors='replace').splitlines():
   if line.startswith('CGF_PROGRESS '):events.append(json.loads(line[13:]))
  r=old[p['case_id']]
  rows.append({'case':p['case_id'],'status':p['status'],'comparison':p['reference_comparison'],'version':p['prediction_version'],'warmup':p['trace_summary']['warmup_invocations'],'measured':p['trace_summary']['measured_invocations'],
  'old_seconds':r['runner_seconds'],'new_seconds':p['runner_elapsed_seconds'],'speedup':r['runner_seconds']/p['runner_elapsed_seconds'],
  'old_peak_rss_gib':r['max_rss_gib'],'new_peak_rss_gib':max(e.get('max_rss_bytes',0) for e in events)/1024**3,
  'old_cgroup_peak_gib':r['cgroup_peak_gib'],'new_cgroup_peak_gib':max(e.get('cgroup_memory_peak_bytes',0) for e in events)/1024**3,
  'allocated':p['predicted_allocated_bytes'],'reserved':p['predicted_reserved_bytes'],
  'phases':{e['phase']:e['elapsed_seconds'] for e in events if e['event']=='phase_end'}})
print(json.dumps(rows,indent=2))

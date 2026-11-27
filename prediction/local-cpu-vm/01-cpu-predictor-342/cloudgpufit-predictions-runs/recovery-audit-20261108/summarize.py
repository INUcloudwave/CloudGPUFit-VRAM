import json
from pathlib import Path
root=Path('/home/inuos/cloudgpufit-predictions')
rows=[]
for d in sorted(root.glob('recovery-check-*')):
 for f in sorted(d.glob('*.prediction.json')):
  p=json.loads(f.read_text()); row={'run':d.name,'case':p['case_id'],'status':p['status'],'error_kind':p.get('error_kind'),'comparison':p.get('reference_comparison'),'allocated':p.get('predicted_allocated_bytes'),'reserved':p.get('predicted_reserved_bytes'),'runner_seconds':p.get('runner_elapsed_seconds')}
  log=f.with_name(f.name.replace('.prediction.json','.log'))
  events=[]
  if log.exists():
   for line in log.read_text(errors='replace').splitlines():
    if line.startswith('CGF_PROGRESS '):events.append(json.loads(line[13:]))
  row['max_rss_gib']=max((e.get('max_rss_bytes',0) for e in events),default=0)/1024**3
  row['cgroup_peak_gib']=max((e.get('cgroup_memory_peak_bytes',0) for e in events),default=0)/1024**3
  row['phases']={e['phase']:round(e['elapsed_seconds'],4) for e in events if e['event']=='phase_end'}
  rows.append(row)
print(json.dumps(rows,indent=2))

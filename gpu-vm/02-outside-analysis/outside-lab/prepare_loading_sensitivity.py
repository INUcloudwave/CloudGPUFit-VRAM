#!/usr/bin/env python3
"""Prepare/dry-run by default. --execute explicitly starts sequential local GPU runs."""
import argparse,hashlib,json,random,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
LAB=Path(__file__).resolve().parent
GPU='GPU-17eb7299-8b39-914d-7c5b-26908502904f'
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--sudo',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--cache',default='/home/inugpu/cloudgpufit-bench/hf-cache');a=p.parse_args()
 rng=random.Random(20261108);jobs=[]
 for block in range(3):
  modes=['LAZY','EAGER'];rng.shuffle(modes)
  for mode in modes:jobs.append(dict(block=block,mode=mode))
 commands=[]
 for ix,j in enumerate(jobs):
  out=(a.output.resolve() if a.output else LAB.parent/'results/loading-sensitivity-01')/f'{ix:02d}_block{j["block"]}_{j["mode"]}'
  cmd=[sys.executable,str(LAB/'container_run_qwen_bf16.py'),'pilot','--plan','loading-sensitivity-plan.json','--revisions','qwen-revisions.json','--repeats','1','--timeout','900','--cache',a.cache,'--gpu-uuid',GPU,'--env','CUDA_MODULE_LOADING='+j['mode'],'--output',str(out)]
  if a.sudo:cmd.append('--sudo')
  commands.append(dict(**j,command=cmd))
 schedule=dict(seed=20261108,jobs=commands,total_workers=12,global_repeat='block; child repeat is always 0; do not pool suite rows without block/mode',source_sha256={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in [Path(__file__),LAB/'probe.py',LAB/'pilot.py',LAB/'container_run_qwen_bf16.py',LAB/'loading-sensitivity-plan.json',LAB/'qwen-revisions.json',LAB/'criteria.json']})
 if not a.execute:print(json.dumps(schedule,indent=2));return
 if not a.output:p.error('--execute requires a new --output directory')
 a.output.mkdir(parents=True,exist_ok=False)
 schedule.update(status='running',started_utc=datetime.now(timezone.utc).isoformat(),results=[])
 def save():(a.output/'orchestration.json').write_text(json.dumps(schedule,indent=2))
 save()
 try:
  for ix,j in enumerate(commands):
   with (a.output/f'{ix:02d}.log').open('w') as f:code=subprocess.call(j['command'],stdout=f,stderr=subprocess.STDOUT)
   schedule['results'].append(dict(block=j['block'],mode=j['mode'],exit_code=code));save()
   if code:raise RuntimeError(f'Nonzero exit {code}; inspect log; no retry')
   child=Path(j['command'][j['command'].index('--output')+1])/'session'
   suite=json.loads((child/'suite-manifest.json').read_text());ms=[json.loads(x.read_text()) for x in child.glob('*/manifest.json')]
   if suite['status']!='complete' or len(ms)!=2 or any(m['status']!='ok' or m['late_cuda_diagnostics']['module_loading']['mode']!=j['mode'] or not m['uuid_check']['match'] for m in ms):raise RuntimeError('Post-run status or actual loading mode mismatch')
  schedule['status']='complete'
 except BaseException as e:schedule.update(status='error_or_interrupted',error=str(e));raise
 finally:save()
if __name__=='__main__':main()

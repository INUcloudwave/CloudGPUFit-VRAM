#!/usr/bin/env python3
"""Balanced 10-block confirmation, direct fresh worker only. No GPU in dry-run."""
import argparse,datetime,hashlib,json,random,subprocess,sys
from pathlib import Path
LAB=Path(__file__).resolve().parent
IMAGE='sha256:457da255e6e2de7ee05ff7814674c90abcf6d1ed04e73df3cc4eebcdb31fb032'
PROBE='7df9a02ce9ca938b9702ee2c9fd7d0414d0a4cfc0cfe5ab87c97d7a069243d88'
GPU='GPU-17eb7299-8b39-914d-7c5b-26908502904f'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path,default=LAB.parent/'results/loading-confirmation-01');p.add_argument('--cache',type=Path,default=Path('/home/inugpu/cloudgpufit-bench/hf-cache'));a=p.parse_args()
 orders=[['LAZY','EAGER'] for _ in range(5)]+[['EAGER','LAZY'] for _ in range(5)];random.Random(20261109).shuffle(orders)
 jobs=[dict(block=b,mode=m,run_id=f'block{b:02d}_{m}_gpt2_bf16_direct',child=f'{2*b+i:02d}_block{b:02d}_{m}') for b,order in enumerate(orders) for i,m in enumerate(order)]
 sources={x.name:sha(x) for x in LAB.glob('*.py')};inputs={n:sha(LAB/n) for n in ['loading-confirmation-plan.json','qwen-revisions.json','criteria.json','LOADING_CONFIRMATION_DESIGN_KO.md']}
 schedule=dict(experiment_id='outside-loading-confirmation-01',seed=20261109,jobs=jobs,total_workers=20,source_sha256=sources,input_sha256=inputs,image_id=IMAGE,gpu_uuid=GPU,history='direct',timeout_seconds=900)
 if not a.execute:print(json.dumps(schedule,indent=2));return
 assert sha(LAB/'probe.py')==PROBE,'Probe mismatch'
 assert a.cache.is_dir(),'Cache missing'
 info=json.loads(subprocess.check_output(['docker','image','inspect',IMAGE],text=True))[0];assert info['Id']==IMAGE
 root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
 schedule.update(status='running',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),results=[])
 def save():(root/'orchestration.json').write_text(json.dumps(schedule,indent=2))
 save()
 try:
  for j in jobs:
   assert all(sha(LAB/n)==h for n,h in sources.items()),'Source changed during run'
   assert all(sha(LAB/n)==h for n,h in inputs.items()),'Input changed during run'
   d=root/j['child'];d.mkdir();(d/'session').mkdir()
   (d/'container-inspect.json').write_text(json.dumps(info,indent=2))
   worker=['python','/lab/probe.py','--output','/results/session/'+j['run_id'],'--gpu-uuid',GPU,'--run-id',j['run_id'],'--repeat',str(j['block']),'--scenario','model','--plan','/lab/loading-confirmation-plan.json','--revisions','/lab/qwen-revisions.json','--case-id','core__decoder_llm__small__bf16__01','--history','direct','--threads','4']
   name='outside-confirm-'+j['run_id'].lower().replace('_','-')
   cmd=['docker','run','--name',name,'--rm','--pull=never','--network','none','--gpus','device='+GPU,'--pid=host','-v',str(LAB)+':/lab:ro','-w','/lab','-v',str(a.cache.resolve())+':/hf:ro','-v',str(d)+':/results']
   env={'PYTHONDONTWRITEBYTECODE':'1','OUTSIDE_IMAGE_ID':IMAGE,'OUTSIDE_INSTANCE_LABEL':'local-rtx3090','OUTSIDE_IMAGE_REPODIGESTS':json.dumps(info.get('RepoDigests',[])),'CUDA_VISIBLE_DEVICES':GPU,'CUDA_MODULE_LOADING':j['mode'],'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','HF_HOME':'/hf','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
   for k,v in env.items():cmd+=['-e',k+'='+v]
   cmd+=[IMAGE]+worker
   suite=dict(status='running',jobs=[j],results=[],source_sha256=sources,input_sha256=inputs,command=cmd)
   sp=d/'session/suite-manifest.json';sp.write_text(json.dumps(suite,indent=2))
   print('START',j['run_id'],flush=True)
   with (d/'worker.log').open('w') as f:
    try:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900).returncode
    except subprocess.TimeoutExpired:
     subprocess.run(['docker','stop','--time','10',name],stdout=f,stderr=subprocess.STDOUT,timeout=30);code=124
   result=dict(block=j['block'],mode=j['mode'],run_id=j['run_id'],exit_code=code)
   suite['results']=[result];suite['status']='complete' if code==0 else 'error';sp.write_text(json.dumps(suite,indent=2))
   schedule['results'].append(result);save()
   if code:raise RuntimeError(f'Worker exit {code}; inspect {d}/worker.log; no retry')
   m=json.loads((d/'session'/j['run_id']/'manifest.json').read_text())
   assert m['status']=='ok' and m['uuid_check']['match'] and m['late_cuda_diagnostics']['module_loading']['mode']==j['mode'],'Worker status/UUID/actual mode mismatch'
   print('OK',j['run_id'],flush=True)
  schedule['status']='complete'
 except BaseException as e:schedule.update(status='error_or_interrupted',error=str(e));raise
 finally:save()
if __name__=='__main__':main()

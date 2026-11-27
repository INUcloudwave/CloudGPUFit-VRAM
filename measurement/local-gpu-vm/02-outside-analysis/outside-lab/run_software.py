#!/usr/bin/env python3
"""Software/image sensitivity on RTX3090: S0 control vs one variant image, 3 paired blocks x 2 models.
Dry-run prints schedule; --execute runs on GPU. Same probe/plan/env as previous suites."""
import argparse,datetime,hashlib,json,random,subprocess
from pathlib import Path
LAB=Path(__file__).resolve().parent
S0='sha256:457da255e6e2de7ee05ff7814674c90abcf6d1ed04e73df3cc4eebcdb31fb032'
PROBE='7df9a02ce9ca938b9702ee2c9fd7d0414d0a4cfc0cfe5ab87c97d7a069243d88'
GPU='GPU-17eb7299-8b39-914d-7c5b-26908502904f'
MODELS={'gpt2':'core__decoder_llm__small__bf16__01','qwen7b':'core__decoder_llm__qwen7b__bf16__01'}
SEEDS={'S1':20261121,'S2':20261122,'S3':20261123,'S4':20261124}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('variant',choices=sorted(SEEDS));p.add_argument('--execute',action='store_true')
 p.add_argument('--cache',type=Path,default=Path('/home/inugpu/cloudgpufit-bench/hf-cache'));a=p.parse_args()
 var=json.loads((LAB/'images/variants.json').read_text())[a.variant]
 rng=random.Random(SEEDS[a.variant]);jobs=[]
 for b in range(3):
  units=[(img,m) for img in ('S0',a.variant) for m in MODELS];rng.shuffle(units)
  for i,(img,m) in enumerate(units):
   rid=f'block{b:02d}_{img.lower()}_{m}_bf16_direct'
   jobs.append(dict(block=b,image=img,model=m,case_id=MODELS[m],run_id=rid,child=f'{4*b+i:02d}_{rid}'))
 sources={x.name:sha(x) for x in LAB.glob('*.py')}
 inputs={n:sha(LAB/n) for n in ['qwen-bf16-plan.json','qwen-revisions.json','criteria.json','practical-outside-criteria.json','SOFTWARE_DESIGN_KO.md','images/variants.json','images/Dockerfile.variant']}
 out=LAB.parent/f'results/software-{a.variant.lower()}-01'
 schedule=dict(experiment_id=f'outside-software-{a.variant.lower()}-01',variant=a.variant,variant_spec=var,seed=SEEDS[a.variant],jobs=jobs,total_workers=len(jobs),source_sha256=sources,input_sha256=inputs,s0_image_id=S0,gpu_uuid=GPU,attention='eager',history='direct',timeout_seconds=900)
 if not a.execute:print(json.dumps(schedule,indent=2));return
 assert sha(LAB/'probe.py')==PROBE,'Probe mismatch'
 assert a.cache.is_dir()
 mount=var.get('mode')=='mount'
 vref=var['base_image'] if mount else var['tag']
 if mount:assert Path(var['pkgdir'],'environment.lock.txt').is_file(),'Variant packages not installed'
 infos={'S0':json.loads(subprocess.check_output(['docker','image','inspect',S0],text=True))[0],
        a.variant:json.loads(subprocess.check_output(['docker','image','inspect',vref],text=True))[0]}
 assert infos['S0']['Id']==S0
 ids={'S0':S0,a.variant:infos[a.variant]['Id']};schedule['image_ids']=ids
 root=out.resolve();root.mkdir(parents=True,exist_ok=False)
 (root/'variant-image-inspect.json').write_text(json.dumps(infos[a.variant],indent=2))
 lock=Path(var['pkgdir'],'environment.lock.txt').read_text() if mount else subprocess.check_output(['docker','run','--rm','--network','none',ids[a.variant],'cat','/opt/environment.lock.txt'],text=True)
 (root/'variant-environment.lock.txt').write_text(lock)
 schedule.update(status='running',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),results=[])
 def save():(root/'orchestration.json').write_text(json.dumps(schedule,indent=2))
 save()
 try:
  for j in jobs:
   assert all(sha(LAB/n)==h for n,h in sources.items()),'Source changed during run'
   assert all(sha(LAB/n)==h for n,h in inputs.items()),'Input changed during run'
   img=ids[j['image']];d=root/j['child'];d.mkdir();(d/'session').mkdir()
   worker=['python','/lab/probe.py','--output','/results/session/'+j['run_id'],'--gpu-uuid',GPU,'--run-id',j['run_id'],'--repeat',str(j['block']),'--scenario','model','--plan','/lab/qwen-bf16-plan.json','--revisions','/lab/qwen-revisions.json','--case-id',j['case_id'],'--history','direct','--threads','4','--attention','eager']
   name='outside-sw-'+a.variant.lower()+'-'+j['run_id'].lower().replace('_','-')
   cmd=['docker','run','--name',name,'--rm','--pull=never','--network','none','--gpus','device='+GPU,'--pid=host','-v',str(LAB)+':/lab:ro','-w','/lab','-v',str(a.cache.resolve())+':/hf:ro','-v',str(d)+':/results']
   env={'PYTHONDONTWRITEBYTECODE':'1','OUTSIDE_IMAGE_ID':img,'OUTSIDE_INSTANCE_LABEL':'local-rtx3090','OUTSIDE_IMAGE_REPODIGESTS':json.dumps(infos[j['image']].get('RepoDigests',[])),'CUDA_VISIBLE_DEVICES':GPU,'CUDA_MODULE_LOADING':'LAZY','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','HF_HOME':'/hf','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
   if mount and j['image']==a.variant:
    cmd+=['-v',var['pkgdir']+':/opt/pkgs:ro'];env['PYTHONPATH']='/opt/pkgs'
   else:env.pop('PYTHONPATH',None)
   for k,v in env.items():cmd+=['-e',k+'='+v]
   cmd+=[img]+worker
   suite=dict(status='running',jobs=[j],results=[],source_sha256=sources,input_sha256=inputs,command=cmd)
   sp=d/'session/suite-manifest.json';sp.write_text(json.dumps(suite,indent=2))
   print('START',j['run_id'],datetime.datetime.now().strftime('%H:%M:%S'),flush=True)
   with (d/'worker.log').open('w') as f:
    try:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900).returncode
    except subprocess.TimeoutExpired:
     subprocess.run(['docker','stop','--time','10',name],stdout=f,stderr=subprocess.STDOUT,timeout=30);code=124
   result=dict(block=j['block'],image=j['image'],model=j['model'],run_id=j['run_id'],exit_code=code)
   suite['results']=[result];suite['status']='complete' if code==0 else 'error';sp.write_text(json.dumps(suite,indent=2))
   schedule['results'].append(result);save()
   if code:raise RuntimeError(f'Worker exit {code}; inspect {d}/worker.log; no retry')
   m=json.loads((d/'session'/j['run_id']/'manifest.json').read_text())
   assert m['status']=='ok' and m['uuid_check']['match'] and m['late_cuda_diagnostics']['module_loading']['mode']=='LAZY' and m['case']['case_id']==j['case_id'],'Worker status/UUID/mode/case mismatch'
   print('OK',j['run_id'],m.get('torch'),datetime.datetime.now().strftime('%H:%M:%S'),flush=True)
  schedule['status']='complete'
 except BaseException as e:schedule.update(status='error_or_interrupted',error=str(e));raise
 finally:save()
if __name__=='__main__':main()

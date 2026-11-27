#!/usr/bin/env python3
"""Step 2 on the AWS instance: 3 blocks x {S0,S4} x models, fresh container per worker, then tar results.
Usage: python3 run.py [--label g6.xlarge-pilot] [--dry-run]"""
import argparse,datetime,hashlib,json,random,subprocess,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
PROBE='7df9a02ce9ca938b9702ee2c9fd7d0414d0a4cfc0cfe5ab87c97d7a069243d88'
MODELS={'core__decoder_llm__small__bf16__01':'gpt2','core__decoder_llm__qwen7b__bf16__01':'qwen7b'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sh(*c):return subprocess.check_output(c,text=True).strip()
def main():
 p=argparse.ArgumentParser();p.add_argument('--label',default='');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
 if a.label and (not all(c.isalnum() or c in '.-_' for c in a.label) or a.label in ('.','..')):p.error('Label must contain only letters, numbers, dot, dash, underscore')
 assert sha(HERE/'lab/probe.py')==PROBE,'probe.py changed'
 inst=dict(l.split('=',1) for l in (HERE/'env/instance.txt').read_text().splitlines() if '=' in l)
 label=a.label or inst.get('instance-type','unknown')
 plan=(HERE/'env/plan-path.txt').read_text().strip();cases=[c['case_id'] for c in json.loads((HERE/plan).read_text())['cases']]
 gpus=sh('nvidia-smi','--query-gpu=uuid','--format=csv,noheader').splitlines();assert len(gpus)==1,'Expose one GPU';gpu=gpus[0].strip()
 assert not sh('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'),'Other GPU process running'
 images={k:json.loads(sh('docker','image','inspect','outside-aws:'+k.lower()))[0] for k in ('S0','S4')}
 rng=random.Random(20261201);jobs=[]
 for b in range(3):
  units=[(i,c) for i in images for c in cases];rng.shuffle(units)
  for n,(i,c) in enumerate(units):
   rid=f'block{b:02d}_{i.lower()}_{MODELS[c]}_bf16_direct';jobs.append(dict(block=b,image=i,case_id=c,model=MODELS[c],run_id=rid,child=f'{len(units)*b+n:02d}_{rid}'))
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 out=HERE/f'results/{label}-{stamp}'
 sched=dict(schema_version=2,seed=20261201,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256={f.name:sha(f) for f in [HERE/'run.py',HERE/'setup.sh',HERE/'run_all.sh']},label=label,instance=inst,gpu_uuid=gpu,plan=plan,image_ids={k:v['Id'] for k,v in images.items()},jobs=jobs,probe_sha256=PROBE,input_sha256={f.name:sha(f) for f in (HERE/'lab').iterdir() if f.is_file()})
 if a.dry_run:print(json.dumps(sched,indent=2));return
 out.mkdir(parents=True,exist_ok=False);sched.update(status='running',results=[])
 save=lambda:(out/'orchestration.json').write_text(json.dumps(sched,indent=2));save()
 for i,info in images.items():(out/f'image-{i}.json').write_text(json.dumps(info,indent=2))
 try:
  for j in jobs:
   assert all(sha(HERE/'lab'/n)==h for n,h in sched['input_sha256'].items()),'Input/source changed'
   d=out/j['child'];(d/'session').mkdir(parents=True)
   img=images[j['image']]['Id'];name='outside-aws-'+j['run_id'].replace('_','-')
   env={'PYTHONDONTWRITEBYTECODE':'1','OUTSIDE_IMAGE_ID':img,'OUTSIDE_INSTANCE_LABEL':label,'OUTSIDE_IMAGE_REPODIGESTS':'[]','CUDA_VISIBLE_DEVICES':gpu,'CUDA_MODULE_LOADING':'LAZY','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','HF_HOME':'/hf','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
   cmd=['docker','run','--name',name,'--rm','--pull=never','--network','none','--gpus','device='+gpu,'--pid=host','-v',str(HERE/'lab')+':/lab:ro','-w','/lab','-v',str(HERE/'hf-cache')+':/hf:ro','-v',str(d)+':/results']
   for k,v in env.items():cmd+=['-e',k+'='+v]
   cmd+=[img,'python','/lab/probe.py','--output','/results/session/'+j['run_id'],'--gpu-uuid',gpu,'--run-id',j['run_id'],'--repeat',str(j['block']),'--scenario','model','--plan','/lab/'+Path(plan).name,'--revisions','/lab/qwen-revisions.json','--case-id',j['case_id'],'--history','direct','--threads','4','--attention','eager']
   (d/'session/suite-manifest.json').write_text(json.dumps(dict(status='running',jobs=[j],command=cmd),indent=2))
   print('START',j['run_id'],datetime.datetime.now().strftime('%H:%M:%S'),flush=True)
   with (d/'worker.log').open('w') as f:
    try:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1200).returncode
    except subprocess.TimeoutExpired:subprocess.run(['docker','stop','--time','10',name],timeout=30);code=124
   sched['results'].append(dict(run_id=j['run_id'],exit_code=code));save()
   (d/'session/suite-manifest.json').write_text(json.dumps(dict(status='complete' if code==0 else 'error',jobs=[j],results=[dict(exit_code=code)],command=cmd),indent=2))
   if code:raise RuntimeError(f'worker exit {code}: see {d}/worker.log (no retry)')
   m=json.loads((d/'session'/j['run_id']/'manifest.json').read_text())
   assert m['status']=='ok' and m['uuid_check']['match'] and m['late_cuda_diagnostics']['module_loading']['mode']=='LAZY','Manifest status/UUID/runtime mode mismatch'
   assert m['case']['dtype']=='bf16' and m['case']['case_id']==j['case_id'] and m['args']['attention']=='eager','Case/attention mismatch'
   print('OK',j['run_id'],m.get('torch'),flush=True)
  sched['status']='complete'
 except BaseException as e:sched.update(status='error_or_interrupted',error=str(e));raise
 finally:
  save()
  tgz=HERE/f'results/{out.name}.tar.gz'
  with tarfile.open(tgz,'w:gz') as t:t.add(out,arcname=out.name);t.add(HERE/'env',arcname=out.name+'/env')
  print('RESULT_ARCHIVE',tgz,flush=True)
if __name__=='__main__':main()

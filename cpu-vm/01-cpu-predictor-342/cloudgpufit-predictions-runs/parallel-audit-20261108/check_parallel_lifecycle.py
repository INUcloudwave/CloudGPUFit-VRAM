import json,signal,subprocess,sys,time,uuid
from pathlib import Path
sys.path.insert(0,'/home/inuos/CloudGPUFit-cpu-singlepass-20261108/tools')
import run_prediction_recovery as r
from prediction_scheduler import run_cases
OUT=Path(__file__).resolve().parent
IMAGE='cloudgpufit-cpu-validation:20261108'
def interrupt(*_):raise KeyboardInterrupt
signal.signal(signal.SIGTERM,interrupt)
def execute_factory(token,mode):
 def execute(case,cancel):
  cid=case['case_id'];name=token+'-'+cid
  timeout=1 if mode=='timeout' and cid=='a' else 60
  code="import sys,time,json;sys.stdin.read();time.sleep(%d);print(json.dumps({'status':'ok'}))"%(120 if mode=='cancel' or cid=='a' else 2)
  c=r.job_command(IMAGE,'/home/inuos/xmem-work','/home/inuos/.cache/huggingface',64*1024**2,1)
  c=c[:c.index(IMAGE)+1]+['-u','-c',code]
  return r.run_job(name,c,{},OUT/(name+'.log'),timeout,cancel)
 return execute
if len(sys.argv)>1:
 token,mode=sys.argv[1:]
 results={}
 def record(case,result):results[case['case_id']]=result
 flags=run_cases([{'case_id':c} for c in 'ab'],execute_factory(token,mode),record,
   jobs=2,threshold_bytes=20*r.GIB,reserve_bytes=6*r.GIB,poll_seconds=.05,launch_interval_seconds=0)
 (OUT/(token+'.json')).write_text(json.dumps({'flags':flags,'results':results},indent=2));sys.exit(0)
all_results={}
for mode in ['timeout','cancel']:
 token='cgf-parallel-'+mode+'-'+uuid.uuid4().hex[:8]
 p=subprocess.Popen([sys.executable,__file__,token,mode])
 if mode=='cancel':
  for _ in range(200):
   raw=subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True)
   if sum(n.startswith(token) for n in raw.splitlines())==2:break
   if p.poll() is not None:raise RuntimeError('worker exited before two jobs started')
   time.sleep(.05)
  else:raise RuntimeError('two jobs did not start')
  time.sleep(.2);p.terminate()
 assert p.wait(timeout=30)==0
 data=json.loads((OUT/(token+'.json')).read_text());assert len(data['results'])==2
 if mode=='cancel':
  assert data['flags']['interrupted']
  assert all(v['error_kind']=='cancelled' for v in data['results'].values()),data
 else:
  assert data['results']['a']['error_kind']=='timeout',data
  assert data['results']['b']['status']=='ok',data
 for c in 'ab':assert subprocess.run(['docker','inspect',token+'-'+c],capture_output=True).returncode!=0
 all_results[mode]=data
 print('PASS real parallel',mode,flush=True)
(OUT/'parallel-lifecycle-results.json').write_text(json.dumps(all_results,indent=2))

import json,os,signal,subprocess,sys,time,uuid
from pathlib import Path
sys.path.insert(0,'/home/inuos/CloudGPUFit-cpu-recovery-20261108/tools')
import run_prediction_recovery as r
OUT=Path(__file__).resolve().parent
IMAGE='cloudgpufit-cpu-validation:20261108'
def interrupt(*_): raise KeyboardInterrupt
signal.signal(signal.SIGTERM,interrupt)
def command(code,cap):
 c=r.job_command(IMAGE,'/home/inuos/xmem-work','/home/inuos/.cache/huggingface',cap,1)
 return c[:c.index(IMAGE)+1]+['-u','-c',code]
def one(kind,name):
 cap=64*1024**2
 codes={'normal_exit':'import sys;sys.stdin.read();sys.exit(7)',
 'exit137':'import sys;sys.stdin.read();sys.exit(137)',
 'oom':'import sys;sys.stdin.read();x=bytearray(256*1024**2)',
 'timeout':'import sys,time;sys.stdin.read();time.sleep(120)',
 'cancelled':'import sys,time;sys.stdin.read();time.sleep(120)'}
 result=r.run_job(name,command(codes[kind],cap),{},OUT/(name+'.log'),1 if kind=='timeout' else 60)
 (OUT/(name+'.json')).write_text(json.dumps(result,indent=2))
 return result
if len(sys.argv)>1:
 one('cancelled',sys.argv[1]);sys.exit(0)
results={}
for kind in ('normal_exit','exit137','oom','timeout','cancelled'):
 name='cgf-validation-'+kind.replace('_','-')+'-'+uuid.uuid4().hex[:8]
 if kind=='cancelled':
  p=subprocess.Popen([sys.executable,__file__,name])
  for _ in range(100):
   state=subprocess.run(['docker','inspect','--format','{{.State.Running}}',name],capture_output=True,text=True)
   if state.returncode==0 and state.stdout.strip()=='true':break
   if p.poll() is not None: raise RuntimeError('Cancellation worker failed before running')
   time.sleep(.1)
  else: raise RuntimeError('Cancellation worker did not start')
  p.terminate();assert p.wait(timeout=30)==0
  result=json.loads((OUT/(name+'.json')).read_text())
 else: result=one(kind,name)
 expected={'normal_exit':'process_exit_without_json','exit137':'process_exit_without_json','oom':'cgroup_oom','timeout':'timeout','cancelled':'cancelled'}[kind]
 assert result['error_kind']==expected,(kind,result)
 assert result['oom_killed']==(kind=='oom'),(kind,result)
 assert result['container']['memory_limit_bytes']==64*1024**2
 assert result['container']['memory_swap_bytes']==64*1024**2
 assert all(not m['RW'] for m in result['container']['mounts'])
 gone=subprocess.run(['docker','inspect',name],capture_output=True).returncode!=0
 assert gone,name
 results[kind]={'result':result,'container_removed':gone}
 print('PASS',kind,flush=True)
(OUT/'lifecycle-results.json').write_text(json.dumps(results,indent=2))
assert subprocess.check_output(['docker','inspect','--format','{{.State.Running}}','xmem-cpu'],text=True).strip()=='true'

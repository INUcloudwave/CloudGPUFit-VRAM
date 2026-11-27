import subprocess
from pathlib import Path
root='/home/inuos/CloudGPUFit-cpu-singlepass-20261108'
out=Path(__file__).resolve().parent
code='''import os,shutil,subprocess,sys
shutil.copytree('/recovery','/tmp/validation',ignore=shutil.ignore_patterns('.git','__pycache__'))
os.chdir('/tmp/validation')
os.environ['PYTHONPATH']='/tmp/validation:/work/xMem'
os.environ['XMEM_REFERENCE_ROOT']='/work/xMem'
subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],check=True)
subprocess.run([sys.executable,'tools/smoke_native_inference.py'],check=True)
'''
args=['docker','run','--rm','--name','cgf-singlepass-tests-20261108','--network','none','--memory',str(8*1024**3),'--memory-swap',str(8*1024**3),'--cpus','2','--mount',f'type=bind,src={root},dst=/recovery,readonly','--mount','type=bind,src=/home/inuos/xmem-work,dst=/work,readonly','--mount','type=bind,src=/home/inuos/.cache/huggingface,dst=/root/.cache/huggingface,readonly','-e','PYTHONDONTWRITEBYTECODE=1','-e','HF_HUB_OFFLINE=1','-e','OPENBLAS_NUM_THREADS=1','--entrypoint','python','cloudgpufit-cpu-validation:20261108','-u','-c',code]
with (out/'vm-tests.log').open('x') as log:
 try: result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=240)
 finally: subprocess.run(['docker','rm','-f','cgf-singlepass-tests-20261108'],capture_output=True)
print('VM tests exit',result.returncode)
raise SystemExit(result.returncode)

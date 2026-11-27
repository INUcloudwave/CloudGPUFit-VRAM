#!/usr/bin/env python3
"""Reuse an existing immutable image. GPU exposure only in explicit GPU phases."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess

BASELINE='sha256:457da255e6e2de7ee05ff7814674c90abcf6d1ed04e73df3cc4eebcdb31fb032'

def main(a):
    docker=['sudo','docker'] if a.sudo else ['docker']
    info=json.loads(subprocess.check_output(docker+['image','inspect',a.image],text=True))[0]
    image_id=info['Id']
    if image_id!=a.expected_image_id:raise SystemExit('Image ID mismatch; inspect it and define a separate environment before executing.')
    package=Path(__file__).resolve().parent
    cmd=docker+['run','--rm','--pull=never','--network','bridge' if a.phase=='cache-download' else 'none',
        '-v',str(package)+':/lab:ro','-w','/lab','-e','PYTHONDONTWRITEBYTECODE=1',
        '-e','OUTSIDE_IMAGE_ID='+image_id,'-e','OUTSIDE_INSTANCE_LABEL='+a.instance_label,
        '-e','OUTSIDE_IMAGE_REPODIGESTS='+json.dumps(info.get('RepoDigests',[]))]
    if a.cache:
        cache=Path(a.cache).expanduser().resolve()
        if not cache.is_dir():raise SystemExit('Existing HF cache directory required')
        cmd+=['-v',str(cache)+':/hf:'+('rw' if a.phase=='cache-download' else 'ro'),'-e','HF_HOME=/hf']
    for item in a.env:
        key,sep,val=item.partition('=')
        if not sep or key not in ('CUDA_MODULE_LOADING','CUDA_MODULE_DATA_LOADING','CUBLAS_WORKSPACE_CONFIG',
             'PYTORCH_CUDA_ALLOC_CONF','PYTORCH_ALLOC_CONF'):raise SystemExit('Unrecognised controlled environment override')
        cmd+=['-e',item]
    if a.phase=='check':
        script="import torch, transformers, pynvml, json, subprocess; print(json.dumps({'torch':torch.__version__,'cuda_build':torch.version.cuda,'transformers':transformers.__version__,'cuda_initialized':torch.cuda.is_initialized()})); raise SystemExit(subprocess.call(['python','-m','pip','check']))"
        tail=['python','-c',script]
    elif a.phase in ('cache-check','cache-download'):
        if not a.cache:raise SystemExit('--cache required')
        tail=['python','prepare_cache.py']+(['--check-only'] if a.phase=='cache-check' else [])
    elif a.phase=='plan':tail=['python','pilot.py','--tier',a.tier,'--repeats',str(a.repeats)]
    else:
        if not a.gpu_uuid or not a.output:raise SystemExit('--gpu-uuid and --output required')
        if a.tier=='models' and not a.cache:raise SystemExit('--cache required for model tier')
        output=Path(a.output).expanduser().resolve();output.mkdir(parents=True,exist_ok=False)
        (output/'container-inspect.json').write_text(json.dumps(dict(Id=image_id,RepoDigests=info.get('RepoDigests'),
            Created=info.get('Created'),Architecture=info.get('Architecture'),Os=info.get('Os'),
            ImageEnv=info.get('Config',{}).get('Env')),indent=2))
        cmd+=['--gpus','device='+a.gpu_uuid,'--pid=host','-v',str(output)+':/results',
              '-e','CUDA_VISIBLE_DEVICES='+a.gpu_uuid,'-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1']
        if a.phase=='smoke':
            tail=['python','probe.py','--scenario','init','--gpu-uuid',a.gpu_uuid,
                  '--run-id','smoke','--output','/results/smoke']
        else:
            tail=['python','pilot.py','--execute','--tier',a.tier,'--gpu-uuid',a.gpu_uuid,
                  '--output','/results/session','--repeats',str(a.repeats)]
    return subprocess.call(cmd+[image_id]+tail)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['check','cache-check','cache-download','plan','smoke','pilot'])
    p.add_argument('--image',default='cloudgpufit-bench:cu124-v1');p.add_argument('--expected-image-id',default=BASELINE)
    p.add_argument('--sudo',action='store_true');p.add_argument('--cache');p.add_argument('--gpu-uuid');p.add_argument('--output')
    p.add_argument('--instance-label',default='local-rtx3090');p.add_argument('--tier',choices=['core','models'],default='core')
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--env',action='append',default=[])
    raise SystemExit(main(p.parse_args()))

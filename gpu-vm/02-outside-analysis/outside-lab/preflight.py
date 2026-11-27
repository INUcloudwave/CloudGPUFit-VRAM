#!/usr/bin/env python3
"""Read-only host inspection: no torch import, CUDA initialization, or changes."""
import json
import platform
import subprocess

def cmd(args):
    try:
        r=subprocess.run(args,capture_output=True,text=True,timeout=20)
        return dict(code=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip())
    except Exception as e:return dict(error=str(e))

if __name__=='__main__':
    print(json.dumps(dict(kernel=platform.release(),
        gpus=cmd(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.total,memory.free,mig.mode.current','--format=csv']),
        processes=cmd(['nvidia-smi']),
        images=cmd(['docker','image','ls','--no-trunc','--format','{{.Repository}}:{{.Tag}} {{.ID}}']),
        mps=cmd(['pgrep','-af','[n]vidia-cuda-mps']),
        docker=cmd(['docker','version','--format','{{.Server.Version}}'])),indent=2))

#!/usr/bin/env python3
"""Plan first, then sequential fresh-interpreter GPU runs. No GPU in dry-run."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

def cells(tier,case_ids):
    if tier=='core':
        result=[dict(name='init_control',scenario='init'),dict(name='init_api_probe',scenario='init',api_probe=True)]
        result += [dict(name=op,scenario=op) for op in ['gemm','conv','attention']]
        result += [dict(name='ops_'+str(i),scenario='ops',op_order=order) for i,order in enumerate(
            ['gemm,conv,attention','conv,attention,gemm','attention,gemm,conv'])]
        result += [dict(name='gemm_endpoint',scenario='gemm',sampling='endpoint'),
                   dict(name='gemm_NTN',scenario='gemm',query_order='NTN')]
        return result
    return [dict(name=c+'_'+h,scenario='model',case_id=c,history=h)
            for c in case_ids for h in ['direct','performance_first']]

def schedule(tier,repeats,case_ids):
    result=[];rng=random.Random(20261103)
    for r in range(repeats):
        block=cells(tier,case_ids);rng.shuffle(block)
        for c in block:result.append(dict(repeat=r,**c))
    return result

def main(a):
    cases=json.loads(Path(a.plan).read_text())['cases']
    ids=a.case_ids.split(',') if a.case_ids else [c['case_id'] for c in cases]
    jobs=schedule(a.tier,a.repeats,ids)
    package=Path(__file__).resolve().parent
    run=dict(schema_version=1,tier=a.tier,seed=20261103,jobs=jobs,args=vars(a),
             source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob('*.py')})
    run['input_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(a.plan),Path(a.revisions),package/'criteria.json']}
    if not a.execute:
        print(json.dumps(run,indent=2));return 0
    if not a.gpu_uuid:raise SystemExit('--gpu-uuid required for execute')
    if os.environ.get('CUDA_MPS_PIPE_DIRECTORY'):raise SystemExit('MPS is outside this protocol')
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False)
    run['status']='running';run['started_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    def save(): (root/'suite-manifest.json').write_text(json.dumps(run,indent=2))
    save();run['results']=[]
    try:
        for ix,job in enumerate(jobs):
            rid=f'{ix:03d}_{job["name"]}_r{job["repeat"]}'
            cmd=[sys.executable,str(package/'probe.py'),'--output',str(root/rid),'--gpu-uuid',a.gpu_uuid,
                 '--run-id',rid,'--repeat',str(job['repeat']),'--scenario',job['scenario'],
                 '--threads',str(a.threads),'--plan',str(Path(a.plan).resolve()),
                 '--revisions',str(Path(a.revisions).resolve())]
            for k,v in job.items():
                if k in ('name','scenario','repeat'):continue
                cmd.append('--'+k.replace('_','-'))
                if v is not True:cmd.append(str(v))
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=a.gpu_uuid
            env['OMP_NUM_THREADS']=str(a.threads);env['MKL_NUM_THREADS']=str(a.threads)
            # No mutation of CUDA loading/allocator variables: inherit and record.
            print(f'[{ix+1}/{len(jobs)}] {rid}',flush=True)
            with (root/(rid+'.log')).open('w') as f:
                try:
                    completed=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=a.timeout)
                    code=completed.returncode
                except subprocess.TimeoutExpired:code=124
            run['results'].append(dict(run_id=rid,exit_code=code,command=cmd));save()
            if code:raise RuntimeError(f'Run failed ({code}); inspect {rid}.log and manifest. No automatic retry.')
        run['status']='complete'
    except BaseException as e:
        run['status']='interrupted_or_error';run['error']=str(e);raise
    finally:save()
    return 0

if __name__=='__main__':
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser();p.add_argument('--tier',choices=['core','models'],default='core')
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--gpu-uuid')
    p.add_argument('--output',default='outside-results');p.add_argument('--execute',action='store_true')
    p.add_argument('--plan',default=str(here/'pilot-plan.json'));p.add_argument('--revisions',default=str(here/'revisions.json'))
    p.add_argument('--case-ids');p.add_argument('--threads',type=int,default=4);p.add_argument('--timeout',type=int,default=180)
    a=p.parse_args()
    if a.repeats<1:p.error('repeats must be positive')
    raise SystemExit(main(a))

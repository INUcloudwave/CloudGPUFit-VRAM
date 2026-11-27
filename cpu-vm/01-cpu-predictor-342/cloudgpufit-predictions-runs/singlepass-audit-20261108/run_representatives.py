import json,subprocess
from pathlib import Path
ROOT=Path('/home/inuos/CloudGPUFit-cpu-singlepass-20261108')
BASE=Path('/home/inuos/cloudgpufit-predictions')
AUDIT=Path(__file__).resolve().parent
CASES=[('core__decoder_llm__small__fp32__01','recovery-check-02'),
       ('core__vision_cls__small__fp32__01','recovery-check-03'),
       ('core__text_encdec__small__fp32__01','recovery-check-03'),
       ('core__vision_cls__large__fp32__01','recovery-check-03'),
       ('core__decoder_llm__small__fp32__12','recovery-check-04'),
       ('core__decoder_llm__small__fp16__01','recovery-check-02')]
for i,(cid,reference) in enumerate(CASES,1):
 out=BASE/f'singlepass-check-{i:02}'
 cmd=['python3','-u','tools/run_prediction_recovery.py','--output-root',str(out),'--snapshot-image','cloudgpufit-cpu-validation:20261108','--case-id',cid,'--compare-with',str(BASE/reference),'--threads','4','--memory-gib','32','--host-reserve-gib','6','--timeout-seconds','1800']
 print('RUN',cid,flush=True)
 with (AUDIT/f'check-{i:02}.console.log').open('x') as log:
  p=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 result=json.loads((out/(cid+'.prediction.json')).read_text())
 if result['status']!='ok':raise RuntimeError(result)
 summary=result['trace_summary']
 assert summary['warmup_invocations']==0 and summary['measured_invocations']==1
 assert result['prediction_version']=='4-native-inference-single-request'
 assert result['reference_comparison']['status'] in ('equal','mismatch'),result
 assert p.returncode in (0,2),p.returncode
 print('DONE',cid,result['reference_comparison'],flush=True)
print('All six completed; changed byte values are reported, not forced to match.',flush=True)

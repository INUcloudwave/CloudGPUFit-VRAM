import json,csv,os,collections,statistics as st
R='/home/inuos/cgf-reserved-diag-20261110/inputs/aws_g6_2xlarge/results/20261102T065012Z_v3_bbdabe31'
P='/home/inuos/CloudGPUFit_CPU_predictions_20261110/cpu_predictions_success.csv'
M=2**20
cpu={r['case_id']:r for r in csv.DictReader(open(P,encoding='utf-8-sig'))}
def ld(c,m):
  f=f'{R}/{c}__{m}__r0.json'
  return json.load(open(f)) if os.path.exists(f) else None
rows=[]
for c,r in cpu.items():
  mem=ld(c,'memory'); perf=ld(c,'performance')
  if not mem or mem.get('status')!='ok' or not perf or perf.get('status')!='ok': continue
  x=dict(c=c,w=r['workload_type'],dt=r['dtype'],
    cr=int(r['predicted_reserved_bytes'])/M, ca=int(r['predicted_allocated_bytes'])/M,
    gr=mem['inference_peak_reserved_bytes']/M, ga=mem['inference_peak_allocated_bytes']/M,
    load=perf['memory_baseline']['resident_reserved_bytes']/M,
    pr=perf['inference_peak_reserved_bytes']/M, pa=perf['inference_peak_allocated_bytes']/M)
  x['er']=(x['gr']-x['cr'])/x['gr']; x['ea']=(x['ga']-x['ca'])/x['ga']
  rows.append(x)
by=collections.defaultdict(list)
for x in rows: by[x['w']].append(x)
print('n',len(rows))
print('workload  n  | reserved err: |e|>20% med(e)  | allocated err: |e|>20% med(e) | perf_peak_res==mem_peak_res')
for w,xs in sorted(by.items()):
  print(w,len(xs),'|',sum(abs(x['er'])>.2 for x in xs),f"{st.median(x['er'] for x in xs):+.2f}",'|',
        sum(abs(x['ea'])>.2 for x in xs),f"{st.median(x['ea'] for x in xs):+.2f}",'|',sum(abs(x['pr']-x['gr'])<1 for x in xs))
print()
print('big reserved error cases: workload, alloc ok(<=5%)?')
for w,xs in sorted(by.items()):
  big=[x for x in xs if abs(x['er'])>.2]
  print(w,'big',len(big),'alloc within 5%:',sum(abs(x['ea'])<=.05 for x in big),'alloc 5-20%:',sum(.05<abs(x['ea'])<=.2 for x in big),'alloc>20%:',sum(abs(x['ea'])>.2 for x in big),
        'sign GPU>CPU:',sum(x['er']>0 for x in big))
json.dump(rows,open('/tmp/claude-1000/-home-inuos/ad7fc1ed-4caf-4967-9b97-8b87f53891cd/scratchpad/pop2.json','w'))

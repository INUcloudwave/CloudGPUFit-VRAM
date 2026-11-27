"""CPU main395 (old 1024 table) result.json vs RTX3090 L1024 per-condition CSV. Read-only on inputs."""
import csv, json, os, collections, statistics as st, sys
G='/home/inuos/cgf-fp16-inuos-handoff-20261115/results/rtx3090_main395_L1024/gpu_rtx3090_main395_L1024_per_condition.csv'
C='/home/inuos/cgf-fp16-680-20261111/runs/main395/cases'
NEW={c['condition_id'] for c in json.load(open('/home/inuos/cgf-fp16-inuos-handoff-20261115/config/conditions_main395_l256.json'))['conditions']}
M=2**20
gpu={r['condition_id']:r for r in csv.DictReader(open(G))}
rows=[]
for cid,g in gpu.items():
    p=f'{C}/{cid}/result.json'
    if not os.path.exists(p): continue
    r=json.load(open(p))
    if r.get('status')!='ok' or g['measure_status']!='ok': continue
    s=r['summary']
    gsha=json.loads(g['input_sha256']) if g['input_sha256'].startswith('{') else g['input_sha256']
    csha={k:v['sha256'] for k,v in r['inputs'].items()}
    x=dict(cid=cid, model=g['model'].split('/')[-1], kind=g['workload_type'], in_l256=cid in NEW,
           gattn=g['attn_implementation'], cattn=r['model_info']['attn_implementation'],
           sha_same=(gsha==csha) if isinstance(gsha,dict) else None,
           pr_c=s['first_measure_peak_reserved'], pr_g=int(g['m1_peak_reserved']),
           pa_c=s['first_measure_peak_allocated'], pa_g=int(g['m1_peak_allocated']),
           sr_c=s['first_measure_start_reserved'], sr_g=int(g['m1_start_reserved']),
           lr_c=s['after_load_reserved'], lr_g=int(g['after_load_reserved']),
           la_c=s['after_load_allocated'], la_g=int(g['after_load_allocated']),
           ws_pr_c=r['variant_with_cublas_workspace']['first_measure_peak_reserved'])
    rows.append(x)
def rel(c,g): return (c-g)/g if g else 0
print('compared',len(rows),'| input sha same',sum(1 for x in rows if x['sha_same']),'| attn same',sum(x['gattn']==x['cattn'] for x in rows))
print('attn pairs',collections.Counter((x['model'],x['gattn'],x['cattn']) for x in rows if x['gattn']!=x['cattn']))
for key,lab in [('pr','peak_reserved'),('pa','peak_allocated'),('lr','after_load_reserved'),('sr','m1_start_reserved')]:
    print(f'== {lab}: model  n  |rel|<=1%  <=5%  <=20%  >20%  median(rel)  min  max')
    by=collections.defaultdict(list)
    for x in rows: by[x['model']].append(rel(x[key+'_c'],x[key+'_g']))
    for m,v in by.items():
        a=[abs(t) for t in v]
        print(f'  {m:28} {len(v):3} {sum(t<=.01 for t in a):4} {sum(t<=.05 for t in a):4} {sum(t<=.2 for t in a):4} {sum(t>.2 for t in a):4}  {st.median(v):+.3f} {min(v):+.3f} {max(v):+.3f}')
json.dump(rows,open(os.path.join(os.path.dirname(__file__),'compare_l1024_rows.json'),'w'),indent=0)

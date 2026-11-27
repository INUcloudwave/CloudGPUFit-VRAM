import json,os,collections
R='/home/inuos/cgf-reserved-diag-20261110/inputs/aws_g6_2xlarge/results/20261102T065012Z_v3_bbdabe31'
M=2**20
rows=json.load(open('/tmp/claude-1000/-home-inuos/ad7fc1ed-4caf-4967-9b97-8b87f53891cd/scratchpad/pop2.json'))
def ld(c,m):
  f=f'{R}/{c}__{m}__r0.json'
  return json.load(open(f)) if os.path.exists(f) else None
by=collections.defaultdict(list)
for x in rows:
  fl=ld(x['c'],'first_logits'); x['fl']=fl['inference_peak_reserved_bytes']/M if fl and fl.get('status')=='ok' else 0
  by[x['w']].append(x)
def bucket(e): return 'within5%' if abs(e)<=.05 else ('5-20%' if abs(e)<=.2 else '>20%')
for w,xs in sorted(by.items()):
  c0=collections.Counter(bucket(x['er']) for x in xs)
  c1=collections.Counter(bucket((x['gr']-max(x['cr'],x['load']))/x['gr']) for x in xs)
  c2=collections.Counter(bucket((x['gr']-max(x['cr'],x['load'],x['fl']))/x['gr']) for x in xs)
  print(f"{w:12} n={len(xs)}  CPU only {dict(c0)}\n{'':14} max(CPU,load) {dict(c1)}\n{'':14} max(CPU,load,first_logits) {dict(c2)}")
  if w=='decoder_llm':
    rest=[x for x in xs if abs((x['gr']-max(x['cr'],x['load'],x['fl']))/x['gr'])>.05]
    for x in rest: print(f"   resid {x['c'][20:]:24} cpu {x['cr']:6.0f} load {x['load']:6.0f} fl {x['fl']:6.0f} L4 {x['gr']:6.0f} perf_peak {x['pr']:6.0f}")

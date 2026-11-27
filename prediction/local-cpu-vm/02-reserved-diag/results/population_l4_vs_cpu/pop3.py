import json,os,collections
R='/home/inuos/cgf-reserved-diag-20261110/inputs/aws_g6_2xlarge/results/20261102T065012Z_v3_bbdabe31'
M=2**20
rows=json.load(open('/tmp/claude-1000/-home-inuos/ad7fc1ed-4caf-4967-9b97-8b87f53891cd/scratchpad/pop2.json'))
def ld(c,m):
  f=f'{R}/{c}__{m}__r0.json'
  return json.load(open(f)) if os.path.exists(f) else None
# what does first_logits do / where run
d=ld(rows[0]['c'],'first_logits'); print('first_logits keys:',[k for k in d if 'scope' in k or 'struct' in k or 'timed' in k], d.get('scope'), d.get('timed_region'))
by=collections.defaultdict(list)
for x in rows:
  fl=ld(x['c'],'first_logits')
  x['fl']=fl['inference_peak_reserved_bytes']/M if fl and fl.get('status')=='ok' else None
  by[x['w']].append(x)
for w,xs in sorted(by.items()):
  big=[x for x in xs if abs(x['er'])>.2 and x['fl']]
  close=sum(abs(x['fl']-x['gr'])<=0.03*x['gr'] for x in big)
  print(w,'big',len(big),'first_logits peak within 3% of memory peak:',close,
        '| fl>=mem:',sum(x['fl']>=x['gr']-1 for x in big))
print()
for w in ('decoder_llm','text_encdec','vision_cls'):
  big=sorted([x for x in by[w] if abs(x['er'])>.2],key=lambda x:-abs(x['er']))[:6]
  print('--',w)
  for x in big: print(f"  {x['c'][len(w)+8:]:28} cpu {x['cr']:8.0f} L4 {x['gr']:8.0f} ({x['er']:+.0%}) load {x['load']:7.0f} fl {x['fl'] or 0:8.0f} | alloc cpu {x['ca']:7.0f} L4 {x['ga']:7.0f} ({x['ea']:+.0%})")
print('-- decoder_llm alloc err>20%:')
for x in by['decoder_llm']:
  if abs(x['ea'])>.2: print(f"  {x['c'][20:]:28} alloc cpu {x['ca']:7.0f} L4 {x['ga']:7.0f} ({x['ea']:+.0%}) res err {x['er']:+.0%}")

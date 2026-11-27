import json,glob,sys,statistics as st,collections as C
root=sys.argv[1];per=C.defaultdict(list);gap=[];cnt=C.Counter();vers={}
for f in sorted(glob.glob(root+'/*/session/*/samples.jsonl')):
    rid=f.split('/')[-2];_,img,model=rid.split('_')[:3];stages=C.defaultdict(list)
    vers[img]=json.load(open(f.replace('samples.jsonl','manifest.json'))).get('torch')
    for l in open(f):
        r=json.loads(l);cnt['all']+=1
        if r.get('kind')!='stable' or r.get('outside_bytes') is None:continue
        if r['errors'] or r['reserved_bracket_changed'] or r['other_gpu_pids'] or r['query_end_ns']-r['query_begin_ns']>5e6:cnt['filtered']+=1;continue
        cnt['ok']+=1;stages[r['stage']].append(r['outside_bytes']/2**20)
        d=(r.get('device_memory_info') or {}).get('value')
        if d and r['nvml_process_bytes']:gap.append((d['used_bytes']-r['nvml_process_bytes'])/2**20)
    for s,v in stages.items():per[(img,model,s)].append(st.median(v))
print(dict(cnt),vers)
print('device used - process NVML (MiB): min %.1f max %.1f'%(min(gap),max(gap)))
for k in sorted(per):
    v=per[k];print(*k,'blocks',[round(x) for x in v],'range',round(max(v)-min(v),1))
for m in ('gpt2','qwen7b'):
    for s in ('after_cuda_init','after_model_load','after_first_forward','final_idle'):
        a=per.get(('s0',m,s));b=per.get(('s4',m,s))
        if a and b:print('S4-S0',m,s,round(st.median(b)-st.median(a),1))

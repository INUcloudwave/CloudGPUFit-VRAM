"""GPU trace: after cache release (segment_free), where does the driver put new segments?"""
import pickle,gzip,json,sys
from collections import Counter
M=1<<20
for case in sys.argv[1:]:
    d='/home/inuos/cgf-fp16-inuos-handoff-20261116/traces/'+case
    T=json.load(open(d+'/trace.json'))
    ev=pickle.load(gzip.open(d+'/'+T['history']['snapshot_file'],'rb'))['device_traces'][0]
    live={}      # addr -> size (current segments)
    freed=[]     # (addr,size) ranges freed and not reoccupied
    lo=None; cnt=Counter(); released=False; ex=[]
    for e in ev:
        a=e['action']
        if a=='segment_alloc':
            s,z=e['addr'],e['size']
            if not released:
                kind='pre-release '+('below_all' if lo is None or s+z<=lo else 'not_below')
            else:
                ov=[(fa,fz) for fa,fz in freed if s<fa+fz and fa<s+z]
                if ov:
                    exact=any(fa==s for fa,fz in ov)
                    kind='reuse_freed_range'+(' (same start)' if exact else ' (overlap)')
                    for r in ov: freed.remove(r)
                elif s+z<=lo: kind='below_all'
                else: kind='other_gap'
            cnt[kind]+=1
            if released and len(ex)<12: ex.append((kind,hex(s),round(z/M,1)))
            live[s]=z; lo=s if lo is None else min(lo,s)
        elif a=='segment_free':
            released=True
            z=live.pop(e['addr']); freed.append((e['addr'],z))
    print(case, dict(cnt)); print('  first after release:',ex)

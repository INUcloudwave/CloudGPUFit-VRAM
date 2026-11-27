import numpy as np, collections, sys
for cid in sys.argv[1:]:
    za=np.load(f'/exp/runs/thread_check/t20/cases/{cid}/invocation_fp16_requests.npz')
    for t in ('t7','t1'):
        zb=np.load(f'/exp/runs/thread_check/{t}/cases/{cid}/invocation_fp16_requests.npz')
        sa=za['size'][za['action']==1]; sb=zb['size'][zb['action']==1]
        if len(sa)==len(sb):
            d=np.nonzero(sa!=sb)[0]
            pairs=collections.Counter(zip(sa[d].tolist(),sb[d].tolist()))
            print(cid,t,'n_diff',len(d),'of',len(sa),'top',pairs.most_common(4),'max_size_involved',int(max(sa[d].max(),sb[d].max())) if len(d) else 0)
        else:
            ca,cb=collections.Counter(sa.tolist()),collections.Counter(sb.tolist())
            print(cid,t,'len',len(sa),len(sb),'only20',list((ca-cb).items())[:5],'onlyT',list((cb-ca).items())[:5])

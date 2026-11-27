class VA:
    def __init__(s,mode): s.mode=mode; s.lo=1<<46; s.holes=[]; s.size={}
    def alloc(s,z):
        if s.mode!='desc':
            best=None
            for i,(a,n) in enumerate(s.holes):
                if n>=z and (best is None or a>s.holes[best][0]): best=i
            if best is not None:
                a,n=s.holes.pop(best)
                if s.mode=='reuse_bottom': p=a; rest=(a+z,n-z)
                else: p=a+n-z; rest=(a,n-z)
                if rest[1]>0: s.holes.append(rest)
                s.size[p]=z; return p
        s.lo-=z; s.size[s.lo]=z; return s.lo
    def free(s,p):
        z=s.size.pop(p)
        if s.mode=='desc': return
        a,n=p,z
        # coalesce
        changed=True
        while changed:
            changed=False
            for h in list(s.holes):
                if h[0]+h[1]==a: s.holes.remove(h); a,n=h[0],n+h[1]; changed=True
                elif a+n==h[0]: s.holes.remove(h); n+=h[1]; changed=True
        if a+n==s.lo:  # hole touching the low frontier: give back to frontier
            s.lo=a+n; s.lo=a if False else s.lo
        s.holes.append((a,n))

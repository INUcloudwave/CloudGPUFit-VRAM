import sys, json
sys.argv = [sys.argv[0], "/dev/null"] + sys.argv[1:]
exec(open("/out/verify_guide.py").read().split("\ndef main():")[0])
cid, gdir, cdir = sys.argv[2].split("=")
gs, gm, gpu = gpu_stream(Path(gdir)); cs, cm, _ = cpu_stream(Path(cdir))
gw, cw = window(gs, gm, "warmup_0"), window(cs, cm, "warmup_0")
def big(w, lo):
    out=[]; n=0
    for a,s,k in w:
        if a:
            n+=1
            if s>=lo: out.append((n,s))
    return out
G, C = big(gw, 1<<20), big(cw, 1<<20)
print("first 70 >=1MiB (alloc#, size) GPU | CPU")
for i in range(70):
    g = G[i] if i < len(G) else None; c = C[i] if i < len(C) else None
    print(i, g, c)
# where are GPU 2MiB vs CPU 1MiB requests (exact sizes)
g2=[(n,s) for n,s in G if 2097152-512<s<=2097152][:5]; c1=[(n,s) for n,s in C if 1048576-512<s<=1048576][:5]
print("gpu ~2MiB", g2, "cpu ~1MiB", c1)

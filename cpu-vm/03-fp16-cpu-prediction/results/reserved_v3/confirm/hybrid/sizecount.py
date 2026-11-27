import sys, json
from collections import Counter
import numpy as np
sys.argv = [sys.argv[0], "/dev/null"]
exec(open("/out/verify_guide.py").read().split("\ndef main():")[0])
M = 1 << 20
for cid in ("resnet-18__b128__r512", "resnet-50__b128__r512", "mobilenet-v2-1.0-224__b128__r512"):
    gs, gm, gpu = gpu_stream(Path("/f/rtx3090__trace_models/" + cid))
    z = np.load(f"/runs/confirm_hybrid/{cid}_cpu.npz")
    a, s, k, lay = z["action"], z["size"], z["alloc_id"], z["layer"]
    alay = {int(k[i]): int(lay[i]) for i in range(len(a)) if a[i] == 1}
    cmod = Counter(int(s[i]) for i in range(len(a)) if a[i] == 1 and alay[int(k[i])] == 0)
    g = Counter(sz for x, sz, kk in window(gs, gm, "warmup_1") if x)
    sizes = sorted(set(cmod) | set(g), reverse=True)
    diff = [(round(sz / M, 3), g[sz], cmod[sz]) for sz in sizes if g[sz] != cmod[sz]]
    only_g = [d for d in diff if d[2] == 0]; only_c = [d for d in diff if d[1] == 0]; both = [d for d in diff if d[1] and d[2]]
    print(cid, "| sizes with different counts (MiB, gpu, cpu_model):")
    print("   GPU-only sizes:", only_g[:20])
    print("   CPU-model-only sizes:", only_c[:20])
    print("   both but different counts:", both[:20])

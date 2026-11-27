import sys, json
import numpy as np
sys.argv = [sys.argv[0], "/dev/null"]
exec(open("/out/verify_guide.py").read().split("\ndef main():")[0])
M = 1 << 20
for cid in ("resnet-50__b128__r512", "resnet-18__b128__r512"):
    gs, gm, gpu = gpu_stream(Path("/f/rtx3090__trace_models/" + cid))
    z = np.load(f"/runs/confirm_hybrid/{cid}_cpu2.npz")
    a, s, k, lay, fn = z["action"], z["size"], z["alloc_id"], z["layer"], z["func"]; names = list(z["func_names"])
    alay = {int(k[i]): int(lay[i]) for i in range(len(a)) if a[i] == 1}
    afn = {int(k[i]): names[fn[i]] for i in range(len(a)) if a[i] == 1}
    cm = [(int(a[i]), int(s[i]), int(k[i])) for i in range(len(a)) if alay.get(int(k[i]), 0) == 0]
    calloc = [(sz, kk) for x, sz, kk in cm if x]
    w = window(gs, gm, "warmup_1")
    kmap, ci = {}, 0
    for t, (x, sz, gk) in enumerate(w):
        if not x: continue
        transient = t + 1 < len(w) and not w[t + 1][0] and w[t + 1][2] == gk
        if not transient and ci < len(calloc) and calloc[ci][0] == sz:
            kmap[gk] = calloc[ci][1]; ci += 1
    gseq = [(x, sz, kmap[gk]) for x, sz, gk in w if gk in kmap]
    print(cid, "GPU(mapped) events", len(gseq), "CPU model events", len(cm), "identical:", gseq == cm)
    n = 0
    for i, (ge, ce) in enumerate(zip(gseq, cm)):
        if ge != ce:
            n += 1
            if n <= 6:
                print("  diff at", i, "GPU", ("A" if ge[0] else "f") + f"{ge[1]/M:.0f}", afn.get(ge[2]), ge[2], "| CPU", ("A" if ce[0] else "f") + f"{ce[1]/M:.0f}", afn.get(ce[2]), ce[2],
                      "| ctx CPU", [("A" if x else "f") + f"{sz/M:.0f}" + ("" if x else "~") + afn.get(kk, "") + f"#{kk}" for x, sz, kk in cm[max(0, i - 3):i + 3]])
    print("  differing positions", n)

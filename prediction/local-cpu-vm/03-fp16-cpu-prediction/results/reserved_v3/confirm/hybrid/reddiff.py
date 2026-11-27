import sys, json, difflib
import numpy as np
cid = sys.argv[1]; RUN = "warmup_1"
sys.argv = [sys.argv[0], "/dev/null"]
exec(open("/out/verify_guide.py").read().split("\ndef main():")[0])
gs, gm, gpu = gpu_stream(Path("/f/rtx3090__trace_models/" + cid))
z = np.load(f"/runs/confirm_hybrid/{cid}_cpu.npz")
a, s, k, lay, fn = z["action"], z["size"], z["alloc_id"], z["layer"], z["func"]; names = list(z["func_names"])
alay = {int(k[i]): int(lay[i]) for i in range(len(a)) if a[i] == 1}
afn = {int(k[i]): names[fn[i]] for i in range(len(a)) if a[i] == 1}
M = 1 << 20
cm = [(int(a[i]), int(s[i]), int(k[i])) for i in range(len(a)) if a[i] == 1 and alay.get(int(k[i])) == 0]
gw = [(x, sz, kk) for x, sz, kk in window(gs, gm, RUN) if x]
sm = difflib.SequenceMatcher(None, [sz for _, sz, _ in gw], [sz for _, sz, _ in cm], autojunk=False)
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == "equal":
        continue
    print(f"{tag:7s} GPU-only: {[round(sz / M, 3) for _, sz, _ in gw[i1:i2]]}  CPU-model-only: {[(round(sz / M, 3), afn.get(kk)) for _, sz, kk in cm[j1:j2]]}  (prev CPU model op: {afn.get(cm[j1 - 1][2]) if j1 else None})")

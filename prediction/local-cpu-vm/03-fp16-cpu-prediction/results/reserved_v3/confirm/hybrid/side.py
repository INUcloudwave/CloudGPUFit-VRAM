import sys, json
import numpy as np
sys.argv = [sys.argv[0], "/dev/null"]
exec(open("/out/verify_guide.py").read().split("\ndef main():")[0])
cid = "resnet-18__b128__r512"
gs, gm, gpu = gpu_stream(Path("/f/rtx3090__trace_models/" + cid))
z = np.load(f"/runs/confirm_hybrid/{cid}_cpu.npz")
a, s, k, lay, fn = z["action"], z["size"], z["alloc_id"], z["layer"], z["func"]; names = list(z["func_names"])
alay = {int(k[i]): int(lay[i]) for i in range(len(a)) if a[i] == 1}
w = window(gs, gm, "warmup_1")
M = 1 << 20
def f(x, sz): return ("A" if x else "f") + f"{sz / M:.3f}"
print("GPU warmup_1 first 70:", " ".join(f(x, sz) for x, sz, kk in w[:70]))
print()
print("CPU all first 70 (L=library, func):", " ".join(f(int(a[i]), int(s[i])) + ("L" if alay.get(int(k[i])) == 1 else "") + (f"[{names[fn[i]]}]" if a[i] == 1 else "") for i in range(70)))

"""Conv signature list for the cuDNN workspace probe (runs on INUOS inside the CPU image; reads layer labels).
  A) every conv call of the stored vision cases (exact N, C, H, W, kernel, stride, padding, dilation, groups) - these
     can be checked against existing GPU measurements;
  B) a generic grid for models not in the table.
usage: python gpu_probe/make_signatures.py gpu_probe/conv_signatures.json"""
import glob
import json
import os
import sys

sys.path.insert(0, "/exp/src")
from cgf680 import replay4  # noqa: E402

sigs, seen = [], set()


def add(n, cin, h, w, cout, k, s, p, d, g, src):
    key = (n, cin, h, w, cout, k, s, p, d, g)
    if key in seen:
        return
    seen.add(key)
    sigs.append({"N": n, "C": cin, "H": h, "W": w, "K": cout, "R": k, "stride": s, "pad": p, "dil": d, "groups": g,
                 "src": src})


for f in sorted(glob.glob("/exp/runs/layer_labels/*.npz")):
    cid = os.path.basename(f)[:-4]
    lab = replay4.load_labels(f)
    for c, cp in lab["conv"].items():
        n, cin, h, w = cp["input"]
        cout, _, kh, kw = cp["weight"]
        pad = cp["padding"] if isinstance(cp["padding"], list) else [0, 0]
        add(n, cin, h, w, cout, kh, cp["stride"][0], pad[0], cp["dilation"][0], cp["groups"], "model:" + cid.split("__")[0])
# generic grid (FP16, NCHW): common CNN layer families
CH = [(3, 32), (3, 64), (16, 16), (32, 32), (32, 64), (64, 64), (64, 128), (128, 128), (128, 256), (256, 256),
      (256, 512), (512, 512), (512, 1024), (1024, 1024), (64, 256), (256, 64), (128, 512), (512, 128), (256, 1024),
      (1024, 256), (512, 2048), (2048, 512)]
for n in (1, 8, 32):
    for hw in (7, 14, 28, 56, 112, 224):
        for cin, cout in CH:
            if cin == 3 and hw < 112:
                continue
            for k, s in ((1, 1), (1, 2), (3, 1), (3, 2)):
                if hw // s < 1:
                    continue
                add(n, cin, hw, hw, cout, k, s, k // 2, 1, 1, "grid")
            if cin == 3:
                add(n, cin, hw, hw, cout, 7, 2, 3, 1, 1, "grid")
        for c in (16, 32, 64, 96, 144, 192, 384, 576, 960):       # depthwise
            for k, s in ((3, 1), (3, 2), (5, 1), (5, 2)):
                add(n, c, hw, hw, c, k, s, k // 2, 1, c, "grid_dw")
json.dump({"note": __doc__, "signatures": sigs}, open(sys.argv[1], "w"))
from collections import Counter
print(len(sigs), Counter(s["src"].split(":")[0] for s in sigs))

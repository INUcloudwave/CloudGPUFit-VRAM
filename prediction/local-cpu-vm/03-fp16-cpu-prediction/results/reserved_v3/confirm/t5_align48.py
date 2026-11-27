"""t5-base b128/128/512 warmup_0: align GPU vs CPU large requests (>=1 MiB) and show where CPU has 48 MiB but GPU 24 MiB."""
import sys, json, difflib
sys.argv = [sys.argv[0], "/dev/null"]
exec(open("/out/verify_guide.py").read().split("\ndef main():")[0])
cid = "t5-base__b128__in128__new512"
gs, gm, _ = gpu_stream(Path("/h16/traces/" + cid)); cs, cm, _ = cpu_stream(Path(f"/runs/main395_l256/cases/{cid}"))
def big(w):
    out = []
    for i, (a, s, k) in enumerate(w):
        if a and s >= 1 << 20:
            out.append((i, s))
    return out
g = big(window(gs, gm, "warmup_0")); c = big(window(cs, cm, "warmup_0"))
sm = difflib.SequenceMatcher(None, [s for _, s in g], [s for _, s in c], autojunk=False)
M = 1 << 20
n = 0
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == "equal":
        continue
    n += 1
    if n <= 30:
        print(tag, "gpu", [round(s / M, 2) for _, s in g[i1:i2]][:8], "@", g[i1][0] if i1 < len(g) else None,
              "| cpu", [round(s / M, 2) for _, s in c[j1:j2]][:8], "@", c[j1][0] if j1 < len(c) else None,
              "| big-idx gpu", i1, "cpu", j1, "ctx-before", [round(s / M, 2) for _, s in g[max(0, i1 - 6):i1]])
print("non-equal blocks", n, "gpu big", len(g), "cpu big", len(c), "warmup_0 len gpu", len(window(gs, gm, "warmup_0")), "cpu", len(window(cs, cm, "warmup_0")))

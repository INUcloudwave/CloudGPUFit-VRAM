import sys, json
sys.argv = [sys.argv[0], "/dev/null"] + sys.argv[1:]
exec(open("/out/verify_guide.py").read().split("\ndef main():")[0])
cid, gdir, cdir = sys.argv[2].split("=")
cs, cm, _ = cpu_stream(Path(cdir))
GPUV = 7104
def ws_pos(stream, marks, nth_alloc=33):
    n = 0
    for i in range(marks["warmup_0_begin"], len(stream)):
        if stream[i][0]:
            n += 1
            if n == nth_alloc: return i
def dbl(stream, marks):
    out = list(stream); n = 0
    for w in ("warmup_0", "warmup_1", "warmup_2", "measure_0"):
        b, e = marks[w + "_begin"], marks[w + "_end"]
        prev = None
        for i in range(b, e):
            a, s, k = out[i]
            if a and s == 1048576 and prev == 67108864:
                out[i] = (1, 2097152, k); n += 1
            if a: prev = s
    return out, n
p = ws_pos(cs, cm)
d, n = dbl(cs, cm)
for name, st, inj in [("ws@33", cs, [(p, 8519680)]), ("x2_1MiB", d, []), ("ws@33+x2_1MiB", d, [(p, 8519680)])]:
    r = replay(st, cm, 1 << 40, "descending", inj)
    print(name, "doubled", n if "x2" in name else 0, "m0_start MiB", r["m0_start_reserved"] / MiB, "GPU", GPUV, flush=True)

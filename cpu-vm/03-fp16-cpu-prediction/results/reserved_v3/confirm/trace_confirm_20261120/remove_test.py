"""Cause check for qwen -560 MiB: replay the GPU request stream with selected GPU-only request sizes removed.
If removing the per-layer 1,572,864 B requests (= batch x 12 heads x L x 4 B, attention logsumexp) brings the GPU
stream down to the CPU value, they are the cause. usage: remove_test.py <cid>=<trace dir> ..."""
import sys
exec(open("/out/verify_guide.py").read().split("\ndef main():")[0])


def filtered(stream, marks, drop_sizes):
    drop_ids = {k for a, s, k in stream if a and s in drop_sizes}
    keep_idx, new, newpos = [], [], {}
    for i, (a, s, k) in enumerate(stream):
        newpos[i] = len(new)
        if k in drop_ids:
            continue
        new.append((a, s, k))
    newpos[len(stream)] = len(new)
    m2 = {n: newpos[p] for n, p in marks.items()}
    return new, m2, len(drop_ids)


for spec in sys.argv[1:]:
    cid, gdir = spec.split("=")
    gs, gm, gpu = gpu_stream(Path(gdir))
    out = {"cid": cid, "gpu_MiB": gpu["m0_start_reserved"] / MiB}
    for name, sizes in (("as_is", set()), ("minus_1572864", {1572864}), ("minus_786432", {786432}),
                        ("minus_cublasLt_1MiB", {1048576}), ("minus_both", {1572864, 1048576})):
        st, mk, n = filtered(gs, gm, sizes)
        r = replay(st, mk, gpu["cap"], "descending")
        out[name] = {"removed_allocs": n, "R_MiB": None if r.get("oom") else r["m0_start_reserved"] / MiB}
    print(json.dumps(out), flush=True)

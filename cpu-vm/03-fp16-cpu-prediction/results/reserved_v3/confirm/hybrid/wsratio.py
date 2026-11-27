"""Provisional cuDNN workspace / conv-output ratio from GPU traces: in measure_0, an allocation (>1 MiB) that is freed
by the very next event and directly follows another allocation (the conv output) = transient conv workspace."""
import sys, json
sys.argv = [sys.argv[0], "/dev/null"]
exec(open("/out/verify_guide.py").read().split("\ndef main():")[0])
for name, d in (("resnet-18 b128 r512", "/f/rtx3090__trace_models/resnet-18__b128__r512"),
                ("resnet-50 b128 r512", "/f/rtx3090__trace_models/resnet-50__b128__r512"),
                ("mobilenet b128 r512", "/f/rtx3090__trace_models/mobilenet-v2-1.0-224__b128__r512"),
                ("vit-large b128 r224", "/f/rtx3090__trace_models/vit-large-patch16-224__b128__r224"),
                ("resnet-50 b8 r224", "/h12/reference/traces_attn_default/resnet-50__b8__r224")):
    try:
        gs, gm, gpu = gpu_stream(Path(d))
    except Exception as e:
        print(name, "ERR", e); continue
    w = window(gs, gm, "measure_0")
    rat = []
    for i in range(1, len(w) - 1):
        a, s, k = w[i]
        if a and s > (1 << 20) and not w[i + 1][0] and w[i + 1][2] == k and w[i - 1][0]:
            rat.append((round(s / MiB, 3), round(w[i - 1][1] / MiB, 3), round(s / max(w[i - 1][1], 1), 3)))
    r = [x[2] for x in rat]
    print(json.dumps({"case": name, "n_ws": len(rat), "max_ratio": max(r) if r else None,
                      "ratios_sorted": sorted(set(r), reverse=True)[:12], "examples_ws_MiB_out_MiB_ratio": rat[:8],
                      "gpu_reserved_MiB": gpu["m0_start_reserved"] / MiB}))

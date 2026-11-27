"""Conv layer catalogue from HF image-classification configs (2026-11-23). Each unique structural config is built with
transformers (no weights, meta device: shapes only, no compute, no memory), run once at batch 1 and its own image size,
and every F.conv2d call is recorded. A 'layer config' = (C_in, C_out, kernel, stride, padding, dilation, groups) - it
does not depend on batch or resolution (those only change N, H, W).
usage (image, no network): python tools/conv_catalog_meta.py results/conv_catalog/unique_configs.json results/conv_catalog/hf_configs.jsonl out.jsonl"""
import json
import signal
import sys
import time
import warnings

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoConfig, AutoModelForImageClassification

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
uniq = json.load(open(sys.argv[1]))
cfgs = {}
want = {v["id"] for v in uniq.values()}
for line in open(sys.argv[2]):
    r = json.loads(line)
    if r["id"] in want and "config" in r:
        cfgs[r["id"]] = r["config"]
REC = []
orig = F.conv2d


def hook(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
    t = lambda v: list(v) if isinstance(v, (tuple, list)) else ([v, v] if isinstance(v, int) else v)
    REC.append({"x": list(x.shape), "w": list(w.shape), "stride": t(stride), "padding": t(padding), "dilation": t(dilation),
                "groups": groups})
    return orig(x, w, b, stride, padding, dilation, groups)


F.conv2d = hook
torch.nn.modules.conv.F.conv2d = hook


class TO(Exception):
    pass


def alarm(*_):
    raise TO()


signal.signal(signal.SIGALRM, alarm)
out = open(sys.argv[3], "w")
t0 = time.time()
for h, v in sorted(uniq.items(), key=lambda kv: -kv[1]["downloads"]):
    c = dict(cfgs.get(v["id"], {}))
    row = {"hash": h, "id": v["id"], "model_type": v["model_type"], "downloads": v["downloads"]}
    REC.clear()
    try:
        signal.alarm(60)
        mt = c.pop("model_type", None)
        if not mt or mt == "timm_wrapper" or c.get("auto_map"):
            raise ValueError(f"skip: model_type={mt} auto_map={bool(c.get('auto_map'))}")
        cfg = AutoConfig.for_model(mt, **{k: x for k, x in c.items() if k not in ("architectures", "transformers_version")})
        from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING
        cls = MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING[type(cfg)]
        cls = cls[0] if isinstance(cls, tuple) else cls   # import the modeling module outside the meta context
        with torch.device("meta"):
            m = cls._from_config(cfg).eval()
            s = getattr(cfg, "image_size", None) or getattr(getattr(cfg, "vision_config", None), "image_size", None) or 224
            s = s[0] if isinstance(s, (list, tuple)) else s
            ch = getattr(cfg, "num_channels", None) or getattr(getattr(cfg, "vision_config", None), "num_channels", None) or 3
            with torch.no_grad():
                m(pixel_values=torch.empty(1, ch, s, s))
        row.update(status="ok", image_size=s, convs=list(REC))
    except TO:
        row.update(status="timeout")
    except Exception as e:  # noqa: BLE001
        row.update(status="error", error=f"{type(e).__name__}: {str(e)[:150]}")
    finally:
        signal.alarm(0)
    out.write(json.dumps(row) + "\n")
    out.flush()
print("CATALOG_DONE", len(uniq), round(time.time() - t0), "s", flush=True)

# downloads only the 4 models missing from the shared cache, pinned to explicit revisions
from huggingface_hub import snapshot_download
PINS = {
 "facebook/opt-125m": ("27dcfa74d334bc871f3234de431e71c6eeba5dd6", False),
 "TinyLlama/TinyLlama-1.1B-Chat-v1.0": ("fe8a4ea1ffedaf415f4da2f062534de366a451e6", True),
 "Qwen/Qwen2.5-1.5B-Instruct": ("989aa7980e4cf806f80c7fef2b1adb7bc71aa306", True),
 "microsoft/resnet-18": ("65a5785d9156231087c481e0c7dd33a5ff6f7e3e", True),
}
for m, (rev, safe) in PINS.items():
    p = snapshot_download(m, revision=rev, allow_patterns=["*.json", "*.safetensors" if safe else "pytorch_model.bin"])
    print("OK", m, rev, p, flush=True)

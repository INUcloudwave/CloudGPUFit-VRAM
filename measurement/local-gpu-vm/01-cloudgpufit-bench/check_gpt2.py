import json
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM

torch.manual_seed(20261006)
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False

model = AutoModelForCausalLM.from_pretrained(
    "openai-community/gpt2",
    torch_dtype=torch.float16,
    attn_implementation="eager",
).to("cuda").eval()

# CPU에서 입력 준비. 같은 입력을 반복 사용합니다.
input_ids = torch.randint(3, 50256, (1, 128))
attention_mask = torch.ones_like(input_ids)

def generate():
    with torch.inference_mode():
        return model.generate(
            input_ids=input_ids.to("cuda"),
            attention_mask=attention_mask.to("cuda"),
            do_sample=False,
            num_beams=1,
            use_cache=True,
            min_new_tokens=16,
            max_new_tokens=16,
            pad_token_id=model.config.eos_token_id,
        )

# 워밍업
output = generate()
torch.cuda.synchronize()
del output

# 실행 확인용 측정 1회
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter_ns()
output = generate()
torch.cuda.synchronize()
end = time.perf_counter_ns()

new_tokens = output.shape[1] - input_ids.shape[1]
assert new_tokens == 16

result = {
    "purpose": "smoke_check_only",
    "model": "openai-community/gpt2",
    "model_revision": getattr(model.config, "_commit_hash", None),
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "dtype": "fp16",
    "batch_size": 1,
    "seq_len": 128,
    "actual_new_tokens": new_tokens,
    "generation_ms": (end - start) / 1e6,
    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
}

Path("/results/gpt2-smoke.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(result, ensure_ascii=False, indent=2))
print("GPT-2 실행 및 결과 저장 정상")

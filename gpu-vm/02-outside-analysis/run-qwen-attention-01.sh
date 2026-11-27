#!/usr/bin/env bash
cd /home/inugpu/cloudgpufit-outside || exit 1
python3 outside-lab/run_qwen_attention.py --execute > qwen-attention-01.log 2>&1
attention_exit=$?
printf '%s\n' "$attention_exit" > qwen-attention-01.exitcode
exit "$attention_exit"

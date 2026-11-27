#!/usr/bin/env bash
SP=/tmp/claude-1000/-home-inuos/ad7fc1ed-4caf-4967-9b97-8b87f53891cd/scratchpad/rv
docker run --rm -i --network none --user "$(id -u):$(id -g)" --memory ${MEMLIM:-14g} --cpus 1 -e HOME=/tmp -e PYTHONPATH=/work/xMem \
  --mount type=bind,src=/home/inuos/xmem-work,dst=/work,readonly --mount type=bind,src=/home/inuos/cgf-fp16-inuos-handoff-20261116,dst=/h16,readonly \
  --mount type=bind,src=/home/inuos/cgf-fp16-inuos-handoff-20261112,dst=/h12,readonly \
  --mount type=bind,src=/home/inuos/cgf-fp16-680-20261111/runs,dst=/runs,readonly --mount type=bind,src=$SP,dst=/out \
  --entrypoint python cloudgpufit-cpu-validation:20261108 /out/wstest.py /out/wstest.jsonl "$@"

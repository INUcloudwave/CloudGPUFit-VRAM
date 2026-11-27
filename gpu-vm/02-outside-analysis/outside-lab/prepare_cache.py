#!/usr/bin/env python3
"""CPU-only pinned HF cache preparation/check. Run without --gpus."""
import argparse
import json
from pathlib import Path

def main(a):
    from huggingface_hub import snapshot_download
    pins=json.loads(Path(a.revisions).read_text());plan=json.loads(Path(a.plan).read_text())
    for name in dict.fromkeys(c['model'] for c in plan['cases']):
        pin=pins[name]
        p=Path(snapshot_download(name,revision=pin['revision'],local_files_only=a.check_only,
            allow_patterns=['*.json','*.safetensors' if pin['safetensors'] else 'pytorch_model*.bin']))
        indices=list(p.glob('*.safetensors.index.json' if pin['safetensors'] else 'pytorch_model*.bin.index.json'))
        weights=list(p.glob('*.safetensors' if pin['safetensors'] else 'pytorch_model*.bin'))
        if indices:
            required=set()
            for ip in indices:required.update(json.loads(ip.read_text())['weight_map'].values())
            missing=[w for w in required if not (p/w).is_file()]
            if missing:raise RuntimeError(f'{name}: missing shards: {missing}')
        if not (p/'config.json').is_file() or not weights:raise RuntimeError(f'{name}: missing config or weights')
        print(json.dumps(dict(model=name,revision=pin['revision'],snapshot=str(p),weights=len(weights),status='ok')))

if __name__=='__main__':
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser();p.add_argument('--plan',default=str(here/'pilot-plan.json'))
    p.add_argument('--revisions',default=str(here/'revisions.json'));p.add_argument('--check-only',action='store_true')
    main(p.parse_args())

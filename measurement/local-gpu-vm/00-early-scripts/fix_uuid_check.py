#!/usr/bin/env python3
"""Guarded GPU UUID format fix. CPU-only; keeps failed smoke and backup."""
import hashlib
import json
from pathlib import Path
import runpy
import uuid

OLD_SHA = 'da3e762409a7a9c8f24e9459d793cb8058a87fe7962d38c1f8eddbe602b3dc67'
NEW_SHA = '7df9a02ce9ca938b9702ee2c9fd7d0414d0a4cfc0cfe5ab87c97d7a069243d88'
REPLACEMENTS = [('import traceback\n', 'import traceback\nimport uuid as uuidlib\n'), ('def attempt(fn):', "def canonical_gpu_uuid(value):\n    text=decode(value) if isinstance(value,bytes) else str(value)\n    if text[:4].lower()=='gpu-':text=text[4:]\n    return str(uuidlib.UUID(text))\n\ndef attempt(fn):"), ("        if 'unavailable'!=manifest['torch_gpu']['uuid'] and str(prop.uuid)!=a.gpu_uuid:\n", "        manifest['uuid_check']=dict(torch_raw=manifest['torch_gpu']['uuid'],\n            nvml_raw=manifest['gpu']['uuid'],requested_raw=a.gpu_uuid)\n        identities=[canonical_gpu_uuid(x) for x in\n            (manifest['torch_gpu']['uuid'],manifest['gpu']['uuid'],a.gpu_uuid)]\n        manifest['uuid_check'].update(normalized=identities,match=len(set(identities))==1)\n        if len(set(identities))!=1:\n")]

def key(v):
    s=str(v)
    if s[:4].lower()=='gpu-':s=s[4:]
    return str(uuid.UUID(s))

def main():
    home=Path.home()
    record=home/'cloudgpufit-outside/results/smoke-01/smoke/manifest.json'
    m=json.loads(record.read_text())
    cuda=m['torch_gpu']['uuid'];nvml=m['gpu']['uuid'];requested=m['args']['gpu_uuid']
    print('PyTorch raw UUID:',cuda)
    print('NVML raw UUID:',nvml)
    if len({key(cuda),key(nvml),key(requested)})!=1:
        raise SystemExit('Actual UUID mismatch: no files changed; send these values for diagnosis.')
    p=home/'cloudgpufit-outside/outside-lab/probe.py'
    text=p.read_text();sha=hashlib.sha256(p.read_bytes()).hexdigest()
    if sha==NEW_SHA:
        print('Already patched.');return
    if sha!=OLD_SHA:raise SystemExit('Unexpected source version: no files changed.')
    for before,after in REPLACEMENTS:
        if text.count(before)!=1:raise SystemExit('Source range mismatch: no files changed.')
        text=text.replace(before,after,1)
    if hashlib.sha256(text.encode()).hexdigest()!=NEW_SHA:
        raise SystemExit('Patched hash mismatch: no files changed.')
    compile(text,str(p),'exec')
    backup=p.with_suffix('.py.before-uuid-fix')
    if backup.exists():raise SystemExit('Backup already exists: inspect before retry.')
    backup.write_bytes(p.read_bytes())
    p.write_text(text)
    fn=runpy.run_path(str(p),run_name='uuid_fix_validation')['canonical_gpu_uuid']
    assert fn(cuda)==fn(nvml)
    assert fn('GPU-00000000-0000-0000-0000-000000000001')!=fn('GPU-00000000-0000-0000-0000-000000000002')
    print('UUID fix applied; prefix normalization and mismatch checks passed.')
    print('probe.py SHA256:',NEW_SHA)

if __name__=='__main__':main()

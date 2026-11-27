#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
for name,h in json.loads((root/'package-manifest.json').read_text()).items():
 p=root/name
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise SystemExit('Package mismatch: '+name)
print('PACKAGE_OK')

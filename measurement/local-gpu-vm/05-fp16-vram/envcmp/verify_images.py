#!/usr/bin/env python3
"""Exit 0 iff every image in IMAGES_INFO json is loaded with the recorded image id (index or config digest) and layers."""
import json
import subprocess
import sys

ok = True
for i in json.load(open(sys.argv[1]))["images"]:
    try:
        iid = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", i["tag"]], text=True,
                                      stderr=subprocess.DEVNULL).strip()
        lay = json.loads(subprocess.check_output(["docker", "image", "inspect", "--format", "{{json .RootFS.Layers}}", i["tag"]], text=True))
    except subprocess.CalledProcessError:
        print("not loaded:", i["tag"])
        ok = False
        continue
    good = iid in (i["oci_index_digest"], i["config_digest"]) and lay == i["rootfs_diff_ids"]
    print(i["tag"], iid, "match" if good else "MISMATCH")
    ok &= good
sys.exit(0 if ok else 1)

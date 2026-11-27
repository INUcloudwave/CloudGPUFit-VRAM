#!/usr/bin/env python3
"""Host-side sequential, resumable runner (same script on RTX 3090 and AWS).

  python3 -I run_conditions.py --list pilot|all|<file.json>|id1,id2 --mode measure --gpu-label rtx3090 \
      [--conditions config/conditions_main395.json] [--attn default] \
      [--instance-type g6.xlarge] [--image cloudgpufit-bench:cu124-v1] [--hf-cache DIR] [--root DIR] [--dry-run]

--conditions: manifest file inside <root>/config (mounted read-only at /config); --list all = every id in it.
--attn default (since 2026-11-11 the default) = attn_implementation omitted (transformers default; v2 pilot).

Each condition: fresh `docker run --rm` (fresh worker process), one GPU, sequential.
Result: <root>/runs/<gpu_label>/<condition_id>/<mode>.json (+ .stdout.log/.stderr.log/.launch.json).
A condition whose <mode>.json already exists is skipped (resume). Failures are terminal and kept;
use --retry-status STATUS to re-run them (the previous file is moved to attempts/, never deleted).
"""
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sh(cmd):
    return subprocess.check_output(cmd, text=True).strip()


def imds_instance_type():
    try:
        tok = subprocess.run(["curl", "-s", "-m", "1", "-X", "PUT", "http://169.254.169.254/latest/api/token",
                              "-H", "X-aws-ec2-metadata-token-ttl-seconds: 60"], capture_output=True, text=True).stdout
        if not tok:
            return None
        r = subprocess.run(["curl", "-s", "-m", "1", "-H", f"X-aws-ec2-metadata-token: {tok}",
                            "http://169.254.169.254/latest/meta-data/instance-type"], capture_output=True, text=True)
        return r.stdout.strip() or None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--mode", choices=["measure", "validate", "trace"], default="measure")
    ap.add_argument("--gpu-label", required=True)
    ap.add_argument("--instance-type", default=None)
    ap.add_argument("--image", default="cloudgpufit-bench:cu124-v1")
    ap.add_argument("--hf-cache", default="/home/inugpu/cloudgpufit-bench/hf-cache")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--retry-status", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--conditions", default="conditions.json",
                    help="condition manifest file name inside <root>/config (e.g. conditions_main395.json)")
    ap.add_argument("--attn", choices=["eager", "default"], default="default",
                    help="eager: attn_implementation='eager'; default: transformers default (sdpa where supported)")
    a = ap.parse_args()
    root = Path(a.root)
    cfg = root / "config"
    cond_file = cfg / Path(a.conditions).name
    if Path(a.conditions).name != a.conditions and Path(a.conditions).resolve() != cond_file.resolve():
        sys.exit(f"--conditions must be a file inside {cfg}")
    cond_bytes = cond_file.read_bytes()
    conds = [c["condition_id"] for c in json.loads(cond_bytes)["conditions"]]
    if a.list == "all":
        ids = conds
    elif a.list == "pilot":
        ids = json.loads((cfg / "pilot.json").read_text())["condition_ids"]
    elif a.list.endswith(".json"):
        ids = json.loads(Path(a.list).read_text())["condition_ids"]
    else:
        ids = a.list.split(",")
    unknown = [i for i in ids if i not in set(conds)]
    if unknown:
        sys.exit(f"unknown condition ids: {unknown}")

    image_id = sh(["docker", "image", "inspect", "--format", "{{.Id}}", a.image])
    digests = sh(["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", a.image])
    gpus = sh(["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total", "--format=csv,noheader"]).splitlines()
    if len(gpus) != 1:
        sys.exit(f"expected exactly 1 GPU, found {len(gpus)}: {gpus}")
    uuid = gpus[0].split(",")[0].strip()
    itype = a.instance_type or imds_instance_type()
    base = root / "runs" / a.gpu_label
    if not a.dry_run:
        base.mkdir(parents=True, exist_ok=True)
    session = {"started": now(), "argv": sys.argv, "image": a.image, "image_id": image_id, "repo_digests": digests,
               "gpu": gpus[0], "instance_type": itype, "n_conditions": len(ids), "mode": a.mode, "attn": a.attn,
               "conditions_file": cond_file.name, "conditions_sha256": hashlib.sha256(cond_bytes).hexdigest(),
               "code_sha256": {n: hashlib.sha256((root / "src" / n).read_bytes()).hexdigest()
                               for n in ("vram_worker.py", "run_conditions.py")}}
    if a.dry_run:
        print(json.dumps(session), flush=True)
        for cid in ids:
            print(cid)
        print(f"DRY_RUN selected={len(ids)} unique={len(set(ids))}", flush=True)
        return
    with (base / "sessions.jsonl").open("a") as f:
        f.write(json.dumps(session) + "\n")
    print(json.dumps(session), flush=True)

    done = 0
    for k, cid in enumerate(ids):
        d = base / cid
        res = d / f"{a.mode}.json"
        if res.exists():
            prev = json.loads(res.read_text())
            if prev.get("status") not in a.retry_status:
                print(f"[{k + 1}/{len(ids)}] skip {cid} ({prev.get('status')})", flush=True)
                continue
            att = d / "attempts"
            att.mkdir(exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            for p in d.glob(f"{a.mode}.*"):
                p.rename(att / f"{stamp}__{p.name}")
        apps = sh(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
        if apps:
            sys.exit(f"other GPU process present before {cid}: {apps}")
        d.mkdir(parents=True, exist_ok=True)
        cmd = ["docker", "run", "--rm", "--pull=never", "--network", "none", "--pid=host", "--gpus", f"device={uuid}",
               "--user", f"{os.getuid()}:{os.getgid()}", "--name", f"cgfvram_{os.getpid()}_{k}",
               "-e", "HOME=/tmp", "-e", "HF_HOME=/hf", "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
               "-e", "CUDA_MODULE_LOADING=LAZY", "-e", f"CGF_IMAGE_ID={image_id}", "-e", f"CGF_GPU_LABEL={a.gpu_label}",
               "-e", f"CGF_INSTANCE_TYPE={itype or ''}",
               "-v", f"{root / 'src'}:/src:ro", "-v", f"{cfg}:/config:ro", "-v", f"{a.hf_cache}:/hf:ro",
               "-v", f"{d}:/out", "-w", "/tmp", "--entrypoint", "python", image_id, "-I", "/src/vram_worker.py",
               "--models", "/config/models.json", "--conditions", f"/config/{cond_file.name}",
               "--condition", cid, "--mode", a.mode, "--out", f"/out/{a.mode}.json", "--attn", a.attn]
        launch = {"condition_id": cid, "mode": a.mode, "cmd": " ".join(shlex.quote(c) for c in cmd), "start": now()}
        if a.dry_run:
            print(launch["cmd"])
            continue
        t0 = time.time()
        with (d / f"{a.mode}.stdout.log").open("w") as so, (d / f"{a.mode}.stderr.log").open("w") as se:
            try:
                rc = subprocess.run(cmd, stdout=so, stderr=se, timeout=a.timeout).returncode
            except subprocess.TimeoutExpired:
                subprocess.run(["docker", "stop", "-t", "5", f"cgfvram_{os.getpid()}_{k}"], capture_output=True)
                rc = "timeout"
        launch.update(end=now(), wall_s=round(time.time() - t0, 1), returncode=rc)
        (d / f"{a.mode}.launch.json").write_text(json.dumps(launch, indent=1))
        if res.exists():
            st = json.loads(res.read_text()).get("status")
        else:  # worker died without writing (e.g. host OOM kill = 137, timeout)
            st = "timeout" if rc == "timeout" else ("worker_killed_cpu_oom_suspected" if rc == 137 else f"worker_exit_{rc}")
            res.write_text(json.dumps({"condition_id": cid, "mode": a.mode, "status": st, "returncode": rc,
                                       "note": "written by runner; worker produced no result"}, indent=1))
        done += 1
        line = {"t": now(), "condition_id": cid, "mode": a.mode, "status": st, "rc": rc, "wall_s": launch["wall_s"]}
        with (base / "progress.jsonl").open("a") as f:
            f.write(json.dumps(line) + "\n")
        print(f"[{k + 1}/{len(ids)}] {cid} {a.mode} status={st} rc={rc} wall={launch['wall_s']}s", flush=True)
    print(f"RUNNER_DONE ran={done} {now()}", flush=True)


if __name__ == "__main__":
    main()

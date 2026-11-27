#!/bin/bash
# Usage: bash aws/aws_00_preflight.sh <expected_instance_type e.g. g6.xlarge>
# Confirms the assigned instance/GPU before anything runs. Prints facts; exits 1 on a hard mismatch.
set -u
EXP=${1:?expected instance type}
TOK=$(curl -s -m 2 -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
IT=$(curl -s -m 2 -H "X-aws-ec2-metadata-token: $TOK" http://169.254.169.254/latest/meta-data/instance-type)
IID=$(curl -s -m 2 -H "X-aws-ec2-metadata-token: $TOK" http://169.254.169.254/latest/meta-data/instance-id)
echo "instance_type=$IT instance_id=$IID expected=$EXP"
nvidia-smi --query-gpu=uuid,name,driver_version,memory.total --format=csv,noheader
echo "gpu_count=$(nvidia-smi -L | wc -l)  other_gpu_procs=[$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')]"
docker version --format 'docker {{.Server.Version}}'; docker info --format 'storage {{.Driver}} {{.DriverStatus}} runtimes {{json .Runtimes}}' 2>/dev/null | cut -c1-300
df -h / "$(dirname "$(readlink -f "$0")")/.." | sed 1d
nproc; free -g | sed -n 2p
rc=0
[ "$IT" = "$EXP" ] || { echo "MISMATCH instance type"; rc=1; }
[ "$(nvidia-smi -L | wc -l)" = 1 ] || { echo "runner expects exactly 1 GPU"; rc=1; }
echo PREFLIGHT_RC=$rc; exit $rc

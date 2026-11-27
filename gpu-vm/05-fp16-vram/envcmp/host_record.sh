#!/bin/bash
# Record host / driver / container facts into envcmp/records/<TAG>_<timestamp>/ (read-only; no changes to the host).
#   bash envcmp/host_record.sh <TAG e.g. A_before|C_before|B_before|after_reboot>
set -u
EXP=$(readlink -f "$(dirname "$0")/.."); TAG=${1:?tag}; D=$EXP/envcmp/records/${TAG}_$(date +%Y%m%dT%H%M%S); mkdir -p "$D"
r() { local n=$1; shift; { echo "\$ $*"; "$@"; } > "$D/$n.txt" 2>&1; }
TOK=$(curl -s -m 2 -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
for k in instance-type instance-id ami-id placement/availability-zone; do
  echo "$k=$(curl -s -m 2 -H "X-aws-ec2-metadata-token: $TOK" http://169.254.169.254/latest/meta-data/$k)"; done > "$D/imds.txt"
r uname uname -a; r os_release cat /etc/os-release; r cmdline cat /proc/cmdline
r nvidia_smi nvidia-smi; r nvidia_smi_q nvidia-smi -q
r nvidia_smi_csv nvidia-smi --query-gpu=name,uuid,driver_version,vbios_version,memory.total,persistence_mode,ecc.mode.current,compute_mode,pcie.link.gen.current --format=csv
r proc_driver cat /proc/driver/nvidia/version
r modinfo sh -c 'modinfo nvidia | grep -E "^(filename|version|license|vermagic|srcversion):"'
r lsmod sh -c 'lsmod | grep -i nvidia'
r dkms sh -c 'dkms status 2>/dev/null || echo "no dkms"'
r libcuda_owner sh -c 'L=$(ldconfig -p | awk "/libcuda.so.1 /{print \$NF; exit}"); R=$(readlink -f "$L"); echo "libcuda=$L -> $R"; (dpkg -S "$R" 2>/dev/null || rpm -qf "$R" 2>/dev/null || echo "NOT OWNED BY ANY PACKAGE (runfile install?)")'
r runfile_evidence sh -c 'ls -l /usr/bin/nvidia-uninstall /usr/bin/nvidia-installer /var/log/nvidia-installer.log 2>&1; nvidia-installer --version 2>&1 | head -3; grep -m5 -iE "nvidia-installer log|installer version|kernel module type|Installing.*kernel" /var/log/nvidia-installer.log 2>/dev/null'
r pkgs_deb sh -c 'dpkg-query -W -f "\${db:Status-Abbrev} \${Package} \${Version}\n" "*nvidia*" "*cuda-driver*" 2>/dev/null | grep "^ii" || echo "no deb packages"'
r pkgs_rpm sh -c 'rpm -qa 2>/dev/null | grep -iE "nvidia|cuda-driver" | sort || echo "no rpm"'
r apt_policy sh -c 'for p in $(dpkg-query -W -f "\${Package}\n" "nvidia-driver-*" "nvidia-open*" "cuda-drivers*" "nvidia-kernel-*" "libnvidia-compute-*" 2>/dev/null); do apt-cache policy $p | head -6; done'
r kernels sh -c 'dpkg -l "linux-image-*" 2>/dev/null | grep ^ii; rpm -qa kernel 2>/dev/null; ls /boot | grep -E "vmlinuz"'
r services sh -c 'systemctl list-units --type=service --no-pager | grep -iE "nvidia|dcgm|docker|containerd|unattended|apt-daily"'
r docker sh -c 'docker version --format "client {{.Client.Version}} server {{.Server.Version}}"; docker info --format "{{.Driver}} {{json .Runtimes}} {{.DefaultRuntime}}"; docker image ls --digests | grep cloudgpufit'
r container_toolkit sh -c 'nvidia-ctk --version 2>&1; nvidia-container-cli --version 2>&1 | head -3; cat /etc/nvidia-container-runtime/config.toml 2>/dev/null | grep -vE "^#|^$" | head -30'
r cpu_mem sh -c 'nproc; lscpu | grep -E "Model name|^CPU\(s\)|Thread|NUMA node\(s\)"; free -b'
r disk df -h / "$EXP"
for t in cu124-v1 pt251-cu124-v1; do
  docker image inspect --format "{{.Id}} {{json .RootFS.Layers}}" cloudgpufit-bench:$t > "$D/image_$t.txt" 2>&1
  docker run --rm --pull=never --network none --gpus all --entrypoint python cloudgpufit-bench:$t -c \
   "import torch,json,importlib.metadata as m;print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'cuda_ok':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'cap':list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,'arch':torch._C._cuda_getArchFlags(),'pkgs':sorted(d.metadata['Name']+'=='+d.version for d in m.distributions() if d.metadata['Name'].lower().startswith(('nvidia-','torch','triton')))}))" \
   > "$D/container_$t.json" 2> "$D/container_$t.err"
done
V=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
{ echo "driver=$V"
  grep -q "Open Kernel Module" /proc/driver/nvidia/version && echo "kernel_module=open" || echo "kernel_module=proprietary"
  echo "datacenter_runfile_url_http=$(curl -s -o /dev/null -I -w '%{http_code}' https://us.download.nvidia.com/tesla/$V/NVIDIA-Linux-x86_64-$V.run)"
  echo "unix_driver_runfile_url_http=$(curl -s -o /dev/null -I -w '%{http_code}' https://us.download.nvidia.com/XFree86/Linux-x86_64/$V/NVIDIA-Linux-x86_64-$V.run)"
  grep -q "NOT OWNED" "$D/libcuda_owner.txt" && echo "install_channel=runfile" || { grep -q "^ii" "$D/pkgs_deb.txt" && echo "install_channel=deb" || echo "install_channel=rpm_or_unknown"; }
  echo "kernel=$(uname -r)"; echo "persistence=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader)"
} | tee "$D/DRIVER_FACTS.txt"
echo "RECORD_DIR=$D"

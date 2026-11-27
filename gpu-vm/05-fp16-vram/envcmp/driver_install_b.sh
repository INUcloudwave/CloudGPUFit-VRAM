#!/bin/bash
# Environment B: replace ONLY the host driver with an R580 data-center driver. Dry run unless --apply.
#   bash envcmp/driver_install_b.sh <580.x.y> [--apply]
# Supported automatically: install_channel=runfile (A was installed with the NVIDIA .run installer, the DLAMI default).
# deb/rpm channel: prints the candidate packages and stops (install B from the SAME channel by hand; see GUIDE §5-B).
# Never upgrades the kernel: requires headers of the running kernel and refuses if the running kernel differs from A's.
set -euo pipefail
EXP=$(readlink -f "$(dirname "$0")/.."); V=${1:?R580 version e.g. 580.196.07}; APPLY=${2:-}
[[ $V == 580.* ]] || { echo "ABORT: B must be an R580 driver, got $V"; exit 1; }
A_FACTS=$(ls -d "$EXP"/envcmp/records/A_before_*/DRIVER_FACTS.txt 2>/dev/null | head -1)
[ -n "$A_FACTS" ] || { echo "ABORT: run A first (envcmp/records/A_before_*/DRIVER_FACTS.txt missing)"; exit 1; }
A_KERNEL=$(grep ^kernel= "$A_FACTS" | cut -d= -f2); KMOD=$(grep ^kernel_module= "$A_FACTS" | cut -d= -f2)
CHANNEL=$(grep ^install_channel= "$A_FACTS" | cut -d= -f2)
echo "A facts: $(tr '\n' ' ' < "$A_FACTS")"
[ "$(uname -r)" = "$A_KERNEL" ] || { echo "ABORT: running kernel $(uname -r) != A kernel $A_KERNEL"; exit 1; }
[ -d "/lib/modules/$(uname -r)/build" ] || { echo "ABORT: kernel headers for $(uname -r) missing (install exactly linux-headers-$(uname -r), no upgrade)"; exit 1; }
URL=https://us.download.nvidia.com/tesla/$V/NVIDIA-Linux-x86_64-$V.run
CODE=$(curl -s -o /dev/null -I -w '%{http_code}' "$URL"); echo "B runfile $URL -> HTTP $CODE"
[ "$CODE" = 200 ] || { echo "ABORT: $V is not published as a data-center runfile. Pick a version listed at https://docs.nvidia.com/datacenter/tesla/ (R580) and retry."; exit 1; }
if [ "$CHANNEL" != runfile ]; then
  echo "A was installed from packages ($CHANNEL). Not automated. Candidates from the same channel:"
  (apt-cache madison nvidia-driver-580-open nvidia-driver-580 nvidia-driver-580-server-open nvidia-driver-580-server cuda-drivers-580 nvidia-open-580 2>/dev/null || dnf list --showduplicates 'nvidia-driver*580*' 'kmod-nvidia*580*' 2>/dev/null) | head -40
  echo "See GUIDE §5-B (package branch). STOP."; exit 2
fi
FLAG=--kernel-module-type=$KMOD
echo "PLAN: stop GPU users -> sh NVIDIA-Linux-x86_64-$V.run --silent --dkms $FLAG (replaces the runfile driver of A) -> reboot"
[ "$APPLY" = --apply ] || { echo "dry run only. Re-run with --apply."; exit 0; }
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo "ABORT: GPU processes running"; exit 1; }
mkdir -p ~/drivers && cd ~/drivers
[ -f NVIDIA-Linux-x86_64-$V.run ] || curl -fSLO "$URL"
sha256sum NVIDIA-Linux-x86_64-$V.run | tee "$EXP/envcmp/records/B_runfile_sha256.txt"
sudo systemctl stop nvidia-persistenced 2>/dev/null || true
sudo nvidia-smi -pm 0 || true     # A had persistence Enabled; re-enable after reboot (sudo nvidia-smi -pm 1)
for s in nvidia-dcgm dcgm nvidia-fabricmanager; do sudo systemctl stop $s 2>/dev/null || true; done
# 2026-11-17 fix: --silent answers "abort" while the old module is loaded -> unload it first (attempt 1 failed this way)
# DLAMI loads efa_nv_peermem + gdrdrv, which hold 'nvidia' (seen 2026-11-17); nvidia_drm may not exist -> rmmod one by one
sudo systemctl stop gdrcopy 2>/dev/null || true
lsmod | grep -E "nvidia|gdrdrv|efa_nv_peermem" > "$EXP/envcmp/records/B_lsmod_before_unload.txt" || true
for m in efa_nv_peermem gdrdrv nvidia_uvm nvidia_drm nvidia_modeset nvidia_peermem nvidia; do
  lsmod | grep -q "^$m " && sudo rmmod "$m" || true; done
if lsmod | grep -qE "^nvidia"; then echo "ABORT: nvidia modules still loaded:"; lsmod | grep -E "^nvidia"; sudo lsof /dev/nvidia* 2>/dev/null | head; exit 1; fi
sudo sh NVIDIA-Linux-x86_64-$V.run --silent --dkms "$FLAG" 2>&1 | tee "$EXP/envcmp/records/B_install_console.txt"
sudo cp /var/log/nvidia-installer.log "$EXP/envcmp/records/B_nvidia-installer.log" && sudo chown "$(id -u):$(id -g)" "$EXP/envcmp/records/B_nvidia-installer.log"
echo "INSTALL_DONE. Now: sudo reboot   (then GUIDE §5-B after reboot)"

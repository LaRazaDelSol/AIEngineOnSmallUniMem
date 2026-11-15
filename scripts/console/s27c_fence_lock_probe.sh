#!/bin/bash
# S27c: read-only. Two last register-level facts:
#  (1) did the FIRMWARE declare the range reserved (e820), or did Linux choose it?
#  (2) what do the GPU's own aperture / VM registers read RIGHT NOW?
# No writes. debugfs register files are read-only views; no GPU work is submitted.

echo "=== e820: the map the FIRMWARE handed the kernel ==="
sudo -n dmesg 2>/dev/null | grep -iE "e820|BIOS-e820|user-defined physical RAM map|lastmap|E820" | sed -n "1,26p"

echo; echo "=== debugfs register windows that exist ==="
D=/sys/kernel/debug/dri/0
for f in amdgpu_regs amdgpu_regs_pcie amdgpu_gca_config amdgpu_vm_info amdgpu_firmware_info amdgpu_gem_info; do
  printf "  %-22s readable=%s size=%s\n" "$f" "$([ -r "$D/$f" ] && echo yes || echo no)" "$(sudo -n stat -c%s "$D/$f" 2>/dev/null)"
done

echo; echo "=== VM / GART page-table base as the driver programmed it ==="
sudo -n cat "$D/amdgpu_vm_info" 2>/dev/null | sed -n "1,10p"
echo "--- gca_config (register-block base addresses the driver maps) ---"
sudo -n cat "$D/amdgpu_gca_config" 2>/dev/null | sed -n "1,24p"

echo; echo "=== the range's neighbours in the live map, exact edges ==="
sudo -n grep -iE "^(00)?[0-9a-f]+-[0-9a-f]+ : (System RAM|Reserved)" /proc/iomem | sed -n "1,14p"

echo; echo "=== engine still serving, untouched (read-only ps) ==="
ps -o pid,etime= -p 7776,7773,1366 2>/dev/null | tr -s ' '

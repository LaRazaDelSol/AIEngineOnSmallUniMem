#!/bin/bash
# S26: read-only probe: why can't the GPU touch the reserved low 1GB?
# Reads the GPU's address-aperture registers through BAR0 (read-only mmap),
# the TOP-OF-MEMORY MSRs, and the kernel driver's buffer-backing path.
# Nothing here writes a register or submits GPU work.
cd /tmp   # $HOME has a local mmap.py that shadows the stdlib module
echo "=== GPU BAR0: address-aperture registers ==="
sudo -n python3 - <<'PY'
import mmap, os
fd = os.open("/sys/bus/pci/devices/0000:20:00.0/resource0", os.O_RDONLY)
mm = mmap.mmap(fd, 0, mmap.MAP_SHARED, mmap.PROT_READ)
rd = lambda o: int.from_bytes(mm[o:o+4], "little")
for name, off in [("GARTPRIMBASE",0x80),("GARTSECBASE",0x84),("GARTPRIMLIMIT",0x88),
                  ("GARTSECLIMIT",0x8c),("GARTCONFIG",0x90),("FB_BASE",0x0),("FB_LIMIT",0x4)]:
    print("  %-14s +0x%04x = 0x%08x" % (name, off, rd(off)))
print("--- OSS/GART window 0x0000..0x00a0 ---")
for o in range(0x0000, 0x00a4, 4):
    print("  +0x%04x = 0x%08x" % (o, rd(o)))
PY
echo "=== TOP_OF_MEMORY MSRs ==="
sudo -n modprobe msr 2>/dev/null
for m in 0xc001001a 0xc00101ea; do
  off=$((m)); printf "  MSR %s = 0x" "$m"
  sudo -n dd if=/dev/cpu/0/msr bs=8 count=1 iflag=skip_bytes skip="$off" 2>/dev/null | od -An -tx8 | tr -d ' \n'; echo
done
echo "=== driver: what backs a DEVICE_LOCAL buffer (amdgpu_ttm) ==="
K=~/kbuild/linux/drivers/gpu/drm/amd/amdgpu
grep -rn "use_cpu\|TTM_PL_TT\|amdgpu_ttm_io_mem_ptr\|gmc.flags.*VCACHE\|AMDGPU_GEM_DOMAIN_VRAM" "$K/amdgpu_ttm.c" "$K/amdgpu_ttm.h" 2>/dev/null | grep -v "^\s*\*" | head -12
echo "=== driver: GART table target-address derivation ==="
grep -rn "gart" "$K/amdgpu_gmc.c" "$K/amdgpu_vm.c" 2>/dev/null | grep -iE "pfn|page_to|dma_addr|>> 12|spa|target" | head -12
echo "=== engine: create() signature ==="
grep -n -A3 "Buffer create(VkDeviceSize size" ~/ps5lm/src/runtime/include/ps5lm/vk.hpp
echo "=== engine: pool auto-sized from host RAM ==="
grep -n "auto-size from /proc/meminfo\|pool_gib <= 0" ~/ps5lm/src/runtime/include/ps5lm/flash_engine.hpp | head -4
echo "=== live host RAM ==="; grep -E "^MemTotal" /proc/meminfo

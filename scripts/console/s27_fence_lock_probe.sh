#!/bin/bash
# S27: read-only. Does Linux inherit a FIRMWARE-programmed fence, and where do the
# GPU's aperture registers actually live? No register is written; no GPU work is
# submitted; the serving engine is not touched.

echo "=== engine processes I must NOT interrupt (read-only ps) ==="
ps -o pid,etime,args= -p 7776,7773,1366 2>/dev/null | tr -s ' '

echo; echo "=== current kernel command line ==="
cat /proc/cmdline

echo; echo "=== E820/firmware memory map: what the FIRMWARE declared ==="
for d in /sys/firmware/memmap/*; do
  [ -d "$d" ] || continue
  b=$(cat "$d/base" 2>/dev/null); t=$(cat "$d/top" 2>/dev/null); ty=$(cat "$d/type" 2>/dev/null)
  printf "  base=%s top=%s type=%s\n" "$b" "$t" "$ty"
done | sed -n "1,40p"

echo; echo "=== kernel log: what the kernel inherited for the GPU/IOMMU ==="
sudo -n dmesg 2>/dev/null | grep -iE "exclusion range|reserved memory|DMA.*reserved|AMD-Vi|iommu.*excl|GART|gart|stolen|ps5_" | sed -n "1,40p"

echo; echo "=== driver source: where the GPU aperture registers actually live ==="
K=~/kbuild/linux/drivers/gpu/drm/amd
grep -rn "mmGARTPRIMBASE\|mmGARTSECBASE\|mmGARTPRIMLIMIT\|mmGARTSECLIMIT\|mmFBC_BASE\|mmFBC_CONFIG\|mmMC_F8x004\|mmMC_F8x008" "$K" 2>/dev/null | sed -n "1,25p"

echo; echo "=== driver source: SOC15 block base mapping a register into BAR0 ==="
grep -rn "SOC15_REG_OFFSET(GC, 0, mmMC_F8x004)\|SOC15_REG_OFFSET(OSS\|OSS000x0080\|OSS000x0084\|OSS000x0088\|OSS000x008c" "$K" 2>/dev/null | sed -n "1,20p"

echo; echo "=== driver source: how a GART/VM entry target address is built ==="
grep -rn "page_to_pfn\|system_address\|>> 12" "$K/amdgpu/amdgpu_gart.c" "$K/amdgpu/amdgpu_gmc.c" "$K/amdgpu/amdgpu_vm.c" 2>/dev/null | sed -n "1,20p"

echo; echo "=== GPU BAR0 raw window: gfx10 OSS/GC offsets ==="
cd /tmp
sudo -n python3 - <<'PY'
import mmap, os
p="/sys/bus/pci/devices/0000:20:00.0/resource0"
fd=os.open(p, os.O_RDONLY)
mm=mmap.mmap(fd, 0, mmap.MAP_SHARED, mmap.PROT_READ)
rd=lambda o: int.from_bytes(mm[o:o+4],"little")
for name,off in [("OSS+0x80 GARTPRIMBASE",0x80),("OSS+0x84 GARTSECBASE",0x84),
                 ("OSS+0x88 GARTPRIMLIMIT",0x88),("OSS+0x8c GARTSECLIMIT",0x8c),
                 ("OSS+0x90 GARTCONFIG",0x90)]:
    print("  %-26s = 0x%08x" % (name, rd(off)))
print("--- GC block window (mmio remap) ---")
for o in (0x00800000,0x00800004,0x00800008,0x0080000c,0x00800010,0x00800014):
    print("  GC+0x%x = 0x%08x" % (o, rd(o)))
PY

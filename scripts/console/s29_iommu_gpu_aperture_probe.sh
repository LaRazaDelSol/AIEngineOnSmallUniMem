#!/bin/bash
# S29: read-only. Two decisive reads that no earlier round captured:
#  (1) the AMD IOMMU exclusion range on the NBIF/IOMMU function (PCI config READS
#      via setpci -s ... .l). If the reclaimed range is inside the exclusion
#      range, GPU DMA to that SPA BYPASSES IOMMU translation and hits the raw
#      fabric address check -- which is the mechanism behind the refusal.
#  (2) the GPU's own BAR0 MMIO, scanned READ-ONLY, to locate the aperture
#      register empirically by value-match instead of guessing offsets.
# No config writes, no MMIO writes, no GPU command submission. The engine keeps
# serving on its own DRM client; nothing here opens a GPU context.

echo "=== PCI functions that are AMD fabric / IOMMU ==="
sudo -n lspci -nn 2>/dev/null | grep -iE "IOMMU|host.*bridge|encryption|SAT|PCI bridge" | sed -n "1,10p"

echo; echo "=== AMD IOMMU exclusion-range registers on 0000:00:00.2 (CONFIG READS ONLY) ==="
for off in 0x1a0 0x1a4 0x1a8 0x1ac 0x1b0 0x1b4 0x1b8 0x1bc; do
  printf "  cfg+0x%x = 0x" "$off"
  sudo -n setpci -s 0000:00:00.2 "$off.l" 2>/dev/null
  echo
done

echo; echo "=== IOMMU exclusion-range / reserved regions the kernel programmed ==="
sudo -n dmesg 2>/dev/null | grep -iE "exclusion|IVRS|IVMD|reserved.*region|DMA.*remap.*enabled|AMD-Vi.*enabled" | sed -n "1,12p"

echo; echo "=== GPU BAR0 scanned READ-ONLY for the programmed aperture limit ==="
cd /tmp
sudo -n python3 - <<'PY'
import mmap, os
p="/sys/bus/pci/devices/0000:20:00.0/resource0"
fd=os.open(p, os.O_RDONLY)
mm=mmap.mmap(fd, 0, mmap.MAP_SHARED, mmap.PROT_READ)
# dmesg proved the driver programmed GART as 0x0 - 0x1FFFFFFF, so the limit
# register must read 0x1FFFFFFF (or its high half 0x1F). Find it by value.
targets={0x1FFFFFFF:"GARTPRIMLIMIT?", 0x00000000:"(zero)", 0x00880000:"FBC/GART base?"}
hits={}
for base in (0x00000000, 0x00800000, 0x01000000, 0x01800000, 0x02000000, 0x02800000, 0x03000000, 0x03800000):
    win=bytes(mm[base:base+0x80000])
    for t,name in targets.items():
        if t==0: continue
        tb=t.to_bytes(4,"little"); i=0; n=0
        while True:
            i=win.find(tb,i)
            if i<0 or n>=3: break
            print("  MATCH %s: BAR0+0x%x = 0x%08x" % (name, base+i, t)); n+=1; i+=4
print("--- GC block window 0x800000..0x800040 (read-only) ---")
for o in range(0x800000,0x800044,4):
    print("  BAR0+0x%x = 0x%08x" % (o, int.from_bytes(mm[o:o+4],"little")))
PY

echo; echo "=== engine untouched (read-only ps) ==="
ps -o pid,etime= -p 7776,7773,1366 | tr -s " "

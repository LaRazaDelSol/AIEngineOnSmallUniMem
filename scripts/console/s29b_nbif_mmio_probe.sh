#!/bin/bash
# S29b: read-only. The AMD IOMMU exclusion-range registers live in the NBIF/IOMMU
# function's MMIO BAR, not its config space (config reads of 0x1a0.. returned
# 0xffffffff = unimplemented). Map that BAR READ-ONLY and dump the pair.
# Also confirm whether the kernel programmed any exclusion range at all.
cd /tmp
echo "=== IOMMU/NBIF function BARs (read-only sysfs reads) ==="
for dev in 0000:00:00.2 0000:00:00.0 0000:00:01.1; do
  echo "-- $dev resource files:"
  for r in resource0 resource1 resource2; do
    sz=$(sudo -n stat -c%s "/sys/bus/pci/devices/$dev/$r" 2>/dev/null)
    [ -n "$sz" ] && printf "   %s size=%s\n" "$r" "$sz"
  done
done

echo; echo "=== dump NBIF/IOMMU MMIO offsets read-only (exclusion-range pair) ==="
sudo -n python3 - <<'PY'
import mmap, os
for dev in ("0000:00:00.2","0000:00:00.0","0000:00:01.1"):
    p="/sys/bus/pci/devices/%s/resource0" % dev
    try:
        fd=os.open(p, os.O_RDONLY)
        mm=mmap.mmap(fd, 0, mmap.MAP_SHARED, mmap.PROT_READ)
    except OSError as e:
        print("  %s: mmap failed: %s" % (dev, e)); continue
    rd=lambda o: int.from_bytes(mm[o:o+4],"little")
    print("  --- %s BAR0 (read-only) ---" % dev)
    for name,off in [("EXCL_BASE_LO",0x1a0),("EXCL_BASE_HI",0x1a4),("EXCL_LIMIT_LO",0x1a8),
                     ("EXCL_LIMIT_HI",0x1ac),("EXCL_CFG",0x1b0),("DEV_ATTR_BASE",0x1b4),
                     ("DEV_ATTR_LIMIT",0x1b8),("PPR",0x1c0)]:
        print("    %-16s +0x%04x = 0x%08x" % (name, off, rd(off)))
    mm.close(); os.close(fd)
PY

echo; echo "=== did the kernel program ANY IOMMU exclusion/reserved region? ==="
sudo -n dmesg 2>/dev/null | grep -iE "exclusion|IVRS|IVMD|DMA.*remap|AMD-Vi" | sed -n "1,10p"
echo "--- IVRS table contents (read-only) ---"
sudo -n ls /sys/firmware/acpi/tables/ 2>/dev/null | grep -iE "^IV" | tr "\n" " "; echo

echo; echo "=== engine untouched ==="; ps -o pid,etime= -p 7776,7773,1366 | tr -s " "

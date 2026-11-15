#!/bin/bash
# S27b: read-only. Is the low-memory fence FIRMWARE-owned (immovable from Linux)?
# Reads the E820 map Linux was handed, the driver params that prove the GPU's
# translation path, and which kernel drivers own the fabric registers.
echo "=== E820 map the FIRMWARE handed to Linux (sudo read) ==="
sudo -n sh -c 'for d in /sys/firmware/memmap/*; do
  [ -d "$d" ] || continue
  printf "  base=%s top=%s type=%s\n" "$(cat $d/base)" "$(cat $d/top)" "$(cat $d/type)"
done' | sed -n "1,30p"

echo; echo "=== which driver owns the fabric / memory-controller registers ==="
K=~/kbuild/linux/drivers
ls "$K/ps5" 2>/dev/null | tr "\n" " "; echo
sudo -n dmesg 2>/dev/null | grep -iE "ps5_|amdtee|psp|iap2|ring buffer.*psp|SMU|umc|ecc.*umc" | sed -n "1,18p"

echo; echo "=== driver params that fix the GPU's translation path (read-only) ==="
for f in /sys/module/amdgpu/parameters/gttsize /sys/module/amdgpu/parameters/ps5_cu_unlock; do
  printf "  %s = %s\n" "$f" "$(sudo -n cat "$f" 2>/dev/null)"
done
sudo -n cat /sys/kernel/debug/dri/0/amdgpu_gtt 2>/dev/null | sed -n "1,4p"
sudo -n ls /sys/kernel/debug/dri/0/ 2>/dev/null | tr "\n" " "; echo

echo; echo "=== zone pages: the discriminator for any future reclaim test ==="
sudo -n dmesg 2>/dev/null | grep -iE "^.*DMA32: DMA|DMA32.*managed|Normal.*managed|DMA.*managed|ps5m2|user-defined physical RAM map|user space of physical RAM|last_pfn|max_pfn" | sed -n "1,14p"

echo; echo "=== the range as the kernel sees it RIGHT NOW ==="
sudo -n grep -iE "01000000-3fffffff|00ffffff|7fd8f000|System RAM|Reserved" /proc/iomem | sed -n "1,12p"
echo "--- /proc/self/pagemap-independent: memmap= present? ---"
grep -c "memmap=" /proc/cmdline

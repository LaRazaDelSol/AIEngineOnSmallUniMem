#!/bin/bash
# Verify that a kernel config has what the engine needs. Console: bash kconfig-check.sh   (reads /boot/config-$(uname -r))
#   bash kconfig-check.sh path/to/.config        (check a config before building a kernel)
# The console runs the ps5-linux kernel (package linux-ps5, built on the console with GCC); the amdgpu.ps5_cu_unlock
# parameter and the PS5 platform drivers come from that tree, not from a stock CachyOS kernel.
CFG=${1:-/boot/config-$(uname -r)}
[ -r "$CFG" ] || { [ -r /proc/config.gz ] && CFG=/proc/config.gz; }
[ -r "$CFG" ] || { echo "no kernel config found ($CFG)"; exit 2; }
get() { case "$CFG" in *.gz) zcat "$CFG";; *) cat "$CFG";; esac | grep -E "^CONFIG_$1=" | cut -d= -f2; }
rc=0
need() {  # name, accepted regex, why
  v=$(get "$1")
  if echo "${v:-n}" | grep -qE "^($2)$"; then printf '  ok    %-34s %s\n' "$1=${v}" "$3"
  else printf '  FAIL  %-34s want %s: %s\n' "$1=${v:-n}" "$2" "$3"; rc=1; fi
}
echo "kernel config: $CFG"
need HUGETLBFS            y      "2 MiB hugetlb expert slab (GPU 2 MiB TLB entries, memmap returned)"
need HUGETLB_PAGE_OPTIMIZE_VMEMMAP y "frees the slab's struct pages (+~1.5% pool)"
need UDMABUF              y      "zero-copy slab: io_uring registered buffers = the GPU's dma-buf"
need IO_URING             y      "expert streamer (READ_FIXED) and the n-gram row batch"
need BLK_DEV_NVME         "y|m"  "M.2 expert store"
need DRM_AMDGPU           "y|m"  "RADV GFX1013 compute"
need COMPACTION           y      "hugepage reservation after the dense weights (compact_memory retry)"
need KEXEC                y      "soft reboot is the only safe reboot on this console"
need USB_UAS              "y|m"  "USB 10 Gbps mirror devices (--pack-mirror)"
need PREEMPT_DYNAMIC      y      "preempt=full on the cmdline"
[ "$(get KEXEC_HANDOVER)" = y ] && echo "  note  KEXEC_HANDOVER=y: boot with kho=off or 1.5 GiB becomes unpinnable scratch (S14)"
exit $rc

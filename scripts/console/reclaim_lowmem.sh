#!/bin/bash
# Console-side: give Linux the 1,008 MiB of RAM at 0x01000000-0x3FFFFFFF that the loader marks reserved (S25).
#
# What is there (measured 2026-11-10, recon/s25/reserved-ram-*.txt): ordinary DRAM holding the frozen memory image of
# the PlayStation OS that was running when the loader took over (its strings are readable). Over 2 minutes none of
# 4,032 sampled pages changed, and five all-zero pages took a write and a restore. Linux is the SVM host (kvm_amd
# loads, drivers/ps5/svm.c runs VMRUN itself), so no hypervisor is executing from it. The reservation comes from one
# hard-coded e820 entry in ps5-linux-loader (source/prepare_resume.c), not from the hardware.
#
# How: `memmap=exactmap` replaces the firmware memory map with the list below = the map this console reports
# (dmesg "BIOS-e820"), with that one entry turned into RAM. Everything else is kept as it is.
#
# TESTED 2026-11-10: DOES NOT WORK. The CPU can use that memory, the GPU cannot: with the pool's pages in that range the
# kernel logged about 2,600 machine-check lines (addresses all inside the range), vkQueueSubmit failed and the served engine
# produced garbage until the change was reverted (docs/HARDWARE-DECONSTRUCTION.md section 3). `apply` now refuses
# unless called as `apply --i-know-the-gpu-cannot-use-it`. Kept for the record and for CPU-only experiments.
# Original note: if something does still read that image (nothing found so far), the console hangs or reboots some time
# after the next soft reboot. Recovery: hold the power button, cold boot. A cold boot takes its command line from the
# loader stick, never from /boot/efi/cmdline.txt, so it comes back with the old map. Have someone near the console.
#
#   sudo bash reclaim_lowmem.sh status     what the running kernel uses, what the file says
#   sudo bash reclaim_lowmem.sh apply      adds the map to /boot/efi/cmdline.txt (backup kept); then `sudo reboot`
#   sudo bash reclaim_lowmem.sh revert     removes it
# Expected after apply + reboot: MemTotal 14,7xx -> 15,7xx MB, the engine's pool about 500 slots larger.
set -euo pipefail
F=/boot/efi/cmdline.txt
MAP='memmap=exactmap memmap=0x6f000@0x1000 memmap=0x5ff00000@0x100000 memmap=0xc00000$0x60000000 memmap=0x1c00000@0x60c00000 memmap=0x2029000$0x62800000 memmap=0x1b1a7000@0x64829000 memmap=0x397000$0x7f9d0000 memmap=0x8000$0x7fd67000 memmap=0x20000#0x7fd6f000 memmap=0x44671000$0x7fd8f000 memmap=0x10700000$0xd0000000 memmap=0x8000000$0xf0000000 memmap=0x350000000@0x100000000 memmap=0x20000000$0x450000000 memmap=0xf300000@0x470000000 memmap=0xd00000$0x47f300000'
# The list is only valid for the firmware map it was derived from: refuse on any other.
check_map() {  # /proc/iomem of the running kernel (the boot log may have rotated out of the ring buffer)
  for r in "01000000-3fffffff : Reserved" "40000000-5fffffff : System RAM" "60c00000-627fffff : System RAM" "64829000-7f9cffff : System RAM"            "100000000-44fffffff : System RAM" "450000000-46fffffff : Reserved" "470000000-47f2fffff : System RAM"; do
    grep -q "^$r" /proc/iomem || { echo "firmware map differs from the one this list was made for (missing: $r): not applying" >&2; exit 1; }
  done
}
case "${1:-status}" in
  status)
    grep -q 'memmap=exactmap' /proc/cmdline && echo "running kernel: low memory RECLAIMED" || echo "running kernel: firmware map (1,008 MiB reserved)"
    grep -q 'memmap=exactmap' $F && echo "$F: reclaim set (applies at the next soft reboot)" || echo "$F: firmware map"
    grep -E 'MemTotal|MemAvailable' /proc/meminfo ;;
  apply)
    [ "${2:-}" = "--i-know-the-gpu-cannot-use-it" ] || { echo "refused: the GPU cannot access that range (machine checks, 2026-11-10); see the header" >&2; exit 1; }
    grep -q 'memmap=exactmap' /proc/cmdline && { echo "already running with the reclaimed map"; exit 0; }
    check_map
    grep -q 'memmap=exactmap' $F && { echo "already in $F"; exit 0; }
    cp $F $F.pre-reclaim.$(date +%Y%m%d-%H%M%S)
    printf '%s %s\n' "$(tr -d '\n' < $F)" "$MAP" > $F.new && mv $F.new $F
    echo "written. Soft reboot to apply: sudo reboot   (cold boot = recovery, see the header of this script)" ;;
  revert)
    sed -i -E 's/ ?memmap=[^ ]+//g' $F && echo "removed from $F" ;;
  *) echo "usage: $0 status|apply|revert" >&2; exit 2 ;;
esac

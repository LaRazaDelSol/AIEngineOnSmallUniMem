#!/usr/bin/env bash
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
S(){ echo; echo "===== $1 ====="; }
S HOST
hostname; uptime
head -6 /etc/os-release 2>/dev/null
S KERNEL
uname -a
S CPU
lscpu 2>/dev/null | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|MHz|NUMA"
grep -m1 flags /proc/cpuinfo | tr ' ' '\n' | grep -E "^(avx2|avx512f|fma|sha_ni)$" | sort -u
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null
S MEM
free -h
grep -E "^(MemTotal|MemFree|MemAvailable|Cached|SwapTotal):" /proc/meminfo
S GPU_PCI
lspci 2>/dev/null | grep -iE "vga|3d|display|audio|ethernet|nvme|usb" || echo "lspci unavailable"
S DRM
ls -l /dev/dri 2>/dev/null
for c in /sys/class/drm/card[0-9]; do [ -e "$c" ] && echo "$c -> $(readlink -f $c/device)"; done
lsmod 2>/dev/null | grep -E "^(amdgpu|snd_hda|xhci)" | head
S VRAM_GTT
for f in /sys/class/drm/card[0-9]/device/mem_info_vram_total /sys/class/drm/card[0-9]/device/mem_info_gtt_total /sys/class/drm/card[0-9]/device/mem_info_visible_vram_total; do [ -e "$f" ] && echo "$f = $(cat $f)"; done
(dmesg 2>/dev/null || journalctl -kq 2>/dev/null) | grep -iE "amdgpu|gtt|vram|oberon" | head -15
S BLOCK
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,MODEL 2>/dev/null
S DF
df -h 2>/dev/null | grep -vE "tmpfs|overlay|squashfs|efivarfs|tracefs|debugfs|bpf|cgroup" | head -12
S ROOTFS
findmnt / 2>/dev/null
S NVME
nvme list 2>/dev/null || echo "nvme-cli not installed"
S SERVICES
systemctl get-default
systemctl --no-pager --no-legend --state=enabled list-unit-files 2>/dev/null | head -30
S TOP_RSS
ps aux --sort=-rss 2>/dev/null | head -14
S LISTEN
ss -tln 2>/dev/null | head -10
S NET
ip -br addr 2>/dev/null | head -6
S TOOLS
for t in curl wget python3 gcc clang make git fio hdparm nvme vulkaninfo clinfo glxinfo sensors pacman flatpak steam gamescope docker; do command -v $t >/dev/null 2>&1 && echo "have: $t"; done
ldd --version 2>/dev/null | head -1
S GROUPS
id
S THERMAL
for z in /sys/class/thermal/thermal*; do [ -e "$z/temp" ] && echo "$(basename $z): $(cat $z/temp) $(cat $z/type 2>/dev/null)"; done 2>/dev/null | head -10
S SWAP
cat /proc/swaps 2>/dev/null | head -5
S NVME_DMESG
(dmesg 2>/dev/null || journalctl -kq 2>/dev/null) | grep -iE "nvme" | head -20
S DONE

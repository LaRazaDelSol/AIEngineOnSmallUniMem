#!/usr/bin/env bash
# PS5 next-boot diagnostic — run FIRST after power-on, before anything else.
# Every probe is timeout-wrapped so nothing can wedge the SSH session.
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export SYSTEMD_PAGER= SYSTEMD_COLORS=0
T(){ timeout 20 "$@" 2>&1; }
S(){ echo; echo "===== $1 ====="; }
echo "diag-start $(date +%H:%M:%S) uptime:$(cat /proc/uptime 2>/dev/null)"
S PSI_IO_MEMORY_CPU
cat /proc/pressure/io 2>/dev/null
cat /proc/pressure/memory 2>/dev/null
cat /proc/pressure/cpu 2>/dev/null
S KERNEL_CMDLINE
uname -a
cat /proc/cmdline
S DMESG_IO_ERRORS
T dmesg | grep -iE "nvme|ata[0-9]|usb|I/O error|timeout|reset|ext4|btrfs|xfs" | tail -60
S BLOCK_DEVICES
T lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,MODEL,SERIAL
T findmnt /
T findmnt /var
T df -h | grep -vE "tmpfs|overlay|squashfs|efivarfs"
S USB_DETAILS
for d in /sys/bus/usb/devices/*/speed; do [ -e "$d" ] && echo "${d%/speed}: $(cat $d) Mbps"; done
T bash -c 'for i in /sys/bus/usb/devices/*:*; do [ -d "$i/driver" ] && echo "$(basename $i): $(readlink $i/driver)"; done' 2>/dev/null | sed 's|.*/||'
for d in /sys/bus/usb/devices/*/power/control; do [ -e "$d" ] && echo "${d%/power/control}: $(cat $d)"; done
S ROOT_BENCH
ROOTSRC=$(timeout 10 findmnt -n -o SOURCE / 2>/dev/null)
echo "root source: ${ROOTSRC:-unknown}"
ROOTBASE=${ROOTSRC##*/}
ROOTDEV=$(echo "$ROOTBASE" | sed -E 's/[0-9]+$//; s/p$//')
if [ -e "/dev/$ROOTDEV" ]; then
  echo "read-bench /dev/$ROOTDEV (512 MiB, direct I/O, read-only):"
  timeout 45 dd if=/dev/$ROOTDEV of=/dev/null bs=1M count=512 iflag=direct 2>&1 | tail -4
else
  echo "could not resolve root block device"
fi
S NVME_HEALTH
for n in /sys/class/nvme/nvme*; do
  [ -e "$n" ] || continue
  echo "$n model=$(cat $n/model 2>/dev/null) fw=$(cat $n/firmware_rev 2>/dev/null)"
  for h in $n/hwmon*/temp1_input; do [ -e "$h" ] && echo "  temp: $(( $(cat $h) / 1000 )) C"; done
  T nvme smart-log "$(basename $n)" 2>/dev/null | grep -iE "temperature|percentage_used|media_errors|critical_warning|unsafe_shutdowns|data_units"
done
S D_STATE_TASKS
T ps aux | awk '$8 ~ /D/ {print}' | head -15
S MEM
T free -h
S TOP_RSS
T ps aux --sort=-rss | head -15
S TOP_CPU
T ps aux | sort -rnk3 | head -10
S GPU_PCI_DRM
T lspci | grep -iE "vga|3d|display|audio|ethernet|nvme|usb"
ls -l /dev/dri 2>/dev/null
for f in /sys/class/drm/card[0-9]/device/mem_info_vram_total /sys/class/drm/card[0-9]/device/mem_info_gtt_total; do [ -e "$f" ] && echo "$f = $(cat $f)"; done
S INTERNAL_SSD_PRESENT
for d in /dev/nvme*; do [ -e "$d" ] && echo "$d"; done
S SERVICES
systemctl get-default
T systemctl --no-pager --no-legend --state=enabled list-unit-files | head -30
S TOOLS
for t in curl wget python3 gcc clang make git fio hdparm nvme smartctl vulkaninfo clinfo sensors pacman flatpak steam gamescope docker; do command -v $t >/dev/null 2>&1 && echo "have: $t"; done
ldd --version 2>/dev/null | head -1
id
S THERMAL_ZONES
for z in /sys/class/thermal/thermal*; do [ -e "$z/temp" ] && echo "$(basename $z): $(cat $z/temp) $(cat $z/type 2>/dev/null)"; done 2>/dev/null | head -10
S DONE
echo "diag-end $(date +%H:%M:%S)"

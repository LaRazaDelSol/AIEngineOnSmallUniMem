#!/usr/bin/env bash
# PS5 Gate 0/1 measurement battery. Runs ON the PS5 under Linux (ps5-linux-loader / ps5-linux image).
# Read-only everywhere except a scratch file under $SCRATCH (deleted at exit; N7 store persists
# unless N7_KEEP=0). Emits: lines "K<TAB>key<TAB>value" then, if python3 exists, a JSON document
# between JSON_START/JSON_END markers. Driven from the PC by tools/ps5probe/probe.py over ssh
# ("bash -s" on stdin), but safe to run by hand.
set -u
SCRATCH_DIR="${SCRATCH_DIR:-/var/tmp/ps5probe}"   # used if no data mount found
FIO_SECONDS="${FIO_SECONDS:-15}"
# W2 #14: the old 2 GiB default was SLC-cache-bait (reads served from tens of GiB of SLC,
# not NAND). Default 100 GiB (>3x any plausible OEM SLC) and auto-shrink to free-20 GiB.
FIO_SIZE_MB="${FIO_SIZE_MB:-102400}"
FIO_SETTLE_S="${FIO_SETTLE_S:-45}"                # drain SLC/DRAM caches between write and read phases
G1_TARGET="${G1_TARGET:-}"                       # explicit mount for G1; empty = auto (largest free)
N6_SIZE_MB="${N6_SIZE_MB:-8192}"                 # N6 scratch files per device (was 1.5 GiB = SLC bait)
N7="${N7:-0}"                                    # 1 = run the page-cache expert replay (gate N7)
N7_SIZE_GB="${N7_SIZE_GB:-40}"                   # synthetic expert-store size (GiB)
N7_SECONDS="${N7_SECONDS:-90}"                   # steady-state replay duration
N7_CAP_MB="${N7_CAP_MB:-0}"                      # cache cap; 0 = min(35% RAM, 3072M) — W4: PS5's free RAM buys a 2-3 GiB hot pool, so 35% of 16 GiB (5.6G) OVERSTATES the engine's cache
N7_KEEP="${N7_KEEP:-1}"                          # 1 = keep the store between runs for reuse (W2 #19)
KV="$(mktemp)"
# cleanup must NOT nuke a persistent N7 store (W2 #19): delete the fio file + empty dir only
trap 'rm -f "${RUN_SCRATCH}/fio.bin" "$KV"; rmdir "${RUN_SCRATCH}" "$N6_A_DIR" "$N6_B_DIR" 2>/dev/null' EXIT
RUN_SCRATCH=""; RUN_SCRATCH_2=""; N6_A_DIR=""; N6_B_DIR=""

kv() { printf 'K\t%s\t%s\n' "$1" "${2:-}" >>"$KV"; }
# W2 #22: tabs inside values truncated the JSON emitter's split; sanitize to spaces first.
# "~" still encodes newlines (emitter re-expands); a value containing a literal ~ is mangled — audit text only.
runq() { eval "${*}" 2>/dev/null | head -c 4000 | tr '\t' ' ' | tr '\n\r' '~~'; }
sh() { kv "$1" "$(runq "${@:2}")"; }
# bwread: robust to fio's stdout noise ("note: ...queue depth..." notes precede the JSON on
# psync engines) and to key spelling (bw_avg vs bw). Skip everything before the first '{'.
bwread() { python3 -c "
import json,sys
s=sys.stdin.read()
i=s.find('{')
try:
    r=json.loads(s[i:])['jobs'][0]['read']
    v=r.get('bw_avg', r.get('bw', 0)) or 0
    print(int(v/1024))
except Exception:
    pass" <<<"$1" 2>/dev/null; }
# fio silently caps queue depth to 1 with the psync engine (the "note:" line); if libaio is
# compiled in, use it for the read jobs so iodepth actually applies. WSL dry-run 2026-10-21.
# W11 #3: `fio --engines=libaio` is not a valid fio option, so this always failed and
# G1 silently ran the qd1-capped fallback while the console fio HAS libaio/io_uring
# (verified 2026-10-27). Probe the engine properly; keep the label honest.
if command -v fio >/dev/null 2>&1 && fio --enghelp=libaio >/dev/null 2>&1; then IO_R="--ioengine=libaio"; FIO_LABEL=libaio; else IO_R=""; FIO_LABEL=psync-iodepth-capped-to-1; fi
kv fio_ioengine "$FIO_LABEL"

# ---------- system / identity ----------
kv os "$(uname -srm)"
sh kernel_cmdline 'cat /proc/cmdline'
sh vram_txt 'for f in /mnt/*/vram.txt /boot/vram.txt; do [ -e "$f" ] && cat "$f"; done'
sh os_release 'grep PRETTY /etc/os-release'
sh cpu_model 'grep -m1 -i name /proc/cpuinfo'
kv cpu_threads "$(nproc 2>/dev/null)"

# ---------- GPU identity: settles gfx1013-vs-Oberon (adversarial A4) ----------
sh pci_vga 'lspci -nn 2>/dev/null | grep -i -E "vga|display|processing"'
for d in /sys/class/drm/card*/device; do
  [ -e "$d/device" ] || continue
  kv drm_card "$(basename $(dirname $d))"
  kv device_id "$(cat $d/device 2>/dev/null)"
  kv drm_revision "$(cat $d/revision 2>/dev/null)"
  kv asic_type "$(cat $d/asic_type 2>/dev/null)"
  kv vram_total "$(cat $d/mem_info_vram_total 2>/dev/null)"
  kv vram_visible "$(cat $d/mem_info_vram_visible 2>/dev/null)"
  kv gtt_total "$(cat $d/mem_info_gtt_total 2>/dev/null)"
done
if command -v vulkaninfo >/dev/null 2>&1; then
  sh vk_summary 'vulkaninfo --summary'
  kv vk_ext_host "$(vulkaninfo --show-all 2>/dev/null | grep -c 'VK_EXT_external_memory_host')"
  sh vk_heaps 'vulkaninfo 2>/dev/null | grep -B1 -A4 "heapFlags" | head -40'
else
  kv vk_ext_host "no-vulkaninfo"
fi

# ---------- GPU memory knobs (plan gate 0 / adversarial A7) ----------
kv ttm_pages_limit "$(cat /sys/module/ttm/parameters/pages_limit 2>/dev/null)"
kv ttm_page_pool_size "$(cat /sys/module/ttm/parameters/page_pool_size 2>/dev/null)"
kv amdgpu_gttsize "$(cat /sys/module/amdgpu/parameters/gttsize 2>/dev/null)"
sh meminfo 'awk "/MemTotal|MemAvailable/{print \$2 \$1}" /proc/meminfo'
sh tmpfiles_gpu_conf 'grep -r . /etc/tmpfiles.d/ /usr/lib/tmpfiles.d/ 2>/dev/null | grep -i -E "ttm|gtt" | head -4'
sh modprobe_gpu_conf 'grep -r . /etc/modprobe.d/ 2>/dev/null | grep -i -E "ttm|amdgpu" | head -4'

# ---------- clock / boost state (W2 #15: record, don't change; the RUNBOOK orders
# `ps5_control --boost on --fan on` BEFORE this script; post-hoc attribution needs this) ----------
kv scaling_governor "$(cat /sys/devices/system/cpu/cpufreq/policy*/scaling_governor 2>/dev/null | tr '\n' ' ')"
kv cpu_boost "$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null)"
kv ps5_control "$(command -v ps5_control 2>/dev/null) $(ls /usr/local/bin/ps5_control* 2>/dev/null | head -1)"
sh ps5_control_state 'command -v ps5_control >/dev/null && ps5_control --status 2>&1 | head -6'
kv gpu_busy_percent "$(cat /sys/class/drm/card0/device/gpu_busy_percent 2>/dev/null)"

# ---------- storage: find the M.2 (adversarial A6 topology) ----------
sh lsblk_json 'lsblk -b -J -o NAME,SIZE,TYPE,MOUNTPOINT,MODEL 2>/dev/null | head -c 3800'
sh nvme_list 'nvme list'
sh block_devs 'for b in /sys/block/nvme*n* /sys/block/sd?; do [ -e "$b/size" ] || continue; printf "%s:%s:%s|" "$(basename $b)" "$(cat $b/size)" "$(cat $b/device/model 2>/dev/null | tr -s " ")"; done'
sh mounts 'findmnt -rno TARGET,SOURCE,FSTYPE 2>/dev/null | head -25'

# ---------- thermal / clocks ----------
sh hwmon 'for f in /sys/class/hwmon/*/temp*_input /sys/class/thermal/thermal_zone*/temp; do [ -e "$f" ] && printf "%s=%s|" "$f" "$(cat $f)"; done | head -c 1500'
sh clocks 'printf "cpu=%s " "$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq 2>/dev/null)"; printf "gpu_busy=%s " "$(cat /sys/class/drm/card0/device/gpu_busy_percent 2>/dev/null)"'

# ---------- Gate 1: storage battery (adversarial A2; W2 #17 explicit device targeting) ----------
# G1_TARGET=/path overrides; default = largest-free non-root mount, but the DEVICE it lands on is
# always kv'd so a number measured on an M.2 can never be silently attributed to the internal SSD.
# W9-01: plain `df -o` is not a coreutils option; `--output=` is. Errors must be loud, not fallback-silent.
if [ -n "$G1_TARGET" ] && [ -d "$G1_TARGET" ]; then
  MNT="$G1_TARGET"; kv g1_target "explicit:$G1_TARGET"
else
  [ -n "$G1_TARGET" ] && kv g1_target "explicit-target-missing($G1_TARGET):falling-back"
  MNT="$(df -B1 --output=target,avail -x tmpfs -x devtmpfs -x squashfs 2>/dev/null | awk 'NR>1{if($2+0>a){a=$2+0;t=$1}} END{print t}')"
  [ -n "${MNT:-}" ] || MNT="$SCRATCH_DIR"
  [ -n "$G1_TARGET" ] || kv g1_target "auto-largest-free"
fi
[ -d "$MNT" ] || MNT="$SCRATCH_DIR"
RUN_SCRATCH="$MNT/ps5probe-scratch"
# W11 #2: mkdir -p SUCCEEDS on an existing but unwritable dir (S8's root-owned
# /ps5probe-scratch made every steam-run silently resolve here and fail with
# PermissionError on the fio create and the N7 store open). Test WRITABILITY,
# not mkdir success. Canonical steam-store: /var/tmp/ps5probe-scratch.
mkdir -p "$RUN_SCRATCH" 2>/dev/null
if [ ! -w "$RUN_SCRATCH" ]; then
  RUN_SCRATCH="/var/tmp/ps5probe-scratch"
  mkdir -p "$RUN_SCRATCH" 2>/dev/null
fi
if [ ! -w "$RUN_SCRATCH" ]; then echo "FATAL: no writable scratch dir (tried $MNT/ps5probe-scratch and /var/tmp/ps5probe-scratch)" >&2; exit 1; fi
kv scratch_mount "$MNT"
# resolve the backing block device + model of $MNT (W2 #17)
G1_SRC="$(findmnt -n -o SOURCE "$MNT" 2>/dev/null)"
# W9-03: `-k` is not an lsblk option; PKNAME gives the parent disk of a partition
# directly (verified on a loop-partition). Fallback: mount IS a whole device.
G1_REAL="$(lsblk -n -o PKNAME -p "$G1_SRC" 2>/dev/null | head -1)"
[ -n "${G1_REAL:-}" ] || G1_REAL="$G1_SRC"
kv g1_device "${G1_REAL:-unknown}"
if [ -n "${G1_REAL:-}" ]; then
  B="$(basename "$G1_REAL")"
  kv g1_device_model "$(cat /sys/block/$B/device/model 2>/dev/null | tr -s ' ')"
  kv g1_rotational "$(cat /sys/block/$B/queue/rotational 2>/dev/null)"
  PCIADDR="$(cat /sys/block/$B/device/address 2>/dev/null || cat /sys/block/$B/device/device/address 2>/dev/null)"
  [ -n "$PCIADDR" ] && sh pcie_link "lspci -vvv -s $PCIADDR 2>/dev/null | grep -A1 -E 'LnkCap|LnkSta' | head -8"
fi
# shrink the scratch to available room, with 20 GiB headroom (never fill a mount)
AVAIL_MB=$(df -B1M --output=avail "$MNT" 2>/dev/null | tail -1 | tr -dc '0-9'); AVAIL_MB=${AVAIL_MB:-0}
if [ "$AVAIL_MB" -gt 0 ] && [ "$FIO_SIZE_MB" -gt $((AVAIL_MB - 20480)) ]; then
  FIO_SIZE_MB=$((AVAIL_MB - 20480)); [ "$FIO_SIZE_MB" -lt 4096 ] && FIO_SIZE_MB=4096
  kv fio_size_note "shrunk to ${FIO_SIZE_MB}M (free-space guard; SLC-honesty reduced)"
fi
kv fio_size_mb "$FIO_SIZE_MB"
F="$RUN_SCRATCH/fio.bin"

# S11: N7_ONLY=1 skips the G1/N6 fio battery (fast N7-only re-runs — the ×3 fixed-token
# sweeps); G0 identity kvs above still emit for provenance.
if [ "${N7_ONLY:-0}" != "1" ]; then
if command -v fio >/dev/null 2>&1; then
  kv fio_ver "$(fio --version 2>/dev/null | head -1)"
  # S11: drive-temp provenance IN the artifact — a heat-soaked drive silently throttles
  # (the 16:23 battery's G1 rows ran ~25-45% low after 45 min of sweep I/O + 2x100G
  # creates; only recoverable because the temp was on the console). nvme smart-log
  # needs root; the hwmon node is world-readable.
  sh g1_nvme_temp_start 'for d in /sys/class/hwmon/hwmon*; do if [ -e "$d/temp1_input" ] && grep -qs nvme "$d/name"; then echo "$(cat $d/name) $(cat $d/temp1_input)"; fi; done'
  fio --name=create --rw=write --bs=1M --size=${FIO_SIZE_MB}M --direct=1 --filename="$F" --output-format=json >/dev/null 2>&1
  kv settle_seconds "$FIO_SETTLE_S"
  sleep "$FIO_SETTLE_S"   # let SLC->NAND folding and DRAM cache flush settle before reading
  r=$(fio --name=seq --rw=read --bs=1M --iodepth=32 $IO_R --direct=1 --runtime=$FIO_SECONDS --time_based --size=${FIO_SIZE_MB}M --filename="$F" --output-format=json 2>/dev/null); kv fio_seq1m_mb_s "$(bwread "$r")"
  # S11 (user directive): the engine-shaped rand row is SLC-SENSITIVE — on the FIO_SIZE_MB
  # scratch it read 4798 "99% of seq" while the S10 100G chase read 3299-3351. Run it on a
  # persistent 100G SLC-honest scratch (kept for reuse like the N7 store; G1_RAND_KEEP=0
  # deletes). Cold via fadvise so the settle protocol applies to NAND, not SLC.
  # bs=3M approximates the engine's 2,764,800 B expert block (exact value 2700k is not a
  # fio-friendly pow2 for direct IO continuity; 3M keeps comparability with past runs, ~8% high)
  G1_RAND_SIZE_MB="${G1_RAND_SIZE_MB:-102400}"
  AVAIL2_MB=$(df -B1M --output=avail "$MNT" 2>/dev/null | tail -1 | tr -dc '0-9'); AVAIL2_MB=${AVAIL2_MB:-0}
  if [ "$AVAIL2_MB" -gt 0 ] && [ "$G1_RAND_SIZE_MB" -gt $((AVAIL2_MB - 20480)) ]; then
    G1_RAND_SIZE_MB=$((AVAIL2_MB - 20480)); [ "$G1_RAND_SIZE_MB" -lt 4096 ] && G1_RAND_SIZE_MB=$FIO_SIZE_MB
  fi
  RF="$RUN_SCRATCH/g1-rand.bin"
  rsz=$(stat -c %s "$RF" 2>/dev/null || echo 0)
  if [ "$rsz" -lt $((G1_RAND_SIZE_MB * 1048576)) ]; then
    kv g1_rand_scratch "creating ${G1_RAND_SIZE_MB}M SLC-honest rand scratch (one-time)"
    fio --name=randcreate --rw=write --bs=1M --size=${G1_RAND_SIZE_MB}M --direct=1 --filename="$RF" >/dev/null 2>&1
    sleep "$FIO_SETTLE_S"   # S10 chase protocol: a fresh 100G write leaves SLC residue — drain before reading
  else
    kv g1_rand_scratch "reused ${G1_RAND_SIZE_MB}M SLC-honest rand scratch"
  fi
  if [ -e "$RF" ] && [ "$(stat -c %s "$RF")" -ge $((G1_RAND_SIZE_MB * 1048576)) ]; then
    python3 - "$RF" <<'PY' 2>/dev/null
import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)
PY
    r=$(fio --name=rand3 --rw=randread --bs=3M --iodepth=8 $IO_R --direct=1 --runtime=$FIO_SECONDS --time_based --size=${G1_RAND_SIZE_MB}M --filename="$RF" --output-format=json 2>/dev/null); kv fio_rand3m_mb_s "$(bwread "$r")"
    kv g1_rand_scratch_mb "$G1_RAND_SIZE_MB"
  else
    # fallback: old behavior on the (too-small) scratch; the kv says so loudly
    r=$(fio --name=rand3 --rw=randread --bs=3M --iodepth=8 $IO_R --direct=1 --runtime=$FIO_SECONDS --time_based --size=${FIO_SIZE_MB}M --filename="$F" --output-format=json 2>/dev/null); kv fio_rand3m_mb_s "$(bwread "$r")"
    kv g1_rand_scratch_mb "0 (100G create failed; SLC-honesty NOT guaranteed)"
  fi
  [ "${G1_RAND_KEEP:-1}" = "0" ] && rm -f "$RF"
  r=$(fio --name=rand4 --rw=randread --bs=4k --iodepth=64 $IO_R --direct=1 --runtime=8 --time_based --size=${FIO_SIZE_MB}M --filename="$F" --output-format=json 2>/dev/null); kv fio_rand4k_mb_s "$(bwread "$r")"
  sh g1_nvme_temp_end 'for d in /sys/class/hwmon/hwmon*; do if [ -e "$d/temp1_input" ] && grep -qs nvme "$d/name"; then echo "$(cat $d/name) $(cat $d/temp1_input)"; fi; done'
else
  kv fio_ver "absent"
fi
# dd fallback / cross-check (independent of fio): direct seq read of the scratch file
if [ ! -e "$F" ]; then dd if=/dev/zero of="$F" bs=1M count=$FIO_SIZE_MB status=none 2>/dev/null; fi
sz=$(stat -c %s "$F" 2>/dev/null || echo 0)
ddline=$(dd if="$F" of=/dev/null bs=1M iflag=direct 2>&1 >/dev/null | grep -oiE '[0-9.]+ ?(k|M|G)B/s' | tail -1)
kv dd_seq_direct "$ddline"
# buffered regime: warm the page cache with an 8 GiB TAIL window (a whole 100 GiB file
# cannot cache in 16 GiB RAM — W2 #14), then time a full re-read ("trust the OS" number)
TAIL_MB=8192; [ "$FIO_SIZE_MB" -lt "$TAIL_MB" ] && TAIL_MB=$FIO_SIZE_MB
SKIP=$((FIO_SIZE_MB - TAIL_MB))
dd if="$F" bs=1M skip=$SKIP count=$TAIL_MB of=/dev/null status=none 2>/dev/null
t0=$(date +%s.%N); dd if="$F" bs=1M skip=$SKIP count=$TAIL_MB of=/dev/null status=none 2>/dev/null; t1=$(date +%s.%N)
kv buffered_window_mb "$TAIL_MB"
kv buffered_reread_mb_s "$(python3 -c "print(int($TAIL_MB/max(0.01,($t1-$t0))))" 2>/dev/null || echo 0)"
rm -f "$F"

# ---------- N6: concurrent dual-device reads (frontier gate N6, stripe ceiling) ----------
# If Linux sees TWO NVMe-backed mounts (internal 825GB + M.2), read both at once and
# report the aggregate — the app-level-striping bandwidth the 40-60 tok/s ledger needs.
# NOTE (S2): no M.2 installed on the target console; N6 is a purchase branch, not an asset.
dev_of() {  # mountpoint -> nvme namespace name, or empty
  local src real
  src="$(findmnt -n -o SOURCE "$1" 2>/dev/null)"; [ -n "$src" ] || return 0
  real="$(lsblk -n -o NAME -p "$src" 2>/dev/null | tail -1)"
  case "$real" in /dev/nvme*) basename "$real" ;; *) return 0 ;; esac
}
N6_A=""; N6_B=""
for m in $(df -B1 --output=target -x tmpfs -x devtmpfs -x squashfs 2>/dev/null | awk 'NR>1' | sort -u); do
  d="$(dev_of "$m")"; [ -n "$d" ] || continue
  avail=$(df -BG --output=avail "$m" 2>/dev/null | tail -1 | tr -dc '0-9')
  [ "${avail:-0}" -ge 20 ] || continue
  if [ -z "$N6_A" ]; then N6_A="$m:$d"
  elif [ "$d" != "${N6_A#*:}" ]; then N6_B="$m:$d"
  fi
done
if [ -n "$N6_A" ] && [ -n "$N6_B" ] && command -v fio >/dev/null 2>&1; then
  kv n6_devices "${N6_A#*:}+${N6_B#*:}"
  N6_A_DIR="${N6_A%%:*}/ps5probe-scratch"; N6_B_DIR="${N6_B%%:*}/ps5probe-scratch"
  mkdir -p "$N6_A_DIR" "$N6_B_DIR" 2>/dev/null
  AF="$N6_A_DIR/n6.bin"; BF="$N6_B_DIR/n6.bin"
  # W2 #14/#16: 8 GiB files (past SLC on most drives), single-writer create (numjobs=2 on
  # one filename just wrote the same offsets twice)
  fio --name=ca --rw=write --bs=2M --size=${N6_SIZE_MB}M --direct=1 --filename="$AF" >/dev/null 2>&1
  fio --name=cb --rw=write --bs=2M --size=${N6_SIZE_MB}M --direct=1 --filename="$BF" >/dev/null 2>&1
  sleep "$FIO_SETTLE_S"
  fio --name=a --rw=read --bs=1M --iodepth=32 $IO_R --direct=1 --runtime=$FIO_SECONDS --time_based --filename="$AF" --output-format=json >/tmp/ps5probe-n6a.json 2>/dev/null &
  pa=$!
  fio --name=b --rw=read --bs=1M --iodepth=32 $IO_R --direct=1 --runtime=$FIO_SECONDS --time_based --filename="$BF" --output-format=json >/tmp/ps5probe-n6b.json 2>/dev/null &
  pb=$!
  wait $pa; wait $pb
  a="$(bwread "$(cat /tmp/ps5probe-n6a.json 2>/dev/null)")"; b="$(bwread "$(cat /tmp/ps5probe-n6b.json 2>/dev/null)")"
  kv n6_a_mb_s "${a:-0}"; kv n6_b_mb_s "${b:-0}"; kv n6_aggregate_mb_s "$(( ${a:-0} + ${b:-0} ))"
  rm -f /tmp/ps5probe-n6a.json /tmp/ps5probe-n6b.json "$AF" "$BF"
  rmdir "$N6_A_DIR" "$N6_B_DIR" 2>/dev/null
else
  kv n6_devices "not-two-nvme-mounts (A='${N6_A:-none}' B='${N6_B:-none}' fio=$(command -v fio >/dev/null && echo yes || echo no))"
fi
kv n7_only "${N7_ONLY:-0}"
fi  # N7_ONLY guard (S11): G1/N6 skipped, straight to N7

# ---------- N7: page-cache expert replay under synthetic Zipf routing (gate N7) ----------
# Emulates the engine's cache access pattern with NO engine: an N GiB expert store,
# 2,764,800 B chunks (= the engine's real expert-block size, W4 recompute), 480 draws/
# token, Zipf(s=1.1) WITHIN each layer's slots — W2 #20: draws are now top-10 DISTINCT
# (Gumbel top-k = weighted sampling WITHOUT replacement; a real router never repeats an
# expert within a token, iid draws were inflating hits ~+5-10 pts near the 80% gate).
# Cache capped via cgroup (memory.high) to proxy the PS5's hot-expert pool. This is a
# MEASUREMENT of the hit-vs-capacity curve, not a pass/fail gate (>=80% is at/above the
# published static ceiling 0.79). Replace with engine traces when the engine exists.
if [ "$N7" = "1" ]; then
  # S11 geometry override: N7_SLOT_BYTES (+ N7_SPL, default 512) builds the REAL pack
  # geometry — e.g. 1.66 bpw: 512 x 1,019,904 B = 25,065,345,024 B (23.35 GiB) EXACT, so
  # the replay derives spl=512 from the store itself. The file is geometry-named so it
  # never clobbers the canonical 40G/2.6MB-slot n7.bin.
  if [ -n "${N7_SLOT_BYTES:-}" ]; then
    N7_SPL="${N7_SPL:-512}"
    NF="$RUN_SCRATCH/n7-${N7_SLOT_BYTES}x${N7_SPL}.bin"
    NF_WANT=$((48 * N7_SPL * N7_SLOT_BYTES))
    N7_SIZE_GB=$(( (NF_WANT + 1073741824 - 1) / 1073741824 ))
    kv n7_geometry "slot=${N7_SLOT_BYTES}B spl=${N7_SPL} store=${NF_WANT}B exact"
  else
    NF="$RUN_SCRATCH/n7.bin"
    NF_WANT=$((N7_SIZE_GB * 1073741824))
  fi
  free_gb=$(df -BG --output=avail "$MNT" 2>/dev/null | tail -1 | tr -dc '0-9'); free_gb=${free_gb:-0}
  [ "$N7_SIZE_GB" -gt $((free_gb - 15)) ] && N7_SIZE_GB=$((free_gb - 15))
  if [ "$N7_SIZE_GB" -ge 12 ] && command -v python3 >/dev/null 2>&1; then
    have=$(stat -c %s "$NF" 2>/dev/null || echo 0)
    if [ "$have" -lt "$NF_WANT" ]; then
      kv n7_create "writing ${NF_WANT}B store (one-time)"
      # W2 #19: single writer (numjobs=2 on one filename double-wrote the same offsets: 2x time, 2x wear)
      fio --name=n7w --rw=write --bs=4M --size="$NF_WANT" --direct=1 --filename="$NF" >/dev/null 2>&1
      have=$(stat -c %s "$NF" 2>/dev/null || echo 0)
    else
      kv n7_create "reused existing store (want ${NF_WANT}B)"
    fi
    if [ "$have" -lt "$NF_WANT" ]; then
      kv n7_ran "0 (store create failed: fio $(command -v fio >/dev/null 2>&1 && echo present-but-failed || echo ABSENT); have=${have}B want=${NF_WANT}B)"
    else
    if [ "$N7_CAP_MB" = "0" ]; then
      # W4: PS5's realistic hot pool is 2-3 GiB of otherwise-free RAM; cap = min(35% RAM, 3 GiB)
      N7_CAP_MB=$(awk '/MemTotal/{c=int($2*0.35/1024); if (c>3072) c=3072; print c}' /proc/meminfo)
    fi
    PYF="$(mktemp)"
    cat >"$PYF" <<'PYEOF'
import os, sys, mmap, ctypes, math, time, random
fn, secs, s_par = sys.argv[1], float(sys.argv[2]), 1.1
kvf = sys.argv[3]
# S11: argv[4] = fixed-token target (0 = legacy time-boxed mode — the 90s box gave
# 263-296 tokens depending on the cap, confounding hit% with draw count); argv[5]/[6]
# = slot-bytes / slots-per-layer overrides for real-pack-geometry stores (0 = derive).
tok_target = int(sys.argv[4]) if len(sys.argv) > 4 else 0
ch_override = int(sys.argv[5]) if len(sys.argv) > 5 else 0
spl_override = int(sys.argv[6]) if len(sys.argv) > 6 else 0
f = os.open(fn, os.O_RDWR)   # RDWR so the mapping is writable at the mmap layer (W2 #21 context):
# ctypes.from_buffer() requires a writable buffer, and Python >=3.11 enforces it -- an O_RDONLY/
# PROT_READ mapping crashes the replay with "underlying buffer is not writable" (WSL dry-run
# 2026-10-21). The engine of this probe only reads (mincore + pread); the file is never written.
sz = os.fstat(f).st_size
# W11 #4: evict cross-run page-cache residue. Pages cached by a previous run's
# (now-dead) cgroup are reparented and stay resident; mincore() counts them as
# hits no matter THIS run's cap (verified 2026-10-27: after one 8192M run, caps
# 4096-16384 all read ~72.7% because ~13 GiB of residue was serving hits).
# Cold store per run makes the cap the ONLY pool.
os.posix_fadvise(f, 0, 0, os.POSIX_FADV_DONTNEED)
CH = ch_override if ch_override > 0 else 2764800  # default: the engine's actual expert-block size (480 reads/token, W4 recompute) = 675 pages exactly
chunks = sz // CH
L = 48
while chunks // L < 64 and L > 1:
    L //= 2
spl = spl_override if spl_override > 0 else max(chunks // L, 1)   # slots per layer; override = real pack geometry (512 x 1,019,904 for 1.66 bpw)
if spl_override > 0 and chunks < spl_override * L:
    spl = max(chunks // L, 1)   # store too small for the override — derive honestly (visible in n7_slots_per_layer)
slots = spl * L
logw = [-s_par * math.log(i + 1) for i in range(spl)]
mm = mmap.mmap(f, 0, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
libc = ctypes.CDLL("libc.so.6", use_errno=True)
vec = (ctypes.c_ubyte * 8)()
PAGE = 4096
def resident(off):
    # NOTE: `addr & ~PAGE + PAGE` parses as `addr & (-4097 + 4096)` = `addr` (W2 #21). DO NOT
    # "fix" the precedence: passing the unaligned address works because Linux mincore()
    # down-aligns internally, and 3 MiB/675-page chunks are page-aligned so first-page
    # residency == all-or-nothing chunk residency. A precedence "fix" would check the NEXT page.
    addr = ctypes.addressof(ctypes.c_char.from_buffer(mm, off))
    if libc.mincore(ctypes.c_void_p(addr & ~PAGE + PAGE), ctypes.c_size_t(PAGE), vec) != 0:
        return False
    return vec[0] & 1
K = 10
def layer_picks(layer):
    # top-K DISTINCT slots, weighted sampling WITHOUT replacement (Gumbel top-k over
    # log-weights): key_i = log w_i - log(-log u_i); the K largest keys are a weighted
    # sample of size K with no duplicates — what a real top-10-of-512 router does.
    r = random.random
    base = layer * spl
    keys = [((lw - math.log(-math.log(r() or 1e-12))), i) for i, lw in enumerate(logw)]
    keys.sort(reverse=True)
    return [base + keys[j][1] for j in range(min(K, spl))]
hit = miss = 0; miss_bytes = 0; tokens = 0; warm = 30; t0 = time.time()
def _not_done():
    # S11: fixed-token mode draws an IDENTICAL count per run (warmup excluded) so hit%
    # varies only with the cap; legacy mode stays time-boxed.
    return (tokens - warm) < tok_target if tok_target > 0 else (time.time() - t0 < secs)
while _not_done():
    for layer in range(L):
        for off in (p * CH for p in layer_picks(layer)):
            if resident(off):
                hit += 1
            else:
                try:
                    os.pread(f, CH, off); miss += 1; miss_bytes += CH
                except OSError:
                    miss += 1
    tokens += 1
    if tokens == warm:
        hit = miss = 0; miss_bytes = 0; t0 = time.time()
el = max(time.time() - t0, 0.01)
try:  # W9-10: read THIS scope's cgroup, not the root's (system-wide) usage
    _cg = open("/proc/self/cgroup").read().strip().split("::")[-1]
    cur = open("/sys/fs/cgroup" + _cg + "/memory.current").read().strip()
except (OSError, IndexError):
    cur = ""
def _mem_available_mb():
    try:
        for _l in open("/proc/meminfo"):
            if _l.startswith("MemAvailable"):
                return round(int(_l.split()[1]) / 1024)
    except (OSError, IndexError, ValueError):
        pass
    return -1
with open(kvf, "a") as out:
    # S10: record system MemAvailable IN the artifact (user directive 2026-10-27):
    # free memory at replay time must come from the run, never hand-copied into
    # docs. System-wide (cgroup-independent) — the number a cap must be honored
    # against; a cap above this was never actually granted.
    for k, v in [("n7_hit_pct", round(100.0 * hit / max(hit + miss, 1), 1)),
                 ("n7_tokens", tokens),
                 ("n7_store_gb", round(sz / 2**30, 1)), ("n7_miss_gb", round(miss_bytes / 2**30, 2)),
                 ("n7_ssd_mb_s", round(miss_bytes / 2**20 / el, 0)),
                 ("n7_slots", slots), ("n7_slots_per_layer", spl), ("n7_chunk_bytes", CH),
                 ("n7_mem_available_mb", _mem_available_mb()),
                 ("n7_tokens_target", tok_target),
                 ("n7_sampling", "gumbel-topk-without-replacement"),
                 ("n7_cgroup_current", cur)]:
        out.write("K\t%s\t%s\n" % (k, v))
PYEOF
    N7ERR="$(mktemp)"
    # W11: the old guard `[ -w /sys/fs/cgroup ]` misfires for the steam user even though
    # systemd-run --scope -p MemoryHigh=... works (verified on-console 2026-10-27: memory.high
    # reads 104857600 inside a 100M scope). Guard by DOING the thing, not by proxy-perms.
    if command -v systemd-run >/dev/null 2>&1 && systemd-run --quiet --scope -p MemoryHigh=1M true 2>/dev/null; then
      kv n7_mode "cgroup memory.high=${N7_CAP_MB}M${N7_TOKENS:+; fixed ${N7_TOKENS} tokens}${N7_SLOT_BYTES:+; slot=${N7_SLOT_BYTES}B spl=${N7_SPL:-512}}"
      kv n7_cache_mb "$N7_CAP_MB"
      systemd-run --quiet --scope -p MemoryHigh="${N7_CAP_MB}M" -p MemoryMax="$((N7_CAP_MB + 1024))M" \
        python3 "$PYF" "$NF" "$N7_SECONDS" "$KV" "${N7_TOKENS:-0}" "${N7_SLOT_BYTES:-0}" "${N7_SPL:-0}" 2>"$N7ERR"
      N7RC=$?
    else
      kv n7_mode "uncapped (no cgroup access; hits reflect full-RAM LRU)${N7_TOKENS:+; fixed ${N7_TOKENS} tokens}"
      kv n7_cache_mb "$(awk '/MemTotal/{c=int($2*0.35/1024); if (c>3072) c=3072; print c}' /proc/meminfo)"
      python3 "$PYF" "$NF" "$N7_SECONDS" "$KV" "${N7_TOKENS:-0}" "${N7_SLOT_BYTES:-0}" "${N7_SPL:-0}" 2>"$N7ERR"
      N7RC=$?
    fi
    if [ "$N7RC" = "0" ]; then kv n7_ran "1"; else kv n7_ran "0 (replay rc=$N7RC: $(tail -c 300 "$N7ERR" | tr '\t\n\r' ' ~~'))"; fi
    kv n7_zipf_s "1.1"
    rm -f "$PYF" "$N7ERR"
    if [ "$N7_KEEP" = "0" ]; then rm -f "$NF"; fi
    fi
  else
    kv n7_ran "0 (needs >=12G free on $MNT and python3; free=${free_gb}G)"
  fi
fi

# ---------- emit ----------
cat "$KV"
if command -v python3 >/dev/null 2>&1; then
  echo JSON_START
  python3 - "$KV" <<'PY'
import json, sys
d = {}
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    if not line.startswith("K\t"):
        continue
    parts = line.rstrip("\n").split("\t")
    k, v = parts[1], (parts[2] if len(parts) > 2 else "")
    v = v.replace("~", "\n").strip()
    if v.isdigit():
        v = int(v)
    d[k] = v
print(json.dumps(d))
PY
  echo JSON_END
fi
